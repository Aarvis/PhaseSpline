"""Transfer contiguous checkpoint segments from a robot episode to a human episode.

The alignment uses full-rate DINOv3 top-camera embeddings and endpoint-anchored
dynamic time warping. Internal checkpoint boundaries are mapped with separate
before/after neighborhoods, then projected to a strictly increasing target
boundary vector so the output contains no gaps or overlaps.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "dtw_top_rgb.yaml"


@dataclass(frozen=True)
class AlignmentResult:
    path: list[tuple[int, int]]
    accumulated_cost: float
    normalized_cost: float
    distance_matrix: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transfer robot checkpoint labels to a human episode using full-rate DINOv3 DTW."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing target checkpoints.json.")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or "paths" not in config:
        raise ValueError(f"Invalid DTW configuration: {path}")
    return config


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return data


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def l2_normalize_rows(array: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.maximum(norms, 1e-12)


def temporal_smooth(array: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return l2_normalize_rows(array.astype(np.float32, copy=False))
    if window % 2 == 0:
        raise ValueError("temporal_smoothing_window must be odd")
    radius = window // 2
    padded = np.pad(array, ((radius, radius), (0, 0)), mode="edge")
    cumulative = np.vstack(
        [np.zeros((1, array.shape[1]), dtype=np.float64), np.cumsum(padded, axis=0, dtype=np.float64)]
    )
    smoothed = (cumulative[window:] - cumulative[:-window]) / float(window)
    return l2_normalize_rows(smoothed.astype(np.float32))


def load_embeddings(path: Path, key: str, smoothing_window: int) -> tuple[np.ndarray, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        if key not in archive:
            raise KeyError(f"{path} does not contain embedding key {key!r}")
        embeddings = np.asarray(archive[key], dtype=np.float32)
        frame_indices = np.asarray(archive.get("frame_indices", np.arange(embeddings.shape[0])))
        fps_value = archive.get("fps")
        fps = float(np.asarray(fps_value).reshape(-1)[0]) if fps_value is not None else None
    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise ValueError(f"Embeddings must have shape [frames, dimensions], got {embeddings.shape} from {path}")
    if not np.isfinite(embeddings).all():
        raise ValueError(f"Non-finite embeddings found in {path}")
    expected_indices = np.arange(embeddings.shape[0], dtype=frame_indices.dtype)
    if frame_indices.shape != expected_indices.shape or not np.array_equal(frame_indices, expected_indices):
        raise ValueError(f"frame_indices in {path} must be contiguous and zero based")
    embeddings = temporal_smooth(embeddings, smoothing_window)
    return embeddings, {
        "frames": int(embeddings.shape[0]),
        "dimension": int(embeddings.shape[1]),
        "fps": fps,
    }


def validate_reference_checkpoints(data: dict[str, Any], reference_frames: int) -> list[dict[str, Any]]:
    segments = data.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ValueError("Reference checkpoints must contain a non-empty segments list")
    labels = data.get("labels")
    expected_start = 0
    seen_labels: list[str] = []
    for index, segment in enumerate(segments):
        start = int(segment["start_frame"])
        end = int(segment["end_frame_exclusive"])
        inclusive = int(segment["end_frame_inclusive"])
        num_frames = int(segment["num_frames"])
        if int(segment["segment_id"]) != index:
            raise ValueError(f"Reference segment {index} has a non-sequential segment_id")
        if start != expected_start:
            raise ValueError(f"Reference segment {index} starts at {start}; expected {expected_start}")
        if end <= start:
            raise ValueError(f"Reference segment {index} is empty")
        if inclusive != end - 1 or num_frames != end - start:
            raise ValueError(f"Reference segment {index} has inconsistent frame fields")
        expected_start = end
        seen_labels.append(str(segment["label"]))
    if expected_start != reference_frames:
        raise ValueError(
            f"Reference checkpoints end at {expected_start}, but embeddings contain {reference_frames} frames"
        )
    if labels is not None and list(labels) != seen_labels:
        raise ValueError("Reference labels do not exactly match segment label order")
    return segments


def cosine_distance_matrix(reference: np.ndarray, target: np.ndarray, temporal_prior_weight: float) -> np.ndarray:
    distance = 1.0 - reference @ target.T
    if temporal_prior_weight > 0:
        ref_phase = np.linspace(0.0, 1.0, reference.shape[0], dtype=np.float32)[:, None]
        target_phase = np.linspace(0.0, 1.0, target.shape[0], dtype=np.float32)[None, :]
        distance = distance + temporal_prior_weight * np.abs(ref_phase - target_phase)
    return np.maximum(distance, 0.0).astype(np.float32, copy=False)


def dynamic_time_warp(
    distance: np.ndarray,
    diagonal_penalty: float,
    horizontal_penalty: float,
    vertical_penalty: float,
) -> AlignmentResult:
    n_reference, n_target = distance.shape
    cost = np.full((n_reference, n_target), np.inf, dtype=np.float64)
    back = np.full((n_reference, n_target), -1, dtype=np.int8)
    cost[0, 0] = float(distance[0, 0])

    for i in tqdm(range(n_reference), desc="DTW rows", unit="frame"):
        for j in range(n_target):
            if i == 0 and j == 0:
                continue
            candidates: list[tuple[float, int]] = []
            if i > 0 and j > 0:
                candidates.append((cost[i - 1, j - 1] + diagonal_penalty, 2))
            if i > 0:
                candidates.append((cost[i - 1, j] + vertical_penalty, 0))
            if j > 0:
                candidates.append((cost[i, j - 1] + horizontal_penalty, 1))
            best_cost, move = min(candidates, key=lambda item: item[0])
            cost[i, j] = float(distance[i, j]) + best_cost
            back[i, j] = move

    i, j = n_reference - 1, n_target - 1
    path: list[tuple[int, int]] = []
    while True:
        path.append((i, j))
        if i == 0 and j == 0:
            break
        move = int(back[i, j])
        if move == 0:
            i -= 1
        elif move == 1:
            j -= 1
        elif move == 2:
            i -= 1
            j -= 1
        else:
            raise RuntimeError(f"Broken DTW backpointer at ({i}, {j})")
    path.reverse()
    total_cost = float(cost[-1, -1])
    return AlignmentResult(
        path=path,
        accumulated_cost=total_cost,
        normalized_cost=total_cost / max(1, len(path)),
        distance_matrix=distance,
    )


def mapping_from_path(path: Iterable[tuple[int, int]]) -> dict[int, list[int]]:
    mapping: dict[int, list[int]] = {}
    for reference_frame, target_frame in path:
        mapping.setdefault(reference_frame, []).append(target_frame)
    return mapping


def median_target_vote(mapping: dict[int, list[int]], reference_frame: int) -> float | None:
    targets = mapping.get(reference_frame)
    return None if not targets else float(np.median(np.asarray(targets, dtype=np.float32)))


def round_half_up(value: float) -> int:
    return int(math.floor(value + 0.5))


def transfer_boundary_two_sided(
    reference_boundary: int,
    mapping: dict[int, list[int]],
    reference_frames: int,
    target_frames: int,
    side_window: int,
) -> tuple[int, dict[str, Any]]:
    if not 0 < reference_boundary < reference_frames:
        raise ValueError("Only internal reference boundaries can be transferred")
    left_indices = range(max(0, reference_boundary - side_window), reference_boundary)
    right_indices = range(reference_boundary, min(reference_frames, reference_boundary + side_window))
    left_votes = [vote for index in left_indices if (vote := median_target_vote(mapping, index)) is not None]
    right_votes = [vote for index in right_indices if (vote := median_target_vote(mapping, index)) is not None]

    if not left_votes or not right_votes:
        fallback_indices = range(
            max(0, reference_boundary - side_window),
            min(reference_frames, reference_boundary + side_window + 1),
        )
        fallback_votes = [
            vote for index in fallback_indices if (vote := median_target_vote(mapping, index)) is not None
        ]
        if not fallback_votes:
            raw_boundary = round_half_up(reference_boundary * target_frames / reference_frames)
            method = "duration_ratio_fallback"
            left_anchor = None
            right_anchor = None
        else:
            raw_boundary = round_half_up(float(np.median(fallback_votes)))
            method = "symmetric_median_fallback"
            left_anchor = float(np.median(fallback_votes))
            right_anchor = left_anchor
    else:
        left_anchor = float(np.median(left_votes))
        right_anchor = float(np.median(right_votes))
        raw_boundary = round_half_up((left_anchor + right_anchor) / 2.0)
        method = "two_sided_median"

    raw_boundary = int(np.clip(raw_boundary, 1, target_frames - 1))
    diagnostics = {
        "reference_boundary": reference_boundary,
        "method": method,
        "left_reference_frames": list(left_indices),
        "right_reference_frames": list(right_indices),
        "left_target_votes": left_votes,
        "right_target_votes": right_votes,
        "left_target_anchor": left_anchor,
        "right_target_anchor": right_anchor,
        "raw_target_boundary": raw_boundary,
    }
    return raw_boundary, diagnostics


def enforce_contiguous_boundaries(
    raw_internal: list[int], target_frames: int, min_segment_frames: int
) -> list[int]:
    segment_count = len(raw_internal) + 1
    if target_frames < segment_count * min_segment_frames:
        raise ValueError(
            f"Cannot fit {segment_count} segments of at least {min_segment_frames} frames into {target_frames} frames"
        )
    boundaries = [0]
    for index, value in enumerate(raw_internal, start=1):
        minimum = boundaries[-1] + min_segment_frames
        remaining_segments = segment_count - index
        maximum = target_frames - remaining_segments * min_segment_frames
        boundaries.append(int(np.clip(round(value), minimum, maximum)))
    boundaries.append(target_frames)
    return boundaries


def boundary_similarity(
    diagnostic: dict[str, Any], raw_cosine_distance: np.ndarray, path: list[tuple[int, int]]
) -> dict[str, float]:
    reference_frames = set(diagnostic["left_reference_frames"] + diagnostic["right_reference_frames"])
    values = [float(raw_cosine_distance[i, j]) for i, j in path if i in reference_frames]
    mean_distance = float(np.mean(values)) if values else 2.0
    mean_similarity = 1.0 - mean_distance
    confidence = float(np.clip(1.0 - mean_distance / 2.0, 0.0, 1.0))
    return {
        "local_mean_cosine_distance": mean_distance,
        "local_mean_cosine_similarity": mean_similarity,
        "confidence": confidence,
    }


def segments_from_boundaries(
    reference_segments: list[dict[str, Any]], boundaries: list[int], boundary_diagnostics: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    boundary_confidence = {
        int(item["boundary_index"]): float(item["confidence"]) for item in boundary_diagnostics
    }
    output: list[dict[str, Any]] = []
    for index, reference_segment in enumerate(reference_segments):
        start = boundaries[index]
        end = boundaries[index + 1]
        adjacent = [
            boundary_confidence[key]
            for key in (index, index + 1)
            if key in boundary_confidence
        ]
        output.append(
            {
                "segment_id": index,
                "label": str(reference_segment["label"]),
                "start_frame": start,
                "end_frame_exclusive": end,
                "end_frame_inclusive": end - 1,
                "num_frames": end - start,
                "confidence": float(np.mean(adjacent)) if adjacent else 1.0,
                "notes": "Checkpoint label transferred from the simulated robot reference using top-camera DINOv3 DTW.",
                "reference_segment": {
                    "start_frame": int(reference_segment["start_frame"]),
                    "end_frame_exclusive": int(reference_segment["end_frame_exclusive"]),
                },
            }
        )
    return output


def validate_output_segments(segments: list[dict[str, Any]], target_frames: int) -> dict[str, Any]:
    errors: list[str] = []
    expected_start = 0
    total = 0
    for index, segment in enumerate(segments):
        start = int(segment["start_frame"])
        end = int(segment["end_frame_exclusive"])
        if int(segment["segment_id"]) != index:
            errors.append(f"segment {index}: bad segment_id")
        if start != expected_start:
            errors.append(f"segment {index}: expected start {expected_start}, got {start}")
        if end <= start:
            errors.append(f"segment {index}: empty or reversed")
        if int(segment["end_frame_inclusive"]) != end - 1:
            errors.append(f"segment {index}: bad inclusive end")
        if int(segment["num_frames"]) != end - start:
            errors.append(f"segment {index}: bad num_frames")
        expected_start = end
        total += end - start
    if expected_start != target_frames:
        errors.append(f"coverage ends at {expected_start}, expected {target_frames}")
    if total != target_frames:
        errors.append(f"segments cover {total} frames, expected {target_frames}")
    return {
        "valid": not errors,
        "errors": errors,
        "coverage_is_contiguous": not any("expected start" in error for error in errors),
        "coverage_is_complete": expected_start == target_frames and total == target_frames,
        "num_labeled_frames": total,
    }


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def save_diagnostics(
    directory: Path,
    result: AlignmentResult,
    raw_cosine_distance: np.ndarray,
    mapping: dict[int, list[int]],
    boundary_diagnostics: list[dict[str, Any]],
    config: dict[str, Any],
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    path_array = np.asarray(result.path, dtype=np.int32)
    np.savez_compressed(
        directory / "dtw_alignment.npz",
        reference_frame=path_array[:, 0],
        target_frame=path_array[:, 1],
        path_cosine_distance=np.asarray(
            [raw_cosine_distance[i, j] for i, j in result.path], dtype=np.float32
        ),
        distance_matrix=result.distance_matrix,
    )

    if config["output"].get("save_alignment_jsonl", True):
        with (directory / "dtw_alignment.jsonl").open("w", encoding="utf-8") as handle:
            for step, (reference_frame, target_frame) in enumerate(result.path):
                cosine_distance = float(raw_cosine_distance[reference_frame, target_frame])
                handle.write(
                    json.dumps(
                        {
                            "dtw_step": step,
                            "reference_frame": reference_frame,
                            "target_frame": target_frame,
                            "cosine_distance": cosine_distance,
                            "cosine_similarity": 1.0 - cosine_distance,
                        }
                    )
                    + "\n"
                )

    mapping_rows = []
    for reference_frame in sorted(mapping):
        targets = mapping[reference_frame]
        mapping_rows.append(
            {
                "reference_frame": reference_frame,
                "target_frame_min": min(targets),
                "target_frame_median": float(np.median(targets)),
                "target_frame_max": max(targets),
                "num_path_matches": len(targets),
            }
        )
    write_csv(
        directory / "frame_mapping.csv",
        ["reference_frame", "target_frame_min", "target_frame_median", "target_frame_max", "num_path_matches"],
        mapping_rows,
    )
    write_csv(
        directory / "checkpoint_boundary_transfer.csv",
        [
            "boundary_index",
            "reference_boundary",
            "left_target_anchor",
            "right_target_anchor",
            "raw_target_boundary",
            "adjusted_target_boundary",
            "adjustment_frames",
            "local_mean_cosine_distance",
            "local_mean_cosine_similarity",
            "confidence",
            "method",
        ],
        boundary_diagnostics,
    )

    if config["output"].get("save_alignment_plot", True):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(10, 7))
        image = axis.imshow(raw_cosine_distance, origin="lower", aspect="auto", cmap="magma_r")
        axis.plot(path_array[:, 1], path_array[:, 0], color="#7CFC00", linewidth=1.2, label="DTW path")
        for item in boundary_diagnostics:
            axis.axhline(item["reference_boundary"], color="white", alpha=0.22, linewidth=0.7)
            axis.axvline(item["adjusted_target_boundary"], color="#00D7FF", alpha=0.3, linewidth=0.7)
        axis.set_xlabel("Human target frame")
        axis.set_ylabel("Robot reference frame")
        axis.set_title("Top-camera DINOv3 cosine distance and DTW path")
        axis.legend(loc="upper left")
        figure.colorbar(image, ax=axis, label="1 - cosine similarity")
        figure.tight_layout()
        figure.savefig(directory / "dtw_alignment.png", dpi=int(config["output"].get("plot_dpi", 180)))
        plt.close(figure)


def main() -> int:
    args = parse_args()
    config = load_config(args.config.resolve())
    paths = {name: project_path(value) for name, value in config["paths"].items()}
    output_checkpoints = paths["output_checkpoints"]
    if output_checkpoints.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output_checkpoints}. Pass --overwrite to replace it.")

    alignment_cfg = config["alignment"]
    smoothing_window = int(alignment_cfg.get("temporal_smoothing_window", 1))
    embedding_key = str(alignment_cfg.get("embedding_key", "embeddings_l2"))
    print("Loading full-rate DINOv3 embeddings")
    reference_embeddings, reference_embedding_meta = load_embeddings(
        paths["reference_embeddings"], embedding_key, smoothing_window
    )
    target_embeddings, target_embedding_meta = load_embeddings(
        paths["target_embeddings"], embedding_key, smoothing_window
    )
    if reference_embeddings.shape[1] != target_embeddings.shape[1]:
        raise ValueError("Reference and target embedding dimensions differ")

    reference_checkpoints = read_json(paths["reference_checkpoints"])
    reference_segments = validate_reference_checkpoints(
        reference_checkpoints, reference_embeddings.shape[0]
    )
    reference_metadata = read_json(paths["reference_metadata"])
    target_metadata = read_json(paths["target_metadata"])

    print(
        f"Aligning robot {reference_embeddings.shape[0]} frames -> "
        f"human {target_embeddings.shape[0]} frames ({reference_embeddings.shape[1]}D)"
    )
    temporal_prior_weight = float(alignment_cfg.get("temporal_prior_weight", 0.0))
    distance = cosine_distance_matrix(reference_embeddings, target_embeddings, temporal_prior_weight)
    raw_cosine_distance = cosine_distance_matrix(reference_embeddings, target_embeddings, 0.0)
    result = dynamic_time_warp(
        distance,
        diagonal_penalty=float(alignment_cfg.get("diagonal_penalty", 0.0)),
        horizontal_penalty=float(alignment_cfg.get("horizontal_penalty", 0.0)),
        vertical_penalty=float(alignment_cfg.get("vertical_penalty", 0.0)),
    )
    mapping = mapping_from_path(result.path)

    boundary_cfg = config["boundary_transfer"]
    side_window = int(boundary_cfg.get("side_window_reference_frames", 3))
    raw_internal: list[int] = []
    boundary_diagnostics: list[dict[str, Any]] = []
    for boundary_index, segment in enumerate(
        tqdm(reference_segments[:-1], desc="Transfer boundaries", unit="boundary"), start=1
    ):
        raw_boundary, diagnostic = transfer_boundary_two_sided(
            int(segment["end_frame_exclusive"]),
            mapping,
            reference_embeddings.shape[0],
            target_embeddings.shape[0],
            side_window,
        )
        diagnostic["boundary_index"] = boundary_index
        diagnostic.update(boundary_similarity(diagnostic, raw_cosine_distance, result.path))
        raw_internal.append(raw_boundary)
        boundary_diagnostics.append(diagnostic)

    boundaries = enforce_contiguous_boundaries(
        raw_internal,
        target_embeddings.shape[0],
        int(boundary_cfg.get("min_target_segment_frames", 1)),
    )
    for index, diagnostic in enumerate(boundary_diagnostics, start=1):
        adjusted = boundaries[index]
        diagnostic["adjusted_target_boundary"] = adjusted
        diagnostic["adjustment_frames"] = adjusted - int(diagnostic["raw_target_boundary"])

    segments = segments_from_boundaries(reference_segments, boundaries, boundary_diagnostics)
    validation = validate_output_segments(segments, target_embeddings.shape[0])
    if not validation["valid"]:
        raise RuntimeError("Generated checkpoint validation failed: " + "; ".join(validation["errors"]))

    labels = [segment["label"] for segment in segments]
    target_frames = target_embeddings.shape[0]
    payload = {
        "schema_version": "1.0",
        "description": "Robot checkpoint labels transferred to a human pants-folding episode using full-rate top-camera DINOv3 DTW.",
        "template_status": "dtw_transferred_requires_review",
        "task": {
            "name": "human pants folding",
            "garment_type": "pants",
            "annotation_scope": "episode_000000 top RGB video",
        },
        "source_file": str(paths["target_metadata"].parent / target_metadata["exported_parquet"]),
        "video_stem": paths["target_video"].stem,
        "video_file": str(paths["target_video"]),
        "embedding_file": str(paths["target_embeddings"]),
        "reference": {
            "role": "simulated robot reference",
            "video_file": str(paths["reference_video"]),
            "embedding_file": str(paths["reference_embeddings"]),
            "checkpoints_file": str(paths["reference_checkpoints"]),
            "frame_count": reference_embeddings.shape[0],
            "fps": reference_embedding_meta["fps"],
        },
        "frame_indexing": {
            "base": 0,
            "start_frame": 0,
            "end_frame_exclusive": target_frames,
            "end_frame_inclusive": target_frames - 1,
            "note": "Zero-based decoded-video frame indices. Segments use [start_frame, end_frame_exclusive).",
            "annotation_sampling": {
                "type": "full_rate_every_frame",
                "original_frame_stride": 1,
                "original_frame_offset": 0,
                "annotation_frame_k_maps_to_original_frame": "k",
            },
        },
        "summary": {
            "num_labels": len(labels),
            "num_unique_labels": len(set(labels)),
            "num_annotated_segments": len(segments),
            "first_frame": 0,
            "last_frame_inclusive": target_frames - 1,
            "num_labeled_frames": target_frames,
            "coverage_is_contiguous": validation["coverage_is_contiguous"],
            "coverage_is_complete": validation["coverage_is_complete"],
        },
        "method": {
            "name": "single_reference_full_rate_dinov3_dtw",
            "direction": "simulated robot reference -> human target",
            "embedding_key": embedding_key,
            "embedding_dimension": target_embedding_meta["dimension"],
            "distance": "1 - cosine similarity over L2-normalized DINOv3 embeddings",
            "temporal_smoothing_window": smoothing_window,
            "temporal_prior_weight": temporal_prior_weight,
            "dtw_endpoint_constraint": "path is anchored at both sequence endpoints",
            "boundary_mapping": "two-sided median of per-reference-frame target votes",
            "side_window_reference_frames": side_window,
            "continuity": "all segments are generated from one strictly increasing shared boundary vector",
            "accumulated_cost": result.accumulated_cost,
            "normalized_cost": result.normalized_cost,
            "path_length": len(result.path),
        },
        "labels": labels,
        "segments": segments,
        "boundaries": {
            "reference": [0]
            + [int(segment["end_frame_exclusive"]) for segment in reference_segments],
            "target_raw": [0] + raw_internal + [target_frames],
            "target_contiguous": boundaries,
            "diagnostics": boundary_diagnostics,
        },
        "video_metadata": {
            "frame_count": target_frames,
            "fps": target_embedding_meta["fps"] or target_metadata.get("fps"),
            "width": target_metadata.get("frame_size", {}).get("width"),
            "height": target_metadata.get("frame_size", {}).get("height"),
            "duration_seconds": target_metadata.get("duration_seconds"),
        },
        "validation": validation,
        "diagnostics_dir": str(paths["diagnostics_dir"]),
    }

    save_diagnostics(
        paths["diagnostics_dir"],
        result,
        raw_cosine_distance,
        mapping,
        boundary_diagnostics,
        config,
    )
    atomic_write_json(output_checkpoints, payload)
    atomic_write_json(paths["diagnostics_dir"] / "dtw_summary.json", payload)
    print(f"Wrote transferred checkpoints: {output_checkpoints}")
    print(f"Diagnostics: {paths['diagnostics_dir']}")
    print(f"Validation: {target_frames} frames, {len(segments)} contiguous segments, no gaps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
