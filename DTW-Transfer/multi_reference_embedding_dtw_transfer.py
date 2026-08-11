from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency
    yaml = None

try:
    from numba import njit
except ImportError:  # pragma: no cover - optional dependency
    njit = None


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "lehome_multi_reference_frame_embeddings.yaml"
EPISODE_PATTERN = re.compile(r"episode_(\d{6})")


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    dataset_root: Path
    embeddings_root: Path
    annotations_root: Path
    output_root: Path
    video_prefix: str
    default_category_ranges: dict[str, tuple[int, int]]


@dataclass(frozen=True)
class AnnotationRecord:
    episode_index: int
    garment_type: str
    checkpoint_path: Path
    labels: list[str]
    task: dict[str, Any]
    raw: dict[str, Any]


@dataclass(frozen=True)
class EpisodeEmbedding:
    episode_index: int
    embedding_path: Path
    frame_count: int
    dimension: int
    features: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Transfer temporal checkpoint labels by running multi-reference DTW over precomputed "
            "2048-D per-frame embeddings, with garment-specific episode pools."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset", choices=["sim", "human"], required=True)
    parser.add_argument("--garment", action="append", default=[], help="Repeat to limit to named garment types.")
    parser.add_argument("--exclude-garment", action="append", default=[], help="Repeat to exclude garment types.")
    parser.add_argument(
        "--reference-manifest",
        type=Path,
        help=(
            "Optional JSON/YAML manifest with per-garment reference episodes and candidate episode pools. "
            "See manifests/example_reference_manifest.json."
        ),
    )
    parser.add_argument(
        "--candidate-policy",
        choices=["require_manifest", "contiguous_hull_from_references", "manual_annotations_only"],
        default=None,
        help=(
            "How to discover non-reference target episodes when the manifest does not define a candidate pool. "
            "Default comes from config."
        ),
    )
    parser.add_argument(
        "--candidate-range",
        action="append",
        default=[],
        help="Repeat entries like shorts=250:499 to define candidate episode pools without a manifest.",
    )
    parser.add_argument("--embedding-key", default=None, help="Embedding array key inside each NPZ archive.")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--boundary-source-window", type=int, default=None)
    parser.add_argument("--temporal-smoothing-window", type=int, default=None)
    parser.add_argument(
        "--save-mode",
        choices=["dataset", "external"],
        default="dataset",
        help=(
            "Where to save transferred checkpoints. "
            "'dataset' writes into the dataset annotations tree; "
            "'external' writes into DTW-Transfer/outputs."
        ),
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite transferred checkpoint outputs.")
    parser.add_argument("--validate-only", action="store_true", help="Validate discovery and exit.")
    parser.add_argument("--max-references", type=int, default=None, help="Debug limit after sorting reference ids.")
    parser.add_argument("--max-targets", type=int, default=None, help="Debug limit after sorting target ids.")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if yaml is None:
        raise ImportError("PyYAML is required for YAML configs.")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid config: {path}")
    return data


def as_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def canonical_garment_name(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
    aliases = {
        "topsleeves": "top_sleeves",
        "top_sleeve": "top_sleeves",
        "topsleeve": "top_sleeves",
        "top_sleeves": "top_sleeves",
    }
    return aliases.get(normalized, normalized)


def load_dataset_config(config: dict[str, Any], dataset_name: str) -> DatasetConfig:
    dataset_block = config.get("datasets", {}).get(dataset_name)
    if not isinstance(dataset_block, dict):
        raise KeyError(f"Missing datasets.{dataset_name} in config.")
    output_root = dataset_block.get("output_root") or config.get("output", {}).get("output_root")
    if not output_root:
        raise KeyError(f"No output_root configured for dataset {dataset_name}.")
    default_category_ranges: dict[str, tuple[int, int]] = {}
    for garment_text, range_spec in (dataset_block.get("default_category_ranges") or {}).items():
        garment = canonical_garment_name(str(garment_text))
        values = sorted(expand_episode_range(range_spec))
        if not values:
            continue
        default_category_ranges[garment] = (values[0], values[-1] + 1)
    return DatasetConfig(
        name=dataset_name,
        dataset_root=as_path(dataset_block["dataset_root"]),
        embeddings_root=as_path(dataset_block["embeddings_root"]),
        annotations_root=as_path(dataset_block["annotations_root"]),
        output_root=as_path(output_root),
        video_prefix=str(dataset_block["video_prefix"]),
        default_category_ranges=default_category_ranges,
    )


def episode_index_from_text(value: str) -> int:
    stripped = value.strip()
    if stripped.isdigit():
        return int(stripped)
    match = EPISODE_PATTERN.search(stripped)
    if match:
        return int(match.group(1))
    raise ValueError(f"Could not parse episode index from {value!r}")


def parse_episode_entry(value: Any) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return episode_index_from_text(value)
    if isinstance(value, dict):
        for key in ("episode_index", "episode", "index"):
            if key in value:
                return parse_episode_entry(value[key])
        for key in ("checkpoint_path", "path", "annotation_path"):
            if key in value:
                return episode_index_from_text(str(value[key]))
    raise ValueError(f"Unsupported episode entry: {value!r}")


def expand_episode_range(value: Any) -> set[int]:
    if value is None:
        return set()
    if isinstance(value, dict):
        if "start" in value and "end" in value:
            start = int(value["start"])
            end = int(value["end"])
            if end < start:
                raise ValueError(f"Bad episode range {value!r}")
            return set(range(start, end + 1))
        raise ValueError(f"Unsupported episode range object: {value!r}")
    if isinstance(value, str):
        text = value.strip()
        if ":" not in text:
            raise ValueError(f"Expected START:END range, got {value!r}")
        left, right = text.split(":", 1)
        start = int(left)
        end = int(right)
        if end < start:
            raise ValueError(f"Bad episode range {value!r}")
        return set(range(start, end + 1))
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ValueError(f"Expected [start, end], got {value!r}")
        start, end = int(value[0]), int(value[1])
        if end < start:
            raise ValueError(f"Bad episode range {value!r}")
        return set(range(start, end + 1))
    raise ValueError(f"Unsupported episode range entry: {value!r}")


def parse_candidate_range_overrides(entries: list[str]) -> dict[str, set[int]]:
    out: dict[str, set[int]] = defaultdict(set)
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"Expected GARMENT=START:END for --candidate-range, got {entry!r}")
        garment_text, range_text = entry.split("=", 1)
        garment = canonical_garment_name(garment_text)
        out[garment].update(expand_episode_range(range_text))
    return dict(out)


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        if yaml is None:
            raise ImportError("PyYAML is required for YAML manifests.")
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid manifest: {path}")
    return data


def parse_manifest(path: Path | None) -> dict[str, dict[str, set[int]]]:
    if path is None:
        return {}
    data = load_manifest(path)
    garments_block = data.get("garments", data)
    if not isinstance(garments_block, dict):
        raise ValueError("Manifest must map garment names to settings.")

    parsed: dict[str, dict[str, set[int]]] = {}
    for garment_text, spec in garments_block.items():
        garment = canonical_garment_name(str(garment_text))
        if isinstance(spec, list):
            parsed[garment] = {
                "reference_episodes": {parse_episode_entry(item) for item in spec},
                "candidate_episodes": set(),
            }
            continue
        if not isinstance(spec, dict):
            raise ValueError(f"Unsupported manifest garment spec for {garment_text!r}: {spec!r}")
        reference_entries = spec.get("reference_episodes", spec.get("references", []))
        candidate_entries = spec.get("candidate_episodes", spec.get("episodes", []))
        range_entries = spec.get("episode_ranges", spec.get("ranges", spec.get("episode_range")))
        refs = {parse_episode_entry(item) for item in reference_entries}
        candidates = {parse_episode_entry(item) for item in candidate_entries}
        if range_entries is not None:
            if isinstance(range_entries, list) and range_entries and not isinstance(range_entries[0], (int, str, dict)):
                raise ValueError(f"Unsupported range list for garment {garment}.")
            if isinstance(range_entries, list) and range_entries and isinstance(range_entries[0], (dict, str, list, tuple)):
                for range_entry in range_entries:
                    candidates.update(expand_episode_range(range_entry))
            else:
                candidates.update(expand_episode_range(range_entries))
        parsed[garment] = {
            "reference_episodes": refs,
            "candidate_episodes": candidates,
        }
    return parsed


def episode_chunk_name(episode_index: int) -> str:
    return f"chunk-{episode_index // 1000:03d}"


def episode_dir_name(episode_index: int) -> str:
    return f"episode_{episode_index:06d}"


def annotation_path_for_index(annotations_root: Path, episode_index: int) -> Path:
    return annotations_root / episode_chunk_name(episode_index) / episode_dir_name(episode_index) / "checkpoints.json"


def embedding_path_for_index(embeddings_root: Path, episode_index: int) -> Path:
    return embeddings_root / episode_chunk_name(episode_index) / episode_dir_name(episode_index) / "frame_embeddings.npz"


def transferred_output_path_for_index(
    output_root: Path,
    dataset_name: str,
    garment: str,
    episode_index: int,
) -> Path:
    return (
        output_root
        / dataset_name
        / garment
        / "transferred_checkpoints"
        / episode_chunk_name(episode_index)
        / episode_dir_name(episode_index)
        / "checkpoints.json"
    )


def ensure_path_within_root(path: Path, root: Path, label: str) -> None:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"Resolved path escapes {label}: {path}") from exc


def read_template_status(path: Path) -> str | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    status = data.get("template_status")
    return str(status) if status is not None else None


def l2_normalize_rows(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.maximum(norms, 1e-12)


def temporal_smooth(array: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return l2_normalize_rows(array)
    if window % 2 == 0:
        raise ValueError("temporal_smoothing_window must be odd")
    radius = window // 2
    padded = np.pad(array.astype(np.float32, copy=False), ((radius, radius), (0, 0)), mode="edge")
    cumulative = np.vstack(
        [np.zeros((1, array.shape[1]), dtype=np.float64), np.cumsum(padded, axis=0, dtype=np.float64)]
    )
    smoothed = (cumulative[window:] - cumulative[:-window]) / float(window)
    return l2_normalize_rows(smoothed.astype(np.float32))


def load_annotation_record(path: Path) -> AnnotationRecord:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("template_status") != "manually_annotated":
        raise ValueError(f"Annotation is not manual: {path}")
    match = EPISODE_PATTERN.search(path.parent.name)
    if not match:
        raise ValueError(f"Could not parse episode index from annotation path: {path}")
    garment = canonical_garment_name(str(data.get("category_id") or data.get("task", {}).get("garment_type") or ""))
    labels = [str(label) for label in data.get("labels", [])]
    if not garment or not labels:
        raise ValueError(f"Missing garment or labels in {path}")
    return AnnotationRecord(
        episode_index=int(match.group(1)),
        garment_type=garment,
        checkpoint_path=path,
        labels=labels,
        task=dict(data.get("task", {})),
        raw=data,
    )


def discover_manual_annotations(annotations_root: Path) -> dict[int, AnnotationRecord]:
    records: dict[int, AnnotationRecord] = {}
    for path in sorted(annotations_root.rglob("checkpoints.json")):
        try:
            record = load_annotation_record(path)
        except ValueError:
            continue
        records[record.episode_index] = record
    return records


def validate_reference_labels(records: list[AnnotationRecord]) -> list[str]:
    labels = records[0].labels
    for record in records[1:]:
        if record.labels != labels:
            raise ValueError(
                f"Reference label mismatch for garment {records[0].garment_type}: "
                f"{records[0].episode_index} vs {record.episode_index}"
            )
    return labels


def load_embedding(embedding_path: Path, embedding_key: str, smoothing_window: int) -> EpisodeEmbedding:
    if not embedding_path.exists():
        raise FileNotFoundError(embedding_path)
    match = EPISODE_PATTERN.search(str(embedding_path))
    if not match:
        raise ValueError(f"Could not parse episode index from embedding path: {embedding_path}")
    with np.load(embedding_path, allow_pickle=False) as archive:
        if embedding_key not in archive:
            raise KeyError(f"{embedding_path} does not contain key {embedding_key!r}")
        features = np.asarray(archive[embedding_key], dtype=np.float32)
        frame_indices = np.asarray(archive.get("frame_indices", np.arange(features.shape[0])), dtype=np.int64)
    if features.ndim != 2 or features.shape[0] <= 0:
        raise ValueError(f"Expected [frames, dim] embeddings from {embedding_path}, got {features.shape}")
    expected = np.arange(features.shape[0], dtype=np.int64)
    if frame_indices.shape != expected.shape or not np.array_equal(frame_indices, expected):
        raise ValueError(f"frame_indices must be contiguous and zero-based in {embedding_path}")
    features = temporal_smooth(features, smoothing_window)
    return EpisodeEmbedding(
        episode_index=int(match.group(1)),
        embedding_path=embedding_path,
        frame_count=int(features.shape[0]),
        dimension=int(features.shape[1]),
        features=features,
    )


def reference_weight(normalized_cost: float, temperature: float) -> float:
    return float(math.exp(-normalized_cost / max(temperature, 1e-6)))


def cosine_distance_matrix(reference: np.ndarray, target: np.ndarray) -> np.ndarray:
    distance = 1.0 - (reference @ target.T)
    return np.clip(distance, 0.0, 2.0).astype(np.float32, copy=False)


if njit is not None:

    @njit(cache=True)
    def _dtw_cost_and_back_numba(distance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:  # pragma: no cover
        n, m = distance.shape
        cost = np.empty((n, m), dtype=np.float32)
        back = np.empty((n, m), dtype=np.int8)
        for i in range(n):
            for j in range(m):
                cost[i, j] = np.inf
                back[i, j] = -1
        cost[0, 0] = distance[0, 0]
        for i in range(n):
            for j in range(m):
                if i == 0 and j == 0:
                    continue
                best = np.inf
                move = -1
                if i > 0 and cost[i - 1, j] < best:
                    best = cost[i - 1, j]
                    move = 0
                if j > 0 and cost[i, j - 1] < best:
                    best = cost[i, j - 1]
                    move = 1
                if i > 0 and j > 0 and cost[i - 1, j - 1] < best:
                    best = cost[i - 1, j - 1]
                    move = 2
                cost[i, j] = distance[i, j] + best
                back[i, j] = move
        return cost, back


def dtw_cost_and_back(distance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if njit is not None:
        return _dtw_cost_and_back_numba(distance)
    n, m = distance.shape
    cost = np.full((n, m), np.inf, dtype=np.float32)
    back = np.full((n, m), -1, dtype=np.int8)
    cost[0, 0] = distance[0, 0]
    for i in range(n):
        for j in range(m):
            if i == 0 and j == 0:
                continue
            best = np.inf
            move = -1
            if i > 0 and cost[i - 1, j] < best:
                best = cost[i - 1, j]
                move = 0
            if j > 0 and cost[i, j - 1] < best:
                best = cost[i, j - 1]
                move = 1
            if i > 0 and j > 0 and cost[i - 1, j - 1] < best:
                best = cost[i - 1, j - 1]
                move = 2
            cost[i, j] = distance[i, j] + best
            back[i, j] = move
    return cost, back


def backtrack(back: np.ndarray) -> list[tuple[int, int]]:
    i, j = back.shape[0] - 1, back.shape[1] - 1
    path: list[tuple[int, int]] = []
    while True:
        path.append((i, j))
        move = int(back[i, j])
        if move == -1:
            break
        if move == 0:
            i -= 1
        elif move == 1:
            j -= 1
        elif move == 2:
            i -= 1
            j -= 1
        else:  # pragma: no cover - defensive
            raise RuntimeError(f"Bad DTW move {move}")
    path.reverse()
    return path


def path_to_mapping(path: list[tuple[int, int]]) -> dict[int, list[int]]:
    mapping: dict[int, list[int]] = {}
    for source_i, target_j in path:
        mapping.setdefault(int(source_i), []).append(int(target_j))
    return mapping


def map_boundary(
    source_boundary: int,
    mapping: dict[int, list[int]],
    source_len: int,
    target_len: int,
    source_window: int,
) -> int:
    if source_boundary <= 0:
        return 0
    if source_boundary >= source_len:
        return target_len
    start = max(0, source_boundary - source_window)
    end = min(source_len, source_boundary + source_window + 1)
    values: list[int] = []
    for source_i in range(start, end):
        values.extend(mapping.get(source_i, []))
    if values:
        return int(round(float(np.median(np.asarray(values, dtype=np.float32)))))

    for radius in range(source_window + 1, source_len + 1):
        for source_i in (source_boundary - radius, source_boundary + radius):
            if 0 <= source_i < source_len and source_i in mapping:
                return int(round(float(np.median(np.asarray(mapping[source_i], dtype=np.float32)))))
    return min(max(source_boundary, 0), target_len)


def transfer_boundaries_for_reference(
    reference_segments: list[dict[str, Any]],
    mapping: dict[int, list[int]],
    source_len: int,
    target_len: int,
    source_window: int,
) -> list[int]:
    boundaries = [0]
    for segment in reference_segments[:-1]:
        source_boundary = int(segment["end_frame_exclusive"])
        boundaries.append(map_boundary(source_boundary, mapping, source_len, target_len, source_window))
    boundaries.append(target_len)
    return enforce_monotonic_boundaries(boundaries, target_len, min_gap=1)


def enforce_monotonic_boundaries(boundaries: list[int], max_len: int, min_gap: int) -> list[int]:
    out = [0]
    internal_count = len(boundaries) - 2
    for idx, value in enumerate(boundaries[1:-1], start=1):
        min_allowed = out[-1] + min_gap
        remaining = internal_count - idx + 1
        max_allowed = max_len - remaining * min_gap
        out.append(int(max(min_allowed, min(int(round(value)), max_allowed))))
    out.append(max_len)
    return out


def weighted_median(values: list[int], weights: list[float]) -> int:
    pairs = sorted(zip(values, weights), key=lambda item: item[0])
    total = sum(max(0.0, weight) for _, weight in pairs)
    if total <= 0:
        return int(round(float(np.median(np.asarray(values, dtype=np.float32)))))
    cumulative = 0.0
    for value, weight in pairs:
        cumulative += max(0.0, weight)
        if cumulative >= total * 0.5:
            return int(value)
    return int(pairs[-1][0])


def fuse_boundaries(votes: list[dict[str, Any]], top_k: int, target_len: int) -> tuple[list[int], list[dict[str, Any]]]:
    selected = votes[: min(top_k, len(votes))]
    if not selected:
        raise ValueError("No DTW reference votes available.")
    n_boundaries = len(selected[0]["boundaries"])
    fused = [0]
    diagnostics: list[dict[str, Any]] = []
    for boundary_index in range(1, n_boundaries - 1):
        values = [int(vote["boundaries"][boundary_index]) for vote in selected]
        weights = [float(vote["weight"]) for vote in selected]
        boundary = weighted_median(values, weights)
        spread = float(np.std(np.asarray(values, dtype=np.float32)))
        diagnostics.append(
            {
                "boundary_index": boundary_index,
                "weighted_median_frame": boundary,
                "vote_min_frame": int(min(values)),
                "vote_max_frame": int(max(values)),
                "vote_std_frame": spread,
                "num_votes": len(values),
            }
        )
        fused.append(boundary)
    fused.append(target_len)
    return enforce_monotonic_boundaries(fused, target_len, min_gap=1), diagnostics


def boundary_confidence(
    diagnostics: list[dict[str, Any]],
    selected_votes: list[dict[str, Any]],
    temperature: float,
) -> list[dict[str, Any]]:
    mean_cost = float(np.mean([vote["normalized_cost"] for vote in selected_votes])) if selected_votes else 1.0
    match_quality = float(math.exp(-mean_cost / max(temperature, 1e-6)))
    out = []
    for item in diagnostics:
        std = float(item["vote_std_frame"])
        agreement = 1.0 / (1.0 + std / 5.0)
        confidence = max(0.0, min(1.0, agreement * match_quality))
        enriched = dict(item)
        enriched.update(
            {
                "reference_agreement_score": agreement,
                "reference_match_quality": match_quality,
                "confidence": confidence,
            }
        )
        out.append(enriched)
    return out


def build_segment_progress(segment: dict[str, Any]) -> dict[str, Any]:
    start = int(segment["start_frame"])
    end_inclusive = int(segment["end_frame_inclusive"])
    num_frames = int(segment["num_frames"])
    slope = 0.0 if num_frames <= 1 else 1.0 / float(num_frames - 1)
    per_frame = []
    for offset, frame in enumerate(range(start, end_inclusive + 1)):
        progress = 0.0 if num_frames <= 1 else offset * slope
        per_frame.append({"frame": frame, "progress": float(round(progress, 6))})
    control_points = [
        {"frame": start, "progress": 0.0},
        {"frame": end_inclusive, "progress": 1.0 if num_frames > 1 else 0.0},
    ]
    linear_pieces = [
        {
            "start_frame": start,
            "end_frame_inclusive": end_inclusive,
            "start_progress": 0.0,
            "end_progress": 1.0 if num_frames > 1 else 0.0,
            "slope_per_frame": float(round(slope, 9)),
        }
    ]
    return {
        "method": "piecewise_linear_direction_toggles",
        "range": [0, 1],
        "initial_direction": "increasing",
        "slope_per_frame": float(round(slope, 9)),
        "control_points": control_points,
        "linear_pieces": linear_pieces,
        "direction_changes": [],
        "end_progress": 1.0 if num_frames > 1 else 0.0,
        "per_frame": per_frame,
    }


def segments_from_boundaries(
    labels: list[str],
    boundaries: list[int],
    boundary_conf: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    conf_by_end_boundary = {int(item["boundary_index"]): float(item["confidence"]) for item in boundary_conf}
    segments = []
    for idx, label in enumerate(labels):
        start = int(boundaries[idx])
        end = int(boundaries[idx + 1])
        start_conf = conf_by_end_boundary.get(idx, 1.0 if idx == 0 else None)
        end_conf = conf_by_end_boundary.get(idx + 1, 1.0 if idx == len(labels) - 1 else None)
        conf_values = [value for value in (start_conf, end_conf) if value is not None]
        confidence = float(np.mean(conf_values)) if conf_values else 1.0
        segment = {
            "segment_id": idx,
            "label": label,
            "start_frame": start,
            "end_frame_exclusive": end,
            "end_frame_inclusive": end - 1,
            "num_frames": max(0, end - start),
            "confidence": confidence,
            "notes": "Checkpoint label transferred with multi-reference DTW over precomputed 2048-D frame embeddings.",
        }
        segment["progress"] = build_segment_progress(segment)
        segments.append(segment)
    return segments


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


def source_file_for_episode(dataset_root: Path, episode_index: int) -> str:
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        return ""
    info = json.loads(info_path.read_text(encoding="utf-8"))
    template = info.get("data_path")
    if not template:
        return ""
    rel = template.format(episode_chunk=episode_index // 1000, episode_index=episode_index)
    return str((dataset_root / rel).resolve())


def select_garments(
    dataset_cfg: DatasetConfig,
    manual_annotations: dict[int, AnnotationRecord],
    manifest: dict[str, dict[str, set[int]]],
    include: list[str],
    exclude: list[str],
) -> list[str]:
    available = (
        {record.garment_type for record in manual_annotations.values()}
        | set(manifest)
        | set(dataset_cfg.default_category_ranges)
    )
    if include:
        selected = {canonical_garment_name(item) for item in include}
    else:
        selected = set(available)
    selected -= {canonical_garment_name(item) for item in exclude}
    unknown = sorted(selected - available)
    if unknown:
        raise ValueError(f"Requested garments not available from manual annotations or manifest: {unknown}")
    return sorted(selected)


def resolve_reference_records(
    garment: str,
    manual_annotations: dict[int, AnnotationRecord],
    manifest_spec: dict[str, set[int]] | None,
    max_references: int | None,
) -> list[AnnotationRecord]:
    if manifest_spec and manifest_spec.get("reference_episodes"):
        episode_ids = sorted(manifest_spec["reference_episodes"])
        records = []
        for episode_index in episode_ids:
            record = manual_annotations.get(episode_index)
            if record is None:
                annotation_path = annotation_path_for_index(next(iter(manual_annotations.values())).checkpoint_path.parents[2], episode_index)
                raise FileNotFoundError(
                    f"Reference episode {episode_index} for garment {garment} is missing a manual annotation. "
                    f"Expected {annotation_path}"
                )
            if record.garment_type != garment:
                raise ValueError(
                    f"Reference episode {episode_index} is labeled as {record.garment_type}, not {garment}."
                )
            records.append(record)
    else:
        records = sorted(
            [record for record in manual_annotations.values() if record.garment_type == garment],
            key=lambda item: item.episode_index,
        )
    if max_references is not None:
        records = records[:max_references]
    if not records:
        raise ValueError(f"No reference annotations found for garment {garment}.")
    return records


def resolve_candidate_episode_pool(
    dataset_cfg: DatasetConfig,
    garment: str,
    reference_records: list[AnnotationRecord],
    manifest_spec: dict[str, set[int]] | None,
    candidate_policy: str,
    candidate_overrides: dict[str, set[int]],
) -> tuple[list[int], str]:
    if garment in candidate_overrides and candidate_overrides[garment]:
        candidates = set(candidate_overrides[garment])
        source = "cli_candidate_range"
    elif manifest_spec and manifest_spec.get("candidate_episodes"):
        candidates = set(manifest_spec["candidate_episodes"])
        source = "manifest_candidate_episodes"
    elif garment in dataset_cfg.default_category_ranges:
        start, end_exclusive = dataset_cfg.default_category_ranges[garment]
        candidates = set(range(start, end_exclusive))
        source = "dataset_default_category_range"
    elif candidate_policy == "contiguous_hull_from_references":
        ref_indices = [record.episode_index for record in reference_records]
        candidates = set(range(min(ref_indices), max(ref_indices) + 1))
        source = "contiguous_hull_from_references"
    elif candidate_policy == "manual_annotations_only":
        candidates = {record.episode_index for record in reference_records}
        source = "manual_annotations_only"
    else:
        raise ValueError(
            f"No candidate episode pool defined for garment {garment}. "
            f"Provide --reference-manifest, --candidate-range, or use --candidate-policy contiguous_hull_from_references."
        )
    reference_ids = {record.episode_index for record in reference_records}
    targets = sorted(candidates - reference_ids)
    return targets, source


def ensure_output_parent(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists, pass --overwrite to replace it: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def transfer_garment(
    garment: str,
    dataset_cfg: DatasetConfig,
    reference_records: list[AnnotationRecord],
    manual_annotations: dict[int, AnnotationRecord],
    target_episode_indices: list[int],
    candidate_pool_source: str,
    config: dict[str, Any],
    manifest_path: Path | None,
    save_mode: str,
    overwrite: bool,
) -> None:
    transfer_cfg = config["transfer"]
    output_cfg = config["output"]
    labels = validate_reference_labels(reference_records)
    canonical_task = dict(reference_records[0].task)
    canonical_task["garment_type"] = garment

    embedding_key = str(transfer_cfg["embedding_key"])
    smoothing_window = int(transfer_cfg["temporal_smoothing_window"])
    top_k = int(transfer_cfg["top_k"])
    temperature = float(transfer_cfg["temperature"])
    boundary_source_window = int(transfer_cfg["boundary_source_window"])

    reference_embeddings = {
        record.episode_index: load_embedding(
            embedding_path_for_index(dataset_cfg.embeddings_root, record.episode_index),
            embedding_key,
            smoothing_window,
        )
        for record in tqdm(reference_records, desc=f"{garment}: load refs", unit="episode")
    }
    garment_output_root = dataset_cfg.output_root / dataset_cfg.name / garment
    report_root = garment_output_root / "reports"
    report_root.mkdir(parents=True, exist_ok=True)

    manual_annotation_ids = set(manual_annotations)
    pending_targets: list[tuple[int, Path]] = []
    skipped_manual = 0
    skipped_existing = 0
    overwritten_existing = 0

    for episode_index in target_episode_indices:
        if episode_index in manual_annotation_ids:
            tqdm.write(
                f"Skipping garment {garment}, episode {episode_index}: manual annotation already exists in dataset."
            )
            skipped_manual += 1
            continue

        if save_mode == "dataset":
            out_path = annotation_path_for_index(dataset_cfg.annotations_root, episode_index)
            ensure_path_within_root(out_path, dataset_cfg.annotations_root, "annotations_root")
        else:
            out_path = transferred_output_path_for_index(
                dataset_cfg.output_root,
                dataset_cfg.name,
                garment,
                episode_index,
            )
            ensure_path_within_root(out_path, dataset_cfg.output_root, "output_root")

        if out_path.exists():
            existing_status = read_template_status(out_path)
            if existing_status == "manually_annotated":
                tqdm.write(
                    f"Skipping garment {garment}, episode {episode_index}: output path already contains manual annotation."
                )
                skipped_manual += 1
                continue
            if not overwrite:
                tqdm.write(
                    f"Skipping garment {garment}, episode {episode_index}: output already exists, pass --overwrite to replace it."
                )
                skipped_existing += 1
                continue
            overwritten_existing += 1

        pending_targets.append((episode_index, out_path))

    if not pending_targets:
        tqdm.write(
            f"Skipping garment {garment}: nothing to write "
            f"(skipped_manual={skipped_manual}, skipped_existing={skipped_existing})."
        )
        return

    target_embeddings = {
        episode_index: load_embedding(
            embedding_path_for_index(dataset_cfg.embeddings_root, episode_index),
            embedding_key,
            smoothing_window,
        )
        for episode_index, _ in tqdm(pending_targets, desc=f"{garment}: load targets", unit="episode")
    }

    all_scores: list[dict[str, Any]] = []
    for target_episode_index, out_path in tqdm(pending_targets, desc=f"{garment}: transfer", unit="episode"):
        target_embedding = target_embeddings[target_episode_index]
        votes: list[dict[str, Any]] = []
        for reference_record in reference_records:
            reference_embedding = reference_embeddings[reference_record.episode_index]
            if reference_embedding.dimension != target_embedding.dimension:
                raise ValueError(
                    f"Embedding dimension mismatch for garment {garment}: "
                    f"{reference_record.episode_index} has {reference_embedding.dimension}, "
                    f"{target_episode_index} has {target_embedding.dimension}"
                )
            reference_segments = reference_record.raw["segments"]
            if int(reference_segments[-1]["end_frame_exclusive"]) != reference_embedding.frame_count:
                raise ValueError(
                    f"Reference annotation {reference_record.checkpoint_path} ends at "
                    f"{reference_segments[-1]['end_frame_exclusive']} but embeddings have {reference_embedding.frame_count} frames."
                )

            distance = cosine_distance_matrix(reference_embedding.features, target_embedding.features)
            cost, back = dtw_cost_and_back(distance)
            path = backtrack(back)
            mapping = path_to_mapping(path)
            normalized_cost = float(cost[-1, -1] / max(1, len(path)))
            vote = {
                "reference_episode_index": reference_record.episode_index,
                "reference_checkpoint": str(reference_record.checkpoint_path),
                "normalized_cost": normalized_cost,
                "total_cost": float(cost[-1, -1]),
                "path_length": len(path),
                "weight": reference_weight(normalized_cost, temperature),
                "boundaries": transfer_boundaries_for_reference(
                    reference_segments,
                    mapping,
                    source_len=reference_embedding.frame_count,
                    target_len=target_embedding.frame_count,
                    source_window=boundary_source_window,
                ),
            }
            votes.append(vote)
            all_scores.append(
                {
                    "garment": garment,
                    "target_episode_index": target_episode_index,
                    "reference_episode_index": reference_record.episode_index,
                    "normalized_cost": normalized_cost,
                    "total_cost": float(cost[-1, -1]),
                    "path_length": len(path),
                    "weight": vote["weight"],
                }
            )

        votes.sort(key=lambda row: row["normalized_cost"])
        selected_votes = votes[: min(top_k, len(votes))]
        fused_boundaries, boundary_diag = fuse_boundaries(votes, top_k=top_k, target_len=target_embedding.frame_count)
        boundary_conf = boundary_confidence(boundary_diag, selected_votes, temperature=temperature)
        segments = segments_from_boundaries(labels, fused_boundaries, boundary_conf)
        validation = validate_output_segments(segments, target_embedding.frame_count)
        if not validation["valid"]:
            raise RuntimeError(
                f"Generated checkpoint validation failed for garment {garment}, episode {target_episode_index}: "
                + "; ".join(validation["errors"])
            )

        video_stem = f"{dataset_cfg.video_prefix}_{target_episode_index:06d}_top_rgb"
        ensure_output_parent(out_path, overwrite=overwrite)
        payload = {
            "category_id": garment,
            "task": canonical_task,
            "labels": labels,
            "schema_version": "1.0",
            "template_description": (
                "Temporal segment checkpoints transferred with multi-reference DTW over precomputed 2048-D frame embeddings."
            ),
            "template_status": str(output_cfg["template_status"]),
            "video_stem": video_stem,
            "video_file": f"{video_stem}.mp4",
            "source_file": source_file_for_episode(dataset_cfg.dataset_root, target_episode_index),
            "embedding_file": str(target_embedding.embedding_path),
            "frame_indexing": {
                "base": 0,
                "start_frame": 0,
                "end_frame_exclusive": target_embedding.frame_count,
                "end_frame_inclusive": target_embedding.frame_count - 1,
                "note": "Use zero-based decoded-video frame indices. Each segment is [start_frame, end_frame_exclusive).",
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
                "last_frame_inclusive": target_embedding.frame_count - 1,
                "num_labeled_frames": target_embedding.frame_count,
                "coverage_is_contiguous": validation["coverage_is_contiguous"],
                "coverage_is_complete": validation["coverage_is_complete"],
                "num_references_total": len(reference_records),
                "top_k": len(selected_votes),
                "best_reference_episode_index": selected_votes[0]["reference_episode_index"],
                "best_reference_normalized_cost": selected_votes[0]["normalized_cost"],
                "mean_topk_normalized_cost": float(np.mean([vote["normalized_cost"] for vote in selected_votes])),
            },
            "annotation_guidance": {
                "segments": "Segments are sequential and non-overlapping. end_frame_exclusive equals the next segment's start_frame.",
                "coverage": (
                    f"Every decoded frame from 0 through {target_embedding.frame_count - 1} is labeled exactly once."
                ),
                "progress": (
                    "Transferred segments use a simple monotonic 0..1 progress ramp inside each segment. "
                    "Review before downstream use if within-segment progress semantics matter."
                ),
            },
            "segment_schema": {
                "segment_id": "zero-based integer in temporal order",
                "label": "task-specific string included in labels",
                "start_frame": "inclusive zero-based frame index",
                "end_frame_exclusive": "exclusive zero-based frame boundary",
                "end_frame_inclusive": "end_frame_exclusive - 1",
                "num_frames": "end_frame_exclusive - start_frame",
                "progress": "piecewise-linear per-frame progress",
            },
            "method": {
                "name": "multi_reference_frame_embedding_dtw",
                "dataset": dataset_cfg.name,
                "garment_type": garment,
                "embedding_key": embedding_key,
                "embedding_dimension": target_embedding.dimension,
                "distance": "1 - cosine similarity over L2-normalized 2048-D frame embeddings",
                "temporal_smoothing_window": smoothing_window,
                "boundary_mapping": "median target frame votes around each reference boundary",
                "boundary_source_window": boundary_source_window,
                "top_k_fusion": "weighted median of top-K transferred boundaries",
                "reference_weight": "exp(-normalized_dtw_cost / temperature)",
                "temperature": temperature,
                "candidate_pool_policy": transfer_cfg["candidate_policy"],
                "candidate_pool_source": candidate_pool_source,
                "save_mode": save_mode,
                "reference_manifest": str(manifest_path.resolve()) if manifest_path else None,
            },
            "segments": segments,
            "boundaries": {
                "fused": fused_boundaries,
                "confidence": boundary_conf,
            },
            "top_references": [
                {
                    "rank": rank,
                    "reference_episode_index": vote["reference_episode_index"],
                    "reference_checkpoint": vote["reference_checkpoint"],
                    "normalized_cost": vote["normalized_cost"],
                    "total_cost": vote["total_cost"],
                    "path_length": vote["path_length"],
                    "weight": vote["weight"],
                    "boundaries": vote["boundaries"],
                }
                for rank, vote in enumerate(selected_votes, start=1)
            ],
            "all_reference_votes": [
                {
                    "reference_episode_index": vote["reference_episode_index"],
                    "normalized_cost": vote["normalized_cost"],
                    "weight": vote["weight"],
                    "boundaries": vote["boundaries"],
                }
                for vote in votes
            ],
        }
        out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    if output_cfg.get("write_reference_scores_csv", True):
        write_csv(
            report_root / "reference_scores.csv",
            [
                "garment",
                "target_episode_index",
                "reference_episode_index",
                "normalized_cost",
                "total_cost",
                "path_length",
                "weight",
            ],
            all_scores,
        )

    tqdm.write(
        f"Completed garment {garment}: wrote={len(pending_targets)}, overwritten_existing={overwritten_existing}, "
        f"skipped_manual={skipped_manual}, skipped_existing={skipped_existing}"
    )


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    dataset_cfg = load_dataset_config(config, args.dataset)

    transfer_cfg = dict(config.get("transfer", {}))
    if args.embedding_key is not None:
        transfer_cfg["embedding_key"] = args.embedding_key
    if args.top_k is not None:
        transfer_cfg["top_k"] = args.top_k
    if args.temperature is not None:
        transfer_cfg["temperature"] = args.temperature
    if args.boundary_source_window is not None:
        transfer_cfg["boundary_source_window"] = args.boundary_source_window
    if args.temporal_smoothing_window is not None:
        transfer_cfg["temporal_smoothing_window"] = args.temporal_smoothing_window
    if args.candidate_policy is not None:
        transfer_cfg["candidate_policy"] = args.candidate_policy
    config["transfer"] = transfer_cfg

    manual_annotations = discover_manual_annotations(dataset_cfg.annotations_root)
    if not manual_annotations:
        raise RuntimeError(f"No manually annotated checkpoints found in {dataset_cfg.annotations_root}")
    manifest = parse_manifest(args.reference_manifest)
    candidate_overrides = parse_candidate_range_overrides(args.candidate_range)
    garments = select_garments(dataset_cfg, manual_annotations, manifest, args.garment, args.exclude_garment)

    print("LeHome multi-reference DTW transfer")
    print(f"  dataset            : {dataset_cfg.name}")
    print(f"  dataset_root       : {dataset_cfg.dataset_root}")
    print(f"  embeddings_root    : {dataset_cfg.embeddings_root}")
    print(f"  annotations_root   : {dataset_cfg.annotations_root}")
    print(f"  output_root        : {dataset_cfg.output_root}")
    print(f"  save_mode          : {args.save_mode}")
    print(f"  garments           : {', '.join(garments)}")
    print(f"  embedding_key      : {transfer_cfg['embedding_key']}")
    print(f"  top_k              : {transfer_cfg['top_k']}")
    print(f"  temperature        : {transfer_cfg['temperature']}")
    print(f"  smoothing_window   : {transfer_cfg['temporal_smoothing_window']}")
    print(f"  candidate_policy   : {transfer_cfg['candidate_policy']}")
    if dataset_cfg.default_category_ranges:
        print(
            "  default_ranges     : "
            + ", ".join(
                f"{garment}={start}:{end_exclusive - 1}"
                for garment, (start, end_exclusive) in sorted(dataset_cfg.default_category_ranges.items())
            )
        )

    work_items: list[tuple[str, list[AnnotationRecord], list[int], str]] = []
    for garment in garments:
        manifest_spec = manifest.get(garment)
        reference_records = resolve_reference_records(
            garment=garment,
            manual_annotations=manual_annotations,
            manifest_spec=manifest_spec,
            max_references=args.max_references,
        )
        target_episode_indices, candidate_pool_source = resolve_candidate_episode_pool(
            dataset_cfg=dataset_cfg,
            garment=garment,
            reference_records=reference_records,
            manifest_spec=manifest_spec,
            candidate_policy=str(transfer_cfg["candidate_policy"]),
            candidate_overrides=candidate_overrides,
        )
        if args.max_targets is not None:
            target_episode_indices = target_episode_indices[: args.max_targets]
        if not target_episode_indices:
            print(f"  garment {garment}: 0 targets after excluding {len(reference_records)} references")
        else:
            print(
                f"  garment {garment}: {len(reference_records)} references, "
                f"{len(target_episode_indices)} targets"
            )
        work_items.append((garment, reference_records, target_episode_indices, candidate_pool_source))

    if args.validate_only:
        print("Validation OK.")
        return 0

    for garment, reference_records, target_episode_indices, candidate_pool_source in work_items:
        if not target_episode_indices:
            tqdm.write(f"Skipping garment {garment}: no targets.")
            continue
        transfer_garment(
            garment=garment,
            dataset_cfg=dataset_cfg,
            reference_records=reference_records,
            manual_annotations=manual_annotations,
            target_episode_indices=target_episode_indices,
            candidate_pool_source=candidate_pool_source,
            config=config,
            manifest_path=args.reference_manifest,
            save_mode=args.save_mode,
            overwrite=args.overwrite,
        )

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
