from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from common import (
    PairConfig,
    AnnotationIndex,
    HumanSplineEpisode,
    as_path,
    ensure_output_dir,
    interpolate_human_u_with_details,
    load_annotation_index,
    load_config,
    load_human_spline_episode,
    load_pairs,
    load_paths,
    load_processing,
    pair_output_dir,
    save_config_snapshot,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "unseen_val_config.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 3 for unseen validation: export exact-interpolated human start/end u per robot frame."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--pairing-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--pair-id", action="append", default=[], help="Repeat to restrict export to named pair IDs.")
    return parser.parse_args()


def filtered_pairs(pairs: list[PairConfig], requested: list[str]) -> list[PairConfig]:
    if not requested:
        return pairs
    wanted = {value.strip() for value in requested}
    out = [pair for pair in pairs if pair.pair_id in wanted]
    if len(out) != len(wanted):
        missing = sorted(wanted - {pair.pair_id for pair in out})
        raise ValueError(f"Unknown requested pair IDs: {missing}")
    return out


def pairing_npz_path(pairing_root: Path, pair: PairConfig) -> Path:
    return pair_output_dir(pairing_root, pair) / "human_episode_pairings.npz"


def load_pairing_arrays(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        frame_indices = np.asarray(archive["frame_indices"], dtype=np.int64)
        paired_human_episode_indices = np.asarray(archive["paired_human_episode_indices"], dtype=np.int64)
    return frame_indices, paired_human_episode_indices


def load_robot_local_windows(pair: PairConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = pair.robot_episode_dir / "local_raw_bspline_windows.npz"
    with np.load(path, allow_pickle=False) as archive:
        frame_indices = np.asarray(archive["frame_indices"], dtype=np.int64)
        local_start_frame_index = np.asarray(archive["local_start_frame_index"], dtype=np.int64)
        local_end_frame_index = np.asarray(archive["local_end_frame_index"], dtype=np.int64)
    return frame_indices, local_start_frame_index, local_end_frame_index


def build_exact_match_arrays(
    pair: PairConfig,
    frame_indices: np.ndarray,
    paired_human_episode_indices: np.ndarray,
    local_start_frame_index: np.ndarray,
    local_end_frame_index: np.ndarray,
    robot_annotation: AnnotationIndex,
    human_annotation: AnnotationIndex,
    human_spline: HumanSplineEpisode,
    processing_index_dtype: str,
    processing_progress_dtype: str,
    processing_u_dtype: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    num_frames = int(frame_indices.shape[0])
    num_pairings = int(paired_human_episode_indices.shape[1])
    index_dtype = np.dtype(processing_index_dtype)
    progress_dtype = np.dtype(processing_progress_dtype)
    u_dtype = np.dtype(processing_u_dtype)

    robot_start_frame_index = np.asarray(local_start_frame_index, dtype=index_dtype)
    robot_end_frame_index = np.asarray(local_end_frame_index, dtype=index_dtype)
    robot_start_segment_id = robot_annotation.frame_to_segment_id[local_start_frame_index].astype(index_dtype, copy=False)
    robot_end_segment_id = robot_annotation.frame_to_segment_id[local_end_frame_index].astype(index_dtype, copy=False)
    robot_start_progress = robot_annotation.frame_to_progress[local_start_frame_index].astype(progress_dtype, copy=False)
    robot_end_progress = robot_annotation.frame_to_progress[local_end_frame_index].astype(progress_dtype, copy=False)

    human_start_u = np.full((num_frames, num_pairings), fill_value=np.nan, dtype=u_dtype)
    human_end_u = np.full((num_frames, num_pairings), fill_value=np.nan, dtype=u_dtype)
    start_valid_mask = np.zeros((num_frames, num_pairings), dtype=bool)
    end_valid_mask = np.zeros((num_frames, num_pairings), dtype=bool)
    start_lower_frame_index = np.full((num_frames, num_pairings), fill_value=-1, dtype=index_dtype)
    start_upper_frame_index = np.full((num_frames, num_pairings), fill_value=-1, dtype=index_dtype)
    end_lower_frame_index = np.full((num_frames, num_pairings), fill_value=-1, dtype=index_dtype)
    end_upper_frame_index = np.full((num_frames, num_pairings), fill_value=-1, dtype=index_dtype)
    start_lower_progress = np.full((num_frames, num_pairings), fill_value=np.nan, dtype=progress_dtype)
    start_upper_progress = np.full((num_frames, num_pairings), fill_value=np.nan, dtype=progress_dtype)
    end_lower_progress = np.full((num_frames, num_pairings), fill_value=np.nan, dtype=progress_dtype)
    end_upper_progress = np.full((num_frames, num_pairings), fill_value=np.nan, dtype=progress_dtype)
    start_interp_alpha = np.full((num_frames, num_pairings), fill_value=np.nan, dtype=progress_dtype)
    end_interp_alpha = np.full((num_frames, num_pairings), fill_value=np.nan, dtype=progress_dtype)
    start_exact_match_mask = np.zeros((num_frames, num_pairings), dtype=bool)
    end_exact_match_mask = np.zeros((num_frames, num_pairings), dtype=bool)
    start_boundary_clamped_mask = np.zeros((num_frames, num_pairings), dtype=bool)
    end_boundary_clamped_mask = np.zeros((num_frames, num_pairings), dtype=bool)

    start_reason_counter: Counter[str] = Counter()
    end_reason_counter: Counter[str] = Counter()

    for row in tqdm(range(num_frames), desc=f"{pair.pair_id}: frames", unit="frame", leave=False):
        start_segment_id = int(robot_start_segment_id[row])
        end_segment_id = int(robot_end_segment_id[row])
        start_progress = float(robot_start_progress[row])
        end_progress = float(robot_end_progress[row])
        for slot in range(num_pairings):
            start_details = interpolate_human_u_with_details(
                human_annotation,
                human_spline,
                start_segment_id,
                start_progress,
            )
            end_details = interpolate_human_u_with_details(
                human_annotation,
                human_spline,
                end_segment_id,
                end_progress,
            )
            start_reason_counter[start_details["reason"]] += 1
            end_reason_counter[end_details["reason"]] += 1

            if start_details["valid"]:
                human_start_u[row, slot] = np.asarray(start_details["u"], dtype=u_dtype)
                start_valid_mask[row, slot] = True
            if end_details["valid"]:
                human_end_u[row, slot] = np.asarray(end_details["u"], dtype=u_dtype)
                end_valid_mask[row, slot] = True

            start_lower_frame_index[row, slot] = np.asarray(start_details["lower_frame"], dtype=index_dtype)
            start_upper_frame_index[row, slot] = np.asarray(start_details["upper_frame"], dtype=index_dtype)
            end_lower_frame_index[row, slot] = np.asarray(end_details["lower_frame"], dtype=index_dtype)
            end_upper_frame_index[row, slot] = np.asarray(end_details["upper_frame"], dtype=index_dtype)
            start_lower_progress[row, slot] = np.asarray(start_details["lower_progress"], dtype=progress_dtype)
            start_upper_progress[row, slot] = np.asarray(start_details["upper_progress"], dtype=progress_dtype)
            end_lower_progress[row, slot] = np.asarray(end_details["lower_progress"], dtype=progress_dtype)
            end_upper_progress[row, slot] = np.asarray(end_details["upper_progress"], dtype=progress_dtype)
            start_interp_alpha[row, slot] = np.asarray(start_details["alpha"], dtype=progress_dtype)
            end_interp_alpha[row, slot] = np.asarray(end_details["alpha"], dtype=progress_dtype)
            start_exact_match_mask[row, slot] = bool(start_details["exact_match"])
            end_exact_match_mask[row, slot] = bool(end_details["exact_match"])
            start_boundary_clamped_mask[row, slot] = bool(start_details["boundary_clamped"])
            end_boundary_clamped_mask[row, slot] = bool(end_details["boundary_clamped"])

    arrays = {
        "frame_indices": np.asarray(frame_indices, dtype=index_dtype),
        "paired_human_episode_indices": np.asarray(paired_human_episode_indices, dtype=index_dtype),
        "robot_start_frame_index": robot_start_frame_index,
        "robot_end_frame_index": robot_end_frame_index,
        "robot_start_segment_id": robot_start_segment_id,
        "robot_end_segment_id": robot_end_segment_id,
        "robot_start_progress": robot_start_progress,
        "robot_end_progress": robot_end_progress,
        "human_start_u": human_start_u,
        "human_end_u": human_end_u,
        "start_valid_mask": start_valid_mask,
        "end_valid_mask": end_valid_mask,
        "start_lower_frame_index": start_lower_frame_index,
        "start_upper_frame_index": start_upper_frame_index,
        "end_lower_frame_index": end_lower_frame_index,
        "end_upper_frame_index": end_upper_frame_index,
        "start_lower_progress": start_lower_progress,
        "start_upper_progress": start_upper_progress,
        "end_lower_progress": end_lower_progress,
        "end_upper_progress": end_upper_progress,
        "start_interp_alpha": start_interp_alpha,
        "end_interp_alpha": end_interp_alpha,
        "start_exact_match_mask": start_exact_match_mask,
        "end_exact_match_mask": end_exact_match_mask,
        "start_boundary_clamped_mask": start_boundary_clamped_mask,
        "end_boundary_clamped_mask": end_boundary_clamped_mask,
    }
    metadata = {
        "pair_id": pair.pair_id,
        "category_id": pair.category_id,
        "robot_episode_index": pair.robot_episode_index,
        "human_episode_index": pair.human_episode_index,
        "num_frames": num_frames,
        "num_pairings_per_frame": num_pairings,
        "total_frame_pairings": int(num_frames * num_pairings),
        "start_valid_count": int(np.count_nonzero(start_valid_mask)),
        "end_valid_count": int(np.count_nonzero(end_valid_mask)),
        "complete_valid_count": int(np.count_nonzero(start_valid_mask & end_valid_mask)),
        "start_boundary_clamped_count": int(np.count_nonzero(start_boundary_clamped_mask)),
        "end_boundary_clamped_count": int(np.count_nonzero(end_boundary_clamped_mask)),
        "start_exact_match_count": int(np.count_nonzero(start_exact_match_mask)),
        "end_exact_match_count": int(np.count_nonzero(end_exact_match_mask)),
        "start_reason_counts": dict(sorted(start_reason_counter.items())),
        "end_reason_counts": dict(sorted(end_reason_counter.items())),
        "stored_fields": sorted(arrays),
    }
    return arrays, metadata


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    paths = load_paths(config)
    processing = load_processing(config)
    pairs = filtered_pairs(load_pairs(config), args.pair_id)

    pairing_root = as_path(args.pairing_root) if args.pairing_root is not None else paths.output_root / "human_episode_pairings"
    output_root = as_path(args.output_root) if args.output_root is not None else paths.output_root / "exact_frame_human_u_matches"
    overwrite = bool(processing.overwrite or args.overwrite)
    output_root.mkdir(parents=True, exist_ok=True)
    save_config_snapshot(config, output_root, "resolved_config.json")

    run_summary: dict[str, Any] = {
        "stage": "exact_frame_human_u_matches",
        "pairing_root": str(pairing_root),
        "output_root": str(output_root),
        "num_pairs": len(pairs),
        "pairs": [],
    }

    for pair in tqdm(pairs, desc="exact-u matches", unit="pair"):
        pair_pairing_path = pairing_npz_path(pairing_root, pair)
        frame_indices, paired_human_episode_indices = load_pairing_arrays(pair_pairing_path)
        local_window_frame_indices, local_start_frame_index, local_end_frame_index = load_robot_local_windows(pair)
        if not np.array_equal(frame_indices, local_window_frame_indices):
            raise ValueError(f"frame_indices mismatch between pairings and local windows for pair {pair.pair_id}")

        robot_annotation = load_annotation_index(pair.robot_episode_dir / "checkpoints.json", pair.robot_episode_index)
        human_annotation = load_annotation_index(pair.human_episode_dir / "checkpoints.json", pair.human_episode_index)
        human_spline = load_human_spline_episode(pair.human_episode_dir / "spline.npz", pair.human_episode_index)

        arrays, metadata = build_exact_match_arrays(
            pair=pair,
            frame_indices=frame_indices,
            paired_human_episode_indices=paired_human_episode_indices,
            local_start_frame_index=local_start_frame_index,
            local_end_frame_index=local_end_frame_index,
            robot_annotation=robot_annotation,
            human_annotation=human_annotation,
            human_spline=human_spline,
            processing_index_dtype=processing.index_dtype,
            processing_progress_dtype=processing.progress_dtype,
            processing_u_dtype=processing.u_dtype,
        )

        out_dir = pair_output_dir(output_root, pair)
        ensure_output_dir(out_dir, overwrite=overwrite)
        output_path = out_dir / "exact_frame_human_u_matches.npz"
        np.savez_compressed(output_path, **arrays)
        output_metadata = dict(metadata)
        output_metadata["output_npz_path"] = str(output_path.resolve())
        if processing.write_episode_metadata_json:
            (out_dir / "exact_frame_human_u_matches_metadata.json").write_text(
                json.dumps(output_metadata, indent=2) + "\n",
                encoding="utf-8",
            )
        run_summary["pairs"].append(output_metadata)

    (output_root / "run_summary.json").write_text(json.dumps(run_summary, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

