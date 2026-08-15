from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import BSpline
from tqdm.auto import tqdm

from common import (
    EPSILON,
    INVALID_CODE_TO_REASON,
    INVALID_REASON_TO_CODE,
    PairConfig,
    as_path,
    ensure_output_dir,
    load_config,
    load_human_spline_episode,
    load_pairs,
    load_paths,
    load_processing,
    pair_output_dir,
    save_config_snapshot,
    summarize_distribution,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "unseen_val_config.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 4 for unseen validation: extract exact local human spline intervals and summarize validity/stats."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--match-root", type=Path, default=None)
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


def match_npz_path(match_root: Path, pair: PairConfig) -> Path:
    return pair_output_dir(match_root, pair) / "exact_frame_human_u_matches.npz"


def _count_knot_multiplicity(knots: np.ndarray, value: float) -> int:
    return int(np.count_nonzero(np.isclose(knots, value, atol=EPSILON, rtol=0.0)))


def _insert_to_boundary_multiplicity(spline: BSpline, value: float) -> BSpline:
    current_mult = _count_knot_multiplicity(np.asarray(spline.t, dtype=np.float64), value)
    required_additions = max(0, spline.k + 1 - current_mult)
    if required_additions == 0:
        return spline
    return spline.insert_knot(value, m=required_additions)


def restrict_bspline_exact(
    global_knots: np.ndarray,
    global_coefficients: np.ndarray,
    degree: int,
    u_start: float,
    u_end: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    if not np.isfinite(u_start) or not np.isfinite(u_end):
        return None
    if u_end <= u_start + EPSILON:
        return None
    spline = BSpline(
        np.asarray(global_knots, dtype=np.float64),
        np.asarray(global_coefficients, dtype=np.float64),
        degree,
        extrapolate=False,
    )
    spline = _insert_to_boundary_multiplicity(spline, float(u_start))
    spline = _insert_to_boundary_multiplicity(spline, float(u_end))
    knots = np.asarray(spline.t, dtype=np.float64)
    coefficients = np.asarray(spline.c, dtype=np.float64)
    start_matches = np.flatnonzero(np.isclose(knots, u_start, atol=EPSILON, rtol=0.0))
    end_matches = np.flatnonzero(np.isclose(knots, u_end, atol=EPSILON, rtol=0.0))
    if start_matches.size == 0 or end_matches.size == 0:
        raise RuntimeError("Failed to insert local interval boundaries into B-spline knot vector.")
    start_idx = int(start_matches[0])
    end_idx = int(end_matches[-1])
    local_knots = knots[start_idx : end_idx + 1]
    local_num_control_points = int(local_knots.shape[0] - degree - 1)
    if local_num_control_points <= 0:
        return None
    local_coefficients = coefficients[start_idx : start_idx + local_num_control_points]
    return local_knots, local_coefficients


def build_episode_stats(
    pair: PairConfig,
    human_spline,
    arrays: dict[str, np.ndarray],
    index_dtype: str,
    u_dtype: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    frame_indices = np.asarray(arrays["frame_indices"], dtype=np.int64)
    paired_human_episode_indices = np.asarray(arrays["paired_human_episode_indices"], dtype=np.int64)
    human_start_u = np.asarray(arrays["human_start_u"], dtype=np.float64)
    human_end_u = np.asarray(arrays["human_end_u"], dtype=np.float64)
    start_valid_mask = np.asarray(arrays["start_valid_mask"], dtype=bool)
    end_valid_mask = np.asarray(arrays["end_valid_mask"], dtype=bool)
    end_upper_frame_index = np.asarray(arrays["end_upper_frame_index"], dtype=np.int64)
    end_boundary_clamped_mask = np.asarray(arrays["end_boundary_clamped_mask"], dtype=bool)

    num_frames, num_pairings = paired_human_episode_indices.shape
    index_dt = np.dtype(index_dtype)
    u_dt = np.dtype(u_dtype)

    local_interval_valid_mask = np.zeros((num_frames, num_pairings), dtype=bool)
    local_exact_knot_count = np.full((num_frames, num_pairings), fill_value=-1, dtype=index_dt)
    local_unique_knot_count = np.full((num_frames, num_pairings), fill_value=-1, dtype=index_dt)
    local_control_point_count = np.full((num_frames, num_pairings), fill_value=-1, dtype=index_dt)
    human_local_start_u = np.full((num_frames, num_pairings), fill_value=np.nan, dtype=u_dt)
    human_local_end_u = np.full((num_frames, num_pairings), fill_value=np.nan, dtype=u_dt)
    human_local_covered_frame_count = np.full((num_frames, num_pairings), fill_value=-1, dtype=index_dt)
    invalid_reason_code = np.full(
        (num_frames, num_pairings),
        fill_value=INVALID_REASON_TO_CODE["malformed_interval"],
        dtype=np.int16,
    )
    tail_invalid_mask = np.zeros((num_frames, num_pairings), dtype=bool)

    reason_counter: Counter[str] = Counter()
    tail_reason_counter: Counter[str] = Counter()

    last_human_frame = int(human_spline.frame_indices[-1]) if human_spline.frame_indices.size else -1
    last_human_u = float(human_spline.frame_u[-1]) if human_spline.frame_u.size else float("nan")

    for row in tqdm(range(num_frames), desc=f"{pair.pair_id}: local spline stats", unit="frame", leave=False):
        for slot in range(num_pairings):
            start_u = float(human_start_u[row, slot])
            end_u = float(human_end_u[row, slot])
            human_local_start_u[row, slot] = np.asarray(start_u, dtype=u_dt)
            human_local_end_u[row, slot] = np.asarray(end_u, dtype=u_dt)

            if not bool(start_valid_mask[row, slot]) or not bool(end_valid_mask[row, slot]):
                reason = "missing_segment"
            elif not np.isfinite(start_u) or not np.isfinite(end_u):
                reason = "missing_frame_u"
            elif end_u <= start_u + EPSILON:
                reason = "degenerate_interval"
            else:
                covered_mask = (human_spline.frame_u >= start_u - EPSILON) & (human_spline.frame_u <= end_u + EPSILON)
                covered_count = int(np.count_nonzero(covered_mask))
                human_local_covered_frame_count[row, slot] = np.asarray(covered_count, dtype=index_dt)
                if covered_count <= 0:
                    reason = "zero_covered_frames"
                else:
                    local = restrict_bspline_exact(
                        human_spline.global_knots,
                        human_spline.global_coefficients,
                        human_spline.degree,
                        start_u,
                        end_u,
                    )
                    if local is None:
                        reason = "malformed_interval"
                    else:
                        local_knots, local_coefficients = local
                        local_interval_valid_mask[row, slot] = True
                        local_exact_knot_count[row, slot] = np.asarray(local_knots.shape[0], dtype=index_dt)
                        local_unique_knot_count[row, slot] = np.asarray(np.unique(local_knots).shape[0], dtype=index_dt)
                        local_control_point_count[row, slot] = np.asarray(local_coefficients.shape[0], dtype=index_dt)
                        invalid_reason_code[row, slot] = INVALID_REASON_TO_CODE["ok"]
                        reason_counter["ok"] += 1
                        continue

            invalid_reason_code[row, slot] = INVALID_REASON_TO_CODE[reason]
            reason_counter[reason] += 1
            tail_related = (
                bool(end_boundary_clamped_mask[row, slot])
                or int(end_upper_frame_index[row, slot]) == last_human_frame
                or (np.isfinite(end_u) and np.isfinite(last_human_u) and abs(end_u - last_human_u) <= 1e-6)
                or (np.isfinite(end_u) and end_u >= 1.0 - 1e-6)
            )
            if tail_related:
                tail_invalid_mask[row, slot] = True
                tail_reason_counter[reason] += 1

    valid_exact = local_exact_knot_count[local_interval_valid_mask].astype(np.int64, copy=False)
    valid_unique = local_unique_knot_count[local_interval_valid_mask].astype(np.int64, copy=False)
    valid_control = local_control_point_count[local_interval_valid_mask].astype(np.int64, copy=False)
    valid_covered = human_local_covered_frame_count[local_interval_valid_mask].astype(np.int64, copy=False)

    episode_arrays = {
        "frame_indices": frame_indices.astype(index_dt, copy=False),
        "paired_human_episode_indices": paired_human_episode_indices.astype(index_dt, copy=False),
        "human_local_start_u": human_local_start_u,
        "human_local_end_u": human_local_end_u,
        "local_interval_valid_mask": local_interval_valid_mask,
        "human_local_exact_knot_count": local_exact_knot_count,
        "human_local_unique_knot_count": local_unique_knot_count,
        "human_local_control_point_count": local_control_point_count,
        "human_local_covered_frame_count": human_local_covered_frame_count,
        "invalid_reason_code": invalid_reason_code,
        "tail_invalid_mask": tail_invalid_mask,
    }
    metadata = {
        "pair_id": pair.pair_id,
        "category_id": pair.category_id,
        "robot_episode_index": pair.robot_episode_index,
        "human_episode_index": pair.human_episode_index,
        "num_frames": int(num_frames),
        "num_pairings_per_frame": int(num_pairings),
        "total_candidate_samples": int(num_frames * num_pairings),
        "total_valid_samples": int(np.count_nonzero(local_interval_valid_mask)),
        "total_invalid_samples": int(np.count_nonzero(~local_interval_valid_mask)),
        "invalid_reason_counts": dict(sorted(reason_counter.items())),
        "tail_invalid_count": int(np.count_nonzero(tail_invalid_mask)),
        "tail_invalid_reason_counts": dict(sorted(tail_reason_counter.items())),
        "human_local_exact_knot_count_summary": summarize_distribution(valid_exact),
        "human_local_unique_knot_count_summary": summarize_distribution(valid_unique),
        "human_local_control_point_count_summary": summarize_distribution(valid_control),
        "human_local_covered_frame_count_summary": summarize_distribution(valid_covered),
        "stored_fields": sorted(episode_arrays),
    }
    return episode_arrays, metadata


def load_match_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    paths = load_paths(config)
    processing = load_processing(config)
    pairs = filtered_pairs(load_pairs(config), args.pair_id)

    match_root = as_path(args.match_root) if args.match_root is not None else paths.output_root / "exact_frame_human_u_matches"
    output_root = as_path(args.output_root) if args.output_root is not None else paths.output_root / "human_local_bspline_knot_count_stats"
    overwrite = bool(processing.overwrite or args.overwrite)
    output_root.mkdir(parents=True, exist_ok=True)
    save_config_snapshot(config, output_root, "resolved_config.json")

    run_summary: dict[str, Any] = {
        "stage": "human_local_bspline_knot_count_stats",
        "match_root": str(match_root),
        "output_root": str(output_root),
        "num_pairs": len(pairs),
        "pairs": [],
    }

    total_reason_counter: Counter[str] = Counter()
    total_tail_reason_counter: Counter[str] = Counter()
    total_candidate_samples = 0
    total_valid_samples = 0
    total_invalid_samples = 0

    for pair in tqdm(pairs, desc="local human spline stats", unit="pair"):
        human_spline = load_human_spline_episode(pair.human_episode_dir / "spline.npz", pair.human_episode_index)
        arrays = load_match_arrays(match_npz_path(match_root, pair))
        episode_arrays, metadata = build_episode_stats(
            pair=pair,
            human_spline=human_spline,
            arrays=arrays,
            index_dtype=processing.index_dtype,
            u_dtype=processing.u_dtype,
        )

        out_dir = pair_output_dir(output_root, pair)
        ensure_output_dir(out_dir, overwrite=overwrite)
        output_path = out_dir / "human_local_bspline_knot_count_stats.npz"
        np.savez_compressed(output_path, **episode_arrays)
        output_metadata = dict(metadata)
        output_metadata["output_npz_path"] = str(output_path.resolve())
        if processing.write_episode_metadata_json:
            (out_dir / "human_local_bspline_knot_count_stats_metadata.json").write_text(
                json.dumps(output_metadata, indent=2) + "\n",
                encoding="utf-8",
            )

        total_reason_counter.update(metadata["invalid_reason_counts"])
        total_tail_reason_counter.update(metadata["tail_invalid_reason_counts"])
        total_candidate_samples += int(metadata["total_candidate_samples"])
        total_valid_samples += int(metadata["total_valid_samples"])
        total_invalid_samples += int(metadata["total_invalid_samples"])
        run_summary["pairs"].append(output_metadata)

    run_summary["total_candidate_samples"] = total_candidate_samples
    run_summary["total_valid_samples"] = total_valid_samples
    run_summary["total_invalid_samples"] = total_invalid_samples
    run_summary["invalid_reason_counts"] = dict(sorted(total_reason_counter.items()))
    run_summary["tail_invalid_reason_counts"] = dict(sorted(total_tail_reason_counter.items()))

    (output_root / "run_summary.json").write_text(json.dumps(run_summary, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

