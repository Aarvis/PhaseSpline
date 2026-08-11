from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from tqdm.auto import tqdm


DEFAULT_CONFIG = Path(__file__).resolve().with_name("summarize_post_step4_human_local_segments.yaml")
EPSILON = 1e-8
TAIL_U_EPSILON = 1e-6
SUMMARY_PERCENTILES = (1, 2, 5, 15, 25, 50, 75, 85, 95, 99, 99.9)


@dataclass(frozen=True)
class PathsConfig:
    step4_root: Path
    step3_root: Path | None
    human_bspline_root: Path
    output_root: Path


@dataclass(frozen=True)
class ProcessingConfig:
    require_step3: bool
    skip_missing_human_bspline_episodes: bool
    write_per_sample_jsonl: bool
    write_per_sample_npz: bool


@dataclass(frozen=True)
class HumanSplineEpisode:
    episode_index: int
    frame_indices: np.ndarray
    frame_u: np.ndarray
    last_frame_index: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Post-step-4 utility: summarize validity, knot-count distribution, and covered-frame-count "
            "distribution for human local B-spline segments."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--step4-root", type=Path, default=None)
    parser.add_argument("--step3-root", type=Path, default=None)
    parser.add_argument("--human-bspline-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid config: {path}")
    return data


def as_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    return Path(value).expanduser().resolve()


def resolve_paths(config: dict[str, Any], args: argparse.Namespace) -> PathsConfig:
    paths = config.get("paths")
    if not isinstance(paths, dict):
        raise KeyError("Missing paths block in config.")
    step4_root = as_path(args.step4_root or paths.get("step4_root"))
    step3_root = as_path(args.step3_root or paths.get("step3_root"))
    human_bspline_root = as_path(args.human_bspline_root or paths.get("human_bspline_root"))
    output_root = as_path(args.output_root or paths.get("output_root"))
    if step4_root is None or human_bspline_root is None or output_root is None:
        raise KeyError("Config must define paths.step4_root, paths.human_bspline_root, and paths.output_root.")
    return PathsConfig(
        step4_root=step4_root,
        step3_root=step3_root,
        human_bspline_root=human_bspline_root,
        output_root=output_root,
    )


def resolve_processing(config: dict[str, Any]) -> ProcessingConfig:
    processing = config.get("processing", {})
    if not isinstance(processing, dict):
        raise KeyError("processing must be a mapping")
    return ProcessingConfig(
        require_step3=bool(processing.get("require_step3", True)),
        skip_missing_human_bspline_episodes=bool(processing.get("skip_missing_human_bspline_episodes", False)),
        write_per_sample_jsonl=bool(processing.get("write_per_sample_jsonl", False)),
        write_per_sample_npz=bool(processing.get("write_per_sample_npz", False)),
    )


def episode_index_from_parent(path: Path) -> int:
    return int(path.parent.name.split("_")[-1])


def step3_npz_path(root: Path, episode_index: int) -> Path:
    return root / f"chunk-{episode_index // 1000:03d}" / f"episode_{episode_index:06d}" / "frame_human_segment_progress_matches.npz"


def human_spline_npz_path(root: Path, episode_index: int) -> Path:
    return root / f"chunk-{episode_index // 1000:03d}" / f"episode_{episode_index:06d}" / "spline.npz"


def load_human_spline_episode(path: Path, episode_index: int) -> HumanSplineEpisode:
    with np.load(path, allow_pickle=False) as archive:
        frame_indices = np.asarray(archive["frame_indices"], dtype=np.int64)
        frame_u = np.asarray(archive["frame_u"], dtype=np.float64)
    if frame_indices.ndim != 1 or frame_u.ndim != 1 or frame_indices.shape[0] != frame_u.shape[0]:
        raise ValueError(f"Bad human spline frame arrays in {path}")
    return HumanSplineEpisode(
        episode_index=episode_index,
        frame_indices=frame_indices,
        frame_u=frame_u,
        last_frame_index=int(frame_indices[-1]) if frame_indices.size else -1,
    )


def summarize_distribution(values: np.ndarray) -> dict[str, float | int | None]:
    if values.size == 0:
        out: dict[str, float | int | None] = {"count": 0, "mean": None, "min": None}
        for p in SUMMARY_PERCENTILES:
            key = f"p{str(p).replace('.', '_')}"
            out[key] = None
        out["max"] = None
        return out
    arr = values.astype(np.float64, copy=False)
    out = {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "min": float(arr.min()),
    }
    for p in SUMMARY_PERCENTILES:
        key = f"p{str(p).replace('.', '_')}"
        out[key] = float(np.percentile(arr, p))
    out["median"] = float(np.percentile(arr, 50))
    out["max"] = float(arr.max())
    return out


def count_frames_in_interval(frame_u: np.ndarray, start_u: float, end_u: float) -> int:
    left = int(np.searchsorted(frame_u, start_u, side="left"))
    right = int(np.searchsorted(frame_u, end_u, side="right"))
    return max(0, right - left)


def classify_invalid_reason(
    start_match_valid: bool | None,
    end_match_valid: bool | None,
    start_u: float,
    end_u: float,
    local_interval_valid: bool,
    covered_frames: int | None,
) -> str:
    if start_match_valid is not None or end_match_valid is not None:
        if not bool(start_match_valid) and not bool(end_match_valid):
            return "missing_matched_start_and_end_frame"
        if not bool(start_match_valid):
            return "missing_matched_start_frame"
        if not bool(end_match_valid):
            return "missing_matched_end_frame"
    if not np.isfinite(start_u) and not np.isfinite(end_u):
        return "missing_matched_start_and_end_u"
    if not np.isfinite(start_u):
        return "missing_matched_start_u"
    if not np.isfinite(end_u):
        return "missing_matched_end_u"
    if end_u <= start_u + EPSILON:
        return "degenerate_interval_end_u_le_start_u"
    if not local_interval_valid:
        return "local_spline_restriction_failed"
    if covered_frames is not None and covered_frames <= 0:
        return "zero_frame_covered_interval"
    return "other_malformed_sample"


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    paths = resolve_paths(config, args)
    processing = resolve_processing(config)

    if not paths.step4_root.exists():
        raise FileNotFoundError(paths.step4_root)
    if processing.require_step3 and paths.step3_root is None:
        raise ValueError("step3_root is required when processing.require_step3 is true")
    if processing.require_step3 and paths.step3_root is not None and not paths.step3_root.exists():
        raise FileNotFoundError(paths.step3_root)
    if not paths.human_bspline_root.exists():
        raise FileNotFoundError(paths.human_bspline_root)

    step4_files = sorted(paths.step4_root.glob("chunk-*/episode_*/human_local_bspline_knot_counts.npz"))
    if not step4_files:
        raise RuntimeError(f"No step-4 NPZ files found under {paths.step4_root}")

    paths.output_root.mkdir(parents=True, exist_ok=True)

    human_cache: dict[int, HumanSplineEpisode | None] = {}
    invalid_reason_counts: dict[str, int] = {}
    tail_invalid_reason_counts: dict[str, int] = {}
    knot_counts: list[int] = []
    covered_frame_counts: list[int] = []
    valid_samples = 0
    invalid_samples = 0
    total_candidate_samples = 0
    tail_invalid_count = 0
    invalid_end_at_last_frame_count = 0
    invalid_end_u_near_one_count = 0
    invalid_tail_collapsed_count = 0
    per_sample_rows: list[dict[str, Any]] = []

    for step4_path in tqdm(step4_files, desc="episodes", unit="episode"):
        episode_index = episode_index_from_parent(step4_path)
        step3_arrays: dict[str, np.ndarray] | None = None
        if paths.step3_root is not None:
            step3_path = step3_npz_path(paths.step3_root, episode_index)
            if step3_path.exists():
                with np.load(step3_path, allow_pickle=False) as archive:
                    step3_arrays = {name: np.asarray(archive[name]) for name in archive.files}
            elif processing.require_step3:
                raise FileNotFoundError(step3_path)

        with np.load(step4_path, allow_pickle=False) as archive:
            frame_indices = np.asarray(archive["frame_indices"], dtype=np.int64)
            paired_human_episode_indices = np.asarray(archive["paired_human_episode_indices"], dtype=np.int64)
            human_start_frame_index = np.asarray(archive["human_start_frame_index"], dtype=np.int64)
            human_end_frame_index = np.asarray(archive["human_end_frame_index"], dtype=np.int64)
            human_local_start_u = np.asarray(archive["human_local_start_u"], dtype=np.float64)
            human_local_end_u = np.asarray(archive["human_local_end_u"], dtype=np.float64)
            local_interval_valid_mask = np.asarray(archive["local_interval_valid_mask"], dtype=bool)
            local_exact_knot_count = np.asarray(archive["local_exact_knot_count"], dtype=np.int64)

        num_frames, num_pairings = paired_human_episode_indices.shape
        total_candidate_samples += int(num_frames * num_pairings)

        for frame_pos in range(num_frames):
            for pairing_slot in range(num_pairings):
                human_episode_index = int(paired_human_episode_indices[frame_pos, pairing_slot])
                if human_episode_index not in human_cache:
                    human_path = human_spline_npz_path(paths.human_bspline_root, human_episode_index)
                    if human_path.exists():
                        human_cache[human_episode_index] = load_human_spline_episode(human_path, human_episode_index)
                    else:
                        if processing.skip_missing_human_bspline_episodes:
                            human_cache[human_episode_index] = None
                        else:
                            raise FileNotFoundError(human_path)
                human_episode = human_cache[human_episode_index]

                start_u = float(human_local_start_u[frame_pos, pairing_slot])
                end_u = float(human_local_end_u[frame_pos, pairing_slot])
                local_valid = bool(local_interval_valid_mask[frame_pos, pairing_slot])
                knot_count = int(local_exact_knot_count[frame_pos, pairing_slot])
                start_match_valid = None
                end_match_valid = None
                if step3_arrays is not None:
                    start_match_valid = bool(step3_arrays["start_match_valid_mask"][frame_pos, pairing_slot])
                    end_match_valid = bool(step3_arrays["end_match_valid_mask"][frame_pos, pairing_slot])

                covered_frames: int | None = None
                human_last_frame_index = None
                end_frame = int(human_end_frame_index[frame_pos, pairing_slot])
                if human_episode is not None and np.isfinite(start_u) and np.isfinite(end_u) and end_u > start_u + EPSILON:
                    covered_frames = count_frames_in_interval(human_episode.frame_u, start_u, end_u)
                    human_last_frame_index = human_episode.last_frame_index
                elif human_episode is not None:
                    human_last_frame_index = human_episode.last_frame_index

                is_valid = local_valid and covered_frames is not None and covered_frames > 0
                if is_valid:
                    valid_samples += 1
                    knot_counts.append(knot_count)
                    covered_frame_counts.append(int(covered_frames))
                else:
                    invalid_samples += 1
                    reason = classify_invalid_reason(
                        start_match_valid,
                        end_match_valid,
                        start_u,
                        end_u,
                        local_valid,
                        covered_frames,
                    )
                    invalid_reason_counts[reason] = invalid_reason_counts.get(reason, 0) + 1

                    end_at_last_frame = bool(human_last_frame_index is not None and end_frame == human_last_frame_index)
                    end_u_near_one = bool(np.isfinite(end_u) and end_u >= 1.0 - TAIL_U_EPSILON)
                    tail_collapsed = bool(end_at_last_frame and np.isfinite(start_u) and np.isfinite(end_u) and end_u <= start_u + EPSILON)
                    is_tail_invalid = end_at_last_frame or end_u_near_one or tail_collapsed
                    if is_tail_invalid:
                        tail_invalid_count += 1
                        tail_invalid_reason_counts[reason] = tail_invalid_reason_counts.get(reason, 0) + 1
                    if end_at_last_frame:
                        invalid_end_at_last_frame_count += 1
                    if end_u_near_one:
                        invalid_end_u_near_one_count += 1
                    if tail_collapsed:
                        invalid_tail_collapsed_count += 1

                if processing.write_per_sample_jsonl or processing.write_per_sample_npz:
                    per_sample_rows.append(
                        {
                            "robot_episode_index": episode_index,
                            "robot_frame_index": int(frame_indices[frame_pos]),
                            "pairing_slot": pairing_slot,
                            "human_episode_index": human_episode_index,
                            "human_start_frame_index": int(human_start_frame_index[frame_pos, pairing_slot]),
                            "human_end_frame_index": end_frame,
                            "human_local_start_u": start_u,
                            "human_local_end_u": end_u,
                            "local_interval_valid": local_valid,
                            "local_exact_knot_count": knot_count,
                            "covered_frame_count": covered_frames,
                            "is_valid": bool(is_valid),
                        }
                    )

    knot_array = np.asarray(knot_counts, dtype=np.int64)
    covered_frame_array = np.asarray(covered_frame_counts, dtype=np.int64)

    summary = {
        "step4_root": str(paths.step4_root),
        "step3_root": str(paths.step3_root) if paths.step3_root is not None else None,
        "human_bspline_root": str(paths.human_bspline_root),
        "episodes_scanned": len(step4_files),
        "total_candidate_samples": int(total_candidate_samples),
        "valid_samples": int(valid_samples),
        "invalid_samples": int(invalid_samples),
        "valid_fraction": float(valid_samples / total_candidate_samples) if total_candidate_samples > 0 else 0.0,
        "invalid_fraction": float(invalid_samples / total_candidate_samples) if total_candidate_samples > 0 else 0.0,
        "invalid_reason_counts": invalid_reason_counts,
        "tail_invalid_summary": {
            "tail_invalid_count": int(tail_invalid_count),
            "tail_invalid_fraction_of_invalid": float(tail_invalid_count / invalid_samples) if invalid_samples > 0 else 0.0,
            "invalid_end_at_last_frame_count": int(invalid_end_at_last_frame_count),
            "invalid_end_u_near_one_count": int(invalid_end_u_near_one_count),
            "invalid_tail_collapsed_count": int(invalid_tail_collapsed_count),
            "tail_invalid_reason_counts": tail_invalid_reason_counts,
        },
        "human_local_knot_count_distribution": summarize_distribution(knot_array),
        "human_local_covered_frame_count_distribution": summarize_distribution(covered_frame_array),
    }

    (paths.output_root / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    if processing.write_per_sample_jsonl:
        with (paths.output_root / "per_sample_details.jsonl").open("w", encoding="utf-8") as handle:
            for row in per_sample_rows:
                handle.write(json.dumps(row) + "\n")

    if processing.write_per_sample_npz:
        np.savez_compressed(
            paths.output_root / "per_sample_details.npz",
            robot_episode_index=np.asarray([row["robot_episode_index"] for row in per_sample_rows], dtype=np.int32),
            robot_frame_index=np.asarray([row["robot_frame_index"] for row in per_sample_rows], dtype=np.int32),
            pairing_slot=np.asarray([row["pairing_slot"] for row in per_sample_rows], dtype=np.int16),
            human_episode_index=np.asarray([row["human_episode_index"] for row in per_sample_rows], dtype=np.int32),
            human_start_frame_index=np.asarray([row["human_start_frame_index"] for row in per_sample_rows], dtype=np.int32),
            human_end_frame_index=np.asarray([row["human_end_frame_index"] for row in per_sample_rows], dtype=np.int32),
            human_local_start_u=np.asarray([row["human_local_start_u"] for row in per_sample_rows], dtype=np.float32),
            human_local_end_u=np.asarray([row["human_local_end_u"] for row in per_sample_rows], dtype=np.float32),
            local_interval_valid=np.asarray([row["local_interval_valid"] for row in per_sample_rows], dtype=bool),
            local_exact_knot_count=np.asarray([row["local_exact_knot_count"] for row in per_sample_rows], dtype=np.int32),
            covered_frame_count=np.asarray(
                [-1 if row["covered_frame_count"] is None else row["covered_frame_count"] for row in per_sample_rows],
                dtype=np.int32,
            ),
            is_valid=np.asarray([row["is_valid"] for row in per_sample_rows], dtype=bool),
        )

    print(json.dumps(summary, indent=2))
    print(f"Saved summary to {(paths.output_root / 'run_summary.json').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
