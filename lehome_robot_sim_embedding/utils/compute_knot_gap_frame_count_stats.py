from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from tqdm.auto import tqdm


DEFAULT_CONFIG = Path(__file__).resolve().with_name("compute_knot_gap_frame_count_stats.yaml")


def load_config(path: str | Path) -> dict:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise ValueError(f"Configuration root must be a mapping: {config_path}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute the distribution of frame counts covered by knot-span intervals in saved robot-sim B-splines."
        )
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to a YAML config file.")
    parser.add_argument(
        "--bspline-root",
        default=None,
        help="Root directory containing chunk-XXX/episode_XXXXXX/spline.npz files.",
    )
    parser.add_argument(
        "--node-gap",
        type=int,
        default=None,
        help=(
            "Gap in node-span indices. For node_gap=10, the interval is from the start of node i "
            "to the end of node i+10, so there are 9 node spans strictly in between."
        ),
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to save the computed summary as JSON.",
    )
    return parser.parse_args()


def percentile_dict(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(arr)),
        "p1": float(np.percentile(arr, 1)),
        "p2": float(np.percentile(arr, 2)),
        "p5": float(np.percentile(arr, 5)),
        "p10": float(np.percentile(arr, 10)),
        "p15": float(np.percentile(arr, 15)),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p85": float(np.percentile(arr, 85)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "p99_9": float(np.percentile(arr, 99.9)),
        "max": float(np.max(arr)),
    }


def percentile_targets(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(arr)),
        "p1": float(np.percentile(arr, 1)),
        "p2": float(np.percentile(arr, 2)),
        "p5": float(np.percentile(arr, 5)),
        "p10": float(np.percentile(arr, 10)),
        "p15": float(np.percentile(arr, 15)),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p85": float(np.percentile(arr, 85)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "p99_9": float(np.percentile(arr, 99.9)),
        "max": float(np.max(arr)),
    }


def count_frames_in_interval(sorted_frame_u: np.ndarray, start_u: float, end_u: float) -> int:
    left = int(np.searchsorted(sorted_frame_u, start_u, side="left"))
    right = int(np.searchsorted(sorted_frame_u, end_u, side="right"))
    return right - left


def main() -> int:
    args = parse_args()
    config = load_config(args.config)

    bspline_root_value = args.bspline_root or config.get("bspline_root")
    if not bspline_root_value:
        raise ValueError("bspline_root must be provided either in the config or via --bspline-root")
    bspline_root = Path(bspline_root_value).expanduser().resolve()

    node_gap = int(args.node_gap if args.node_gap is not None else config.get("node_gap", 10))
    if node_gap < 0:
        raise ValueError("--node-gap must be non-negative")

    sources = sorted(bspline_root.glob("chunk-*/episode_*/spline.npz"))
    if not sources:
        sources = sorted(bspline_root.glob("episode_*/spline.npz"))
    if not sources:
        raise FileNotFoundError(f"No spline.npz files found under {bspline_root}")

    all_counts: list[np.ndarray] = []
    all_episode_indices: list[np.ndarray] = []
    all_episode_total_frames: list[np.ndarray] = []
    all_episode_total_knots: list[np.ndarray] = []
    all_episode_total_control_points: list[np.ndarray] = []
    all_interval_start_nodes: list[np.ndarray] = []
    episodes_used = 0
    episodes_skipped_too_short = 0
    interval_count = 0

    for source in tqdm(sources, desc="episodes", unit="episode"):
        with np.load(source, allow_pickle=False) as data:
            frame_u = np.asarray(data["frame_u"], dtype=np.float64)
            global_knots = np.asarray(data["global_knots"], dtype=np.float64)
            global_coefficients = np.asarray(data["global_coefficients"], dtype=np.float32)

        if frame_u.ndim != 1 or global_knots.ndim != 1:
            raise ValueError(f"Invalid array rank in {source}")
        if frame_u.size == 0:
            episodes_skipped_too_short += 1
            continue

        unique_knots = np.unique(global_knots)
        if unique_knots.size < node_gap + 2:
            episodes_skipped_too_short += 1
            continue

        counts = np.empty(unique_knots.size - node_gap - 1, dtype=np.int32)
        for start_index in range(counts.shape[0]):
            start_u = float(unique_knots[start_index])
            end_u = float(unique_knots[start_index + node_gap + 1])
            counts[start_index] = count_frames_in_interval(frame_u, start_u, end_u)

        all_counts.append(counts)
        episode_index = int(source.parent.name.split("_")[-1])
        episode_total_frames = int(frame_u.shape[0])
        episode_total_knots = int(global_knots.shape[0])
        episode_total_control_points = int(global_coefficients.shape[0])
        all_episode_indices.append(np.full(counts.shape, episode_index, dtype=np.int32))
        all_episode_total_frames.append(np.full(counts.shape, episode_total_frames, dtype=np.int32))
        all_episode_total_knots.append(np.full(counts.shape, episode_total_knots, dtype=np.int32))
        all_episode_total_control_points.append(np.full(counts.shape, episode_total_control_points, dtype=np.int32))
        all_interval_start_nodes.append(np.arange(counts.shape[0], dtype=np.int32))
        episodes_used += 1
        interval_count += int(counts.shape[0])

    if not all_counts:
        raise RuntimeError(f"No valid node-gap intervals found for node_gap={node_gap} under {bspline_root}")

    merged = np.concatenate(all_counts, axis=0).astype(np.float64, copy=False)
    merged_episode_indices = np.concatenate(all_episode_indices, axis=0)
    merged_episode_total_frames = np.concatenate(all_episode_total_frames, axis=0)
    merged_episode_total_knots = np.concatenate(all_episode_total_knots, axis=0)
    merged_episode_total_control_points = np.concatenate(all_episode_total_control_points, axis=0)
    merged_interval_start_nodes = np.concatenate(all_interval_start_nodes, axis=0)

    detailed_distribution: dict[str, dict[str, float | int]] = {}
    for label, target_value in percentile_targets(merged).items():
        if label == "min":
            selected_index = int(np.argmin(merged))
        elif label == "max":
            selected_index = int(np.argmax(merged))
        else:
            selected_index = int(np.argmin(np.abs(merged - target_value)))
        detailed_distribution[label] = {
            "frame_count": int(merged[selected_index]),
            "target_percentile_value": float(target_value),
            "episode_index": int(merged_episode_indices[selected_index]),
            "episode_total_frames": int(merged_episode_total_frames[selected_index]),
            "episode_total_knots": int(merged_episode_total_knots[selected_index]),
            "episode_total_control_points": int(merged_episode_total_control_points[selected_index]),
            "interval_start_node_index": int(merged_interval_start_nodes[selected_index]),
            "interval_end_node_index": int(merged_interval_start_nodes[selected_index] + node_gap),
        }

    summary = {
        "config_path": str(Path(args.config).expanduser().resolve()),
        "bspline_root": str(bspline_root),
        "node_gap": node_gap,
        "interval_definition": (
            "Each node is a unique knot span [u_i, u_{i+1}]. "
            "For node_gap=g, each interval is [u_i, u_{i+g+1}], i.e. from the start of node i "
            "to the end of node i+g."
        ),
        "episodes_scanned": len(sources),
        "episodes_used": episodes_used,
        "episodes_skipped_too_short": episodes_skipped_too_short,
        "interval_count": interval_count,
        "frame_count_distribution": percentile_dict(merged),
        "frame_count_distribution_detailed": detailed_distribution,
    }

    print(json.dumps(summary, indent=2))

    output_json_value = args.output_json if args.output_json is not None else config.get("output_json")
    if output_json_value:
        output_path = Path(output_json_value).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Saved summary to {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
