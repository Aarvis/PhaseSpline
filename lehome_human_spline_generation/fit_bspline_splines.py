from __future__ import annotations

import argparse
import json
from pathlib import Path

from lehome_spline.bspline import fit_all_bspline_splines
from lehome_spline.config import load_config


DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "joint_full_epoch_bspline.yaml"


def _print_run_summary(output: Path) -> None:
    summary_path = output / "run_summary.json"
    if not summary_path.is_file():
        print(f"No run summary found at {summary_path}")
        return
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)

    episode_results = summary.get("episode_results", [])
    episodes = int(summary.get("episodes", len(episode_results)))
    total_frames = int(summary.get("total_frames", 0))
    total_internal_knots = int(summary.get("total_internal_knots", 0))
    total_control_points = int(summary.get("total_control_points", 0))
    satisfied = int(summary.get("tolerance_satisfied", 0))
    controls_per_frame = total_control_points / total_frames if total_frames else 0.0

    max_internal_knots = max((int(item["num_internal_knots"]) for item in episode_results), default=0)
    max_control_points = max((int(item["num_control_points"]) for item in episode_results), default=0)
    max_epsilon = max((float(item.get("maximum_epsilon_ratio", 0.0)) for item in episode_results), default=0.0)
    max_latent = max((float(item["maximum_latent_rmse"]) for item in episode_results), default=0.0)
    max_cosine = max((float(item["maximum_cosine_distance"]) for item in episode_results), default=0.0)
    max_state = max((float(item["maximum_state_rmse"]) for item in episode_results), default=0.0)

    print("B-spline fit summary")
    print(f"  episodes: {episodes}")
    print(f"  tolerance_satisfied: {satisfied}/{episodes}")
    print(f"  total_frames: {total_frames}")
    print(f"  total_internal_knots: {total_internal_knots}")
    print(f"  total_control_points: {total_control_points}")
    print(f"  control_points/frame: {controls_per_frame:.4f}")
    print(f"  max_internal_knots_fitted: {max_internal_knots}")
    print(f"  max_control_points_fitted: {max_control_points}")
    print(f"  max_epsilon_ratio: {max_epsilon:.6g}")
    print(f"  max_latent_rmse: {max_latent:.6g}")
    print(f"  max_cosine_distance: {max_cosine:.6g}")
    print(f"  max_state_rmse: {max_state:.6g}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit adaptive cubic B-splines over human unified 2048D embeddings.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--save-mode", choices=("dataset", "external"), default=None)
    args = parser.parse_args()
    output = fit_all_bspline_splines(
        load_config(args.config, args.set),
        overwrite=args.overwrite,
        max_episodes=args.max_episodes,
        save_mode=args.save_mode,
    )
    print(f"B-splines written to {output}")
    _print_run_summary(output)


if __name__ == "__main__":
    main()
