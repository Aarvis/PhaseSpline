from __future__ import annotations

import argparse
import json
from pathlib import Path

from lehome_spline.config import load_config
from lehome_spline.spline import fit_all_splines


DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "default.yaml"


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
    total_knots = int(summary.get("total_knots", 0))
    satisfied = int(summary.get("tolerance_satisfied", 0))
    knot_frame_ratio = total_knots / total_frames if total_frames else 0.0

    max_knots = max((int(item["knots"]) for item in episode_results), default=0)
    max_epsilon = max((float(item.get("maximum_epsilon_ratio", 0.0)) for item in episode_results), default=0.0)
    max_latent = max((float(item["maximum_latent_rmse"]) for item in episode_results), default=0.0)
    max_cosine = max((float(item["maximum_cosine_distance"]) for item in episode_results), default=0.0)
    max_state = max((float(item["maximum_state_rmse"]) for item in episode_results), default=0.0)
    mean_latent_oor = float(summary.get("mean_latent_out_of_range_percent", 0.0))
    max_latent_oor = float(summary.get("maximum_latent_out_of_range_percent", 0.0))
    max_latent_overshoot = float(summary.get("maximum_latent_overshoot_std", 0.0))
    mean_state_oor = float(summary.get("mean_state_out_of_range_percent", 0.0))
    max_state_oor = float(summary.get("maximum_state_out_of_range_percent", 0.0))
    max_state_overshoot = float(summary.get("maximum_state_overshoot_std", 0.0))

    print("Spline fit summary")
    print(f"  episodes: {episodes}")
    print(f"  tolerance_satisfied: {satisfied}/{episodes}")
    print(f"  total_frames: {total_frames}")
    print(f"  total_knots: {total_knots}")
    print(f"  knots/frame: {knot_frame_ratio:.4f}")
    print(f"  max_knots_fitted: {max_knots}")
    print(f"  max_epsilon_ratio: {max_epsilon:.6g}")
    print(f"  max_latent_rmse: {max_latent:.6g}")
    print(f"  max_cosine_distance: {max_cosine:.6g}")
    print(f"  max_state_rmse: {max_state:.6g}")
    print(f"  mean_latent_out_of_range_percent: {mean_latent_oor:.4f}")
    print(f"  max_latent_out_of_range_percent: {max_latent_oor:.4f}")
    print(f"  max_latent_overshoot_range_fraction: {max_latent_overshoot:.6g}")
    print(f"  mean_state_out_of_range_percent: {mean_state_oor:.4f}")
    print(f"  max_state_out_of_range_percent: {max_state_oor:.4f}")
    print(f"  max_state_overshoot_range_fraction: {max_state_overshoot:.6g}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit adaptive shared-knot visual splines")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-episodes", type=int, default=None)
    args = parser.parse_args()
    output = fit_all_splines(
        load_config(args.config, args.set),
        overwrite=args.overwrite,
        max_episodes=args.max_episodes,
    )
    print(f"Splines written to {output}")
    _print_run_summary(output)


if __name__ == "__main__":
    main()
