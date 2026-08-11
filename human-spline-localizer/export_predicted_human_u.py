from __future__ import annotations

import argparse
from pathlib import Path

from human_spline_localizer.config import load_config
from human_spline_localizer.inference import export_predicted_human_u


DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "default.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the trained human spline localizer over the sim dataset and export predicted human u for all saved pairings.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--episode", action="append", type=int, default=[], help="Repeat to export only specific robot episodes.")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    output_root = export_predicted_human_u(
        load_config(args.config, args.set),
        checkpoint_path=args.checkpoint,
        output_root=args.output_root,
        batch_size=args.batch_size,
        overwrite=args.overwrite,
        episodes=args.episode,
        max_episodes=args.max_episodes,
        device=args.device,
    )
    print(f"Inference export complete: {output_root}")


if __name__ == "__main__":
    main()
