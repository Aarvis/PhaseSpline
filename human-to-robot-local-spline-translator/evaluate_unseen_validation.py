from __future__ import annotations

import argparse
from pathlib import Path

from human_to_robot_local_spline_translator.unseen_eval import evaluate_unseen_validation


DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "unseen_validation_eval.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate translator on unseen-validation pairs using exact GT human intervals.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Path to unseen-validation eval YAML config.")
    parser.add_argument("--set", dest="overrides", nargs="*", default=None, help="Optional KEY=VALUE config overrides.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = evaluate_unseen_validation(args.config, args.overrides)
    print(f"[unseen-eval] outputs saved to: {output_root}")


if __name__ == "__main__":
    main()

