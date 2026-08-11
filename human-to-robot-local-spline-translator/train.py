from __future__ import annotations

import argparse
from pathlib import Path

from human_to_robot_local_spline_translator.config import load_config
from human_to_robot_local_spline_translator.training import launch_training


DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "default.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the local human-to-robot B-spline translator.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()
    checkpoint = launch_training(load_config(args.config, args.set), resume=args.resume)
    print(f"Training complete: {checkpoint}")


if __name__ == "__main__":
    main()
