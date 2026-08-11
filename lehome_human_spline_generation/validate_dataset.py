from __future__ import annotations

import argparse
import json
from pathlib import Path

from lehome_spline.config import load_config
from lehome_spline.data import validate_dataset
from lehome_spline.utils import atomic_json_dump


DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "default.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the LeHome Parquet dataset contract")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    config = load_config(args.config, args.set)
    report = validate_dataset(
        config["dataset"]["root"],
        max_episodes=args.max_episodes,
        atol=float(config["dataset"]["action_next_state_atol"]),
    )
    if args.output:
        atomic_json_dump(report, args.output)
    print(json.dumps(report, indent=2))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
