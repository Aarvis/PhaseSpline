from __future__ import annotations

import argparse
from pathlib import Path

from lehome_robot_sim_embedding.config import load_config
from lehome_robot_sim_embedding.exporting import export_embeddings


DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "default.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description="Export robot sim posterior means/log-variances")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-episodes", type=int, default=None)
    args = parser.parse_args()
    output = export_embeddings(
        load_config(args.config, args.set),
        args.checkpoint,
        overwrite=args.overwrite,
        max_episodes=args.max_episodes,
    )
    print(f"Embeddings exported to {output}")


if __name__ == "__main__":
    main()

