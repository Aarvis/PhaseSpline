from __future__ import annotations

import argparse
from pathlib import Path

from lehome_robot_sim_embedding.config import load_config
from lehome_robot_sim_embedding.training import train


DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "default.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LeHome robot sim multi-view VAE embeddings")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument("--resume", default=None)
    checkpoint_group.add_argument("--init-checkpoint", default=None)
    args = parser.parse_args()
    checkpoint = train(
        load_config(args.config, args.set),
        resume=args.resume,
        init_checkpoint=args.init_checkpoint,
    )
    print(f"Training complete: {checkpoint}")


if __name__ == "__main__":
    main()

