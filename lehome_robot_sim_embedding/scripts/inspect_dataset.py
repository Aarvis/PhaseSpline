from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq

from lehome_robot_sim_embedding.data import discover_episodes, read_dataset_info


def main() -> None:
    parser = argparse.ArgumentParser(description="Print robot sim dataset metadata and first parquet schema")
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    info = read_dataset_info(args.root)
    episodes = discover_episodes(args.root)
    first = episodes[0]
    print(json.dumps({key: info.get(key) for key in ("total_episodes", "total_frames", "fps")}, indent=2))
    print(json.dumps(info["features"], indent=2))
    print(f"first_episode={first.episode_index:06d} length={first.length}")
    print(pq.read_schema(first.path))


if __name__ == "__main__":
    main()

