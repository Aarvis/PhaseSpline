from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from lehome_robot_sim_embedding.data import ACTION_COLUMN, STATE_COLUMN, VIEW_COLUMNS, discover_episodes


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect one robot sim episode")
    parser.add_argument("root", type=Path)
    parser.add_argument("--episode-index", type=int, default=0)
    args = parser.parse_args()
    records = {record.episode_index: record for record in discover_episodes(args.root)}
    record = records[args.episode_index]
    columns = [*VIEW_COLUMNS.values(), STATE_COLUMN, ACTION_COLUMN, "timestamp", "frame_index"]
    table = pq.read_table(record.path, columns=columns)
    state = np.asarray(table[STATE_COLUMN].to_pylist(), dtype=np.float32)
    action = np.asarray(table[ACTION_COLUMN].to_pylist(), dtype=np.float32)
    print(f"path={record.path}")
    print(f"frames={table.num_rows} state_dim={state.shape[1]} action_dim={action.shape[1]}")
    print(f"timestamp_range=({table['timestamp'][0].as_py()}, {table['timestamp'][-1].as_py()})")
    for view, column in VIEW_COLUMNS.items():
        first = table[column][0].as_py()
        print(f"{view}_has_bytes={bool(first.get('bytes'))} path={first.get('path')}")


if __name__ == "__main__":
    main()

