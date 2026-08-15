from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from common import (
    PairConfig,
    as_path,
    ensure_output_dir,
    load_config,
    load_frame_indices_from_episode_dir,
    load_pairs,
    load_paths,
    load_processing,
    pair_output_dir,
    save_config_snapshot,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "unseen_val_config.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 2 for unseen validation: export deterministic robot-frame to fixed human-episode pairings."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--pair-id", action="append", default=[], help="Repeat to restrict export to named pair IDs.")
    return parser.parse_args()


def filtered_pairs(pairs: list[PairConfig], requested: list[str]) -> list[PairConfig]:
    if not requested:
        return pairs
    wanted = {value.strip() for value in requested}
    out = [pair for pair in pairs if pair.pair_id in wanted]
    if len(out) != len(wanted):
        missing = sorted(wanted - {pair.pair_id for pair in out})
        raise ValueError(f"Unknown requested pair IDs: {missing}")
    return out


def build_pairing_arrays(pair: PairConfig, num_pairings_per_frame: int, index_dtype: str) -> dict[str, np.ndarray]:
    frame_indices = load_frame_indices_from_episode_dir(pair.robot_episode_dir)
    paired = np.full(
        (frame_indices.shape[0], num_pairings_per_frame),
        fill_value=int(pair.human_episode_index),
        dtype=np.dtype(index_dtype),
    )
    return {
        "frame_indices": frame_indices.astype(np.dtype(index_dtype), copy=False),
        "paired_human_episode_indices": paired,
    }


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    paths = load_paths(config)
    processing = load_processing(config)
    pairs = filtered_pairs(load_pairs(config), args.pair_id)

    output_root = as_path(args.output_root) if args.output_root is not None else paths.output_root / "human_episode_pairings"
    overwrite = bool(processing.overwrite or args.overwrite)
    output_root.mkdir(parents=True, exist_ok=True)
    save_config_snapshot(config, output_root, "resolved_config.json")

    run_summary: dict[str, Any] = {
        "stage": "human_episode_pairings",
        "output_root": str(output_root),
        "num_pairs": len(pairs),
        "pairs": [],
    }

    for pair in tqdm(pairs, desc="pairings", unit="pair"):
        arrays = build_pairing_arrays(pair, processing.num_pairings_per_frame, processing.index_dtype)
        out_dir = pair_output_dir(output_root, pair)
        ensure_output_dir(out_dir, overwrite=overwrite)
        output_path = out_dir / "human_episode_pairings.npz"
        np.savez_compressed(output_path, **arrays)
        metadata = {
            "pair_id": pair.pair_id,
            "category_id": pair.category_id,
            "robot_episode_index": pair.robot_episode_index,
            "human_episode_index": pair.human_episode_index,
            "robot_episode_dir": str(pair.robot_episode_dir),
            "human_episode_dir": str(pair.human_episode_dir),
            "num_frames": int(arrays["frame_indices"].shape[0]),
            "num_pairings_per_frame": int(arrays["paired_human_episode_indices"].shape[1]),
            "output_npz_path": str(output_path.resolve()),
        }
        if processing.write_episode_metadata_json:
            (out_dir / "human_episode_pairings_metadata.json").write_text(
                json.dumps(metadata, indent=2) + "\n",
                encoding="utf-8",
            )
        run_summary["pairs"].append(metadata)

    (output_root / "run_summary.json").write_text(json.dumps(run_summary, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

