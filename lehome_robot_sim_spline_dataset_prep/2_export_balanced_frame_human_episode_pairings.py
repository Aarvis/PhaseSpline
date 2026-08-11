from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency
    yaml = None


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "2_human_episode_pairings_config.yaml"


@dataclass(frozen=True)
class PathsConfig:
    sim_dataset_root: Path
    human_dataset_root: Path
    robot_frame_source_root: Path
    output_root: Path


@dataclass(frozen=True)
class ProcessingConfig:
    seed: int
    overwrite: bool
    index_dtype: str
    num_pairings_per_frame: int
    max_robot_episodes_per_category: int | None


@dataclass(frozen=True)
class CategoryConfig:
    category_id: str
    sim_start: int
    sim_end_exclusive: int
    human_start: int
    human_end_exclusive: int


@dataclass(frozen=True)
class RobotEpisodeRecord:
    episode_index: int
    frame_indices: np.ndarray
    num_frames: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a balanced frame-level mapping from robot sim frames to random human episode IDs "
            "within the same garment category."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--category", action="append", default=[], help="Repeat to limit export to named categories.")
    parser.add_argument("--num-pairings-per-frame", type=int, default=None)
    parser.add_argument("--max-robot-episodes-per-category", type=int, default=None)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if yaml is None:
        raise ImportError("PyYAML is required for YAML configs.")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid config: {path}")
    return data


def as_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def canonical_category_id(value: str) -> str:
    return str(value).strip().lower()


def load_paths(config: dict[str, Any], args: argparse.Namespace) -> PathsConfig:
    paths = config.get("paths")
    if not isinstance(paths, dict):
        raise KeyError("Missing paths block in config.")
    seed = int(args.seed) if args.seed is not None else int(config.get("processing", {}).get("seed", 0))
    if args.output_root is not None:
        output_root = as_path(args.output_root)
    else:
        template = paths.get("output_root_template") or paths.get("output_root")
        if not template:
            raise KeyError("Config must define paths.output_root_template or paths.output_root.")
        output_root = as_path(str(template).format(seed=seed))
    return PathsConfig(
        sim_dataset_root=as_path(paths["sim_dataset_root"]),
        human_dataset_root=as_path(paths["human_dataset_root"]),
        robot_frame_source_root=as_path(paths.get("robot_bspline_root") or paths["robot_fitted_spline_root"]),
        output_root=output_root,
    )


def load_processing(config: dict[str, Any], args: argparse.Namespace) -> ProcessingConfig:
    processing = config.get("processing")
    if not isinstance(processing, dict):
        raise KeyError("Missing processing block in config.")
    seed = int(args.seed) if args.seed is not None else int(processing.get("seed", 0))
    num_pairings_per_frame = (
        int(args.num_pairings_per_frame)
        if args.num_pairings_per_frame is not None
        else int(processing.get("num_pairings_per_frame", 4))
    )
    if num_pairings_per_frame <= 0:
        raise ValueError("num_pairings_per_frame must be positive.")
    max_robot_episodes = (
        int(args.max_robot_episodes_per_category)
        if args.max_robot_episodes_per_category is not None
        else (int(processing["max_robot_episodes_per_category"]) if processing.get("max_robot_episodes_per_category") is not None else None)
    )
    return ProcessingConfig(
        seed=seed,
        overwrite=bool(processing.get("overwrite", False) or args.overwrite),
        index_dtype=str(processing.get("index_dtype", "int32")),
        num_pairings_per_frame=num_pairings_per_frame,
        max_robot_episodes_per_category=max_robot_episodes,
    )


def load_categories(config: dict[str, Any], requested: list[str]) -> list[CategoryConfig]:
    categories_block = config.get("categories")
    if not isinstance(categories_block, list):
        raise KeyError("Missing categories list in config.")
    categories = []
    for item in categories_block:
        if not isinstance(item, dict):
            raise ValueError(f"Bad category entry: {item!r}")
        categories.append(
            CategoryConfig(
                category_id=canonical_category_id(item["id"]),
                sim_start=int(item["sim_start"]),
                sim_end_exclusive=int(item["sim_end_exclusive"]),
                human_start=int(item["human_start"]),
                human_end_exclusive=int(item["human_end_exclusive"]),
            )
        )
    if requested:
        wanted = {canonical_category_id(value) for value in requested}
        categories = [item for item in categories if item.category_id in wanted]
        if len(categories) != len(wanted):
            missing = sorted(wanted - {item.category_id for item in categories})
            raise ValueError(f"Unknown requested categories: {missing}")
    return categories


def dataset_data_template(dataset_root: Path) -> str:
    info_path = dataset_root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    template = info.get("data_path")
    if not template:
        raise KeyError(f"meta/info.json for {dataset_root} does not define data_path")
    return str(template)


def dataset_episode_data_path(dataset_root: Path, data_template: str, episode_index: int) -> Path:
    rel = data_template.format(episode_chunk=episode_index // 1000, episode_index=episode_index)
    return (dataset_root / rel).resolve()


def robot_frame_source_path(frame_source_root: Path, episode_index: int) -> Path:
    return frame_source_root / f"chunk-{episode_index // 1000:03d}" / f"episode_{episode_index:06d}" / "spline.npz"


def output_episode_dir(output_root: Path, episode_index: int) -> Path:
    return output_root / f"chunk-{episode_index // 1000:03d}" / f"episode_{episode_index:06d}"


def ensure_output_episode_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory already contains files, pass overwrite to replace it: {path}")
    path.mkdir(parents=True, exist_ok=True)


def available_human_episode_ids(
    dataset_root: Path,
    data_template: str,
    category: CategoryConfig,
) -> list[int]:
    out = []
    for episode_index in range(category.human_start, category.human_end_exclusive):
        if dataset_episode_data_path(dataset_root, data_template, episode_index).exists():
            out.append(episode_index)
    return out


def robot_episode_records(
    frame_source_root: Path,
    category: CategoryConfig,
    max_robot_episodes_per_category: int | None,
) -> list[RobotEpisodeRecord]:
    episode_ids = list(range(category.sim_start, category.sim_end_exclusive))
    if max_robot_episodes_per_category is not None:
        episode_ids = episode_ids[:max_robot_episodes_per_category]
    records: list[RobotEpisodeRecord] = []
    for episode_index in tqdm(
        episode_ids,
        desc=f"{category.category_id}: robot episodes",
        unit="episode",
        leave=False,
    ):
        path = robot_frame_source_path(frame_source_root, episode_index)
        if not path.exists():
            continue
        with np.load(path, allow_pickle=False) as archive:
            frame_indices = np.asarray(archive["frame_indices"], dtype=np.int64)
        records.append(
            RobotEpisodeRecord(
                episode_index=episode_index,
                frame_indices=frame_indices,
                num_frames=int(frame_indices.shape[0]),
            )
        )
    return records


def build_balanced_assignment(
    human_episode_ids: list[int],
    total_robot_frames: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if not human_episode_ids:
        raise ValueError("No human episodes available for balanced assignment.")
    human_ids = np.asarray(human_episode_ids, dtype=np.int64)
    base_repeats = total_robot_frames // human_ids.shape[0]
    remainder = total_robot_frames % human_ids.shape[0]

    pieces = []
    if base_repeats > 0:
        pieces.append(np.repeat(human_ids, base_repeats))
    if remainder > 0:
        shuffled = human_ids.copy()
        rng.shuffle(shuffled)
        pieces.append(shuffled[:remainder])
    if not pieces:
        raise RuntimeError("Failed to create assignment pool.")

    assignment = np.concatenate(pieces, axis=0)
    rng.shuffle(assignment)
    return assignment


def _repair_collisions_in_bank(
    bank: np.ndarray,
    previous_banks: np.ndarray,
    rng: np.random.Generator,
) -> bool:
    if previous_banks.size == 0:
        return True
    num_rows = bank.shape[0]
    order = np.arange(num_rows, dtype=np.int64)
    for _ in range(8):
        collisions = np.flatnonzero(np.any(previous_banks == bank[:, None], axis=1))
        if collisions.size == 0:
            return True
        rng.shuffle(order)
        progress = False
        for row_idx in collisions.tolist():
            current_value = int(bank[row_idx])
            prev_row = previous_banks[row_idx]
            if current_value not in prev_row:
                continue
            for swap_idx in order.tolist():
                if swap_idx == row_idx:
                    continue
                swap_value = int(bank[swap_idx])
                prev_swap = previous_banks[swap_idx]
                if swap_value in prev_row:
                    continue
                if current_value in prev_swap:
                    continue
                bank[row_idx], bank[swap_idx] = bank[swap_idx], bank[row_idx]
                progress = True
                break
        if not progress:
            break
    return not np.any(previous_banks == bank[:, None])


def build_multi_balanced_assignment(
    human_episode_ids: list[int],
    total_robot_frames: int,
    num_pairings_per_frame: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if len(human_episode_ids) < num_pairings_per_frame:
        raise ValueError(
            f"Need at least {num_pairings_per_frame} human episodes to assign distinct pairings per frame, "
            f"found {len(human_episode_ids)}."
        )
    banks: list[np.ndarray] = []
    max_attempts_per_bank = 128
    for _slot in range(num_pairings_per_frame):
        previous = np.column_stack(banks) if banks else np.zeros((total_robot_frames, 0), dtype=np.int64)
        accepted = None
        for _attempt in range(max_attempts_per_bank):
            candidate = build_balanced_assignment(human_episode_ids, total_robot_frames, rng).copy()
            if _repair_collisions_in_bank(candidate, previous, rng):
                accepted = candidate
                break
        if accepted is None:
            raise RuntimeError(
                "Failed to build distinct balanced human-episode pairings per frame after repeated attempts."
            )
        banks.append(accepted)
    assignment = np.column_stack(banks)
    if assignment.shape != (total_robot_frames, num_pairings_per_frame):
        raise RuntimeError("Unexpected assignment matrix shape.")
    if np.any(np.diff(np.sort(assignment, axis=1), axis=1) == 0):
        raise RuntimeError("Distinctness constraint violated: duplicate human episode within a frame row.")
    return assignment


def category_usage_summary(assignment: np.ndarray) -> dict[str, float | int]:
    counts = Counter(int(item) for item in assignment.reshape(-1).tolist())
    usage_values = np.asarray(sorted(counts.values()), dtype=np.float64)
    return {
        "human_episodes_used": len(counts),
        "min_usage_per_human_episode": int(usage_values.min()),
        "mean_usage_per_human_episode": float(usage_values.mean()),
        "median_usage_per_human_episode": float(np.median(usage_values)),
        "max_usage_per_human_episode": int(usage_values.max()),
    }


def save_episode_pairing(
    output_root: Path,
    record: RobotEpisodeRecord,
    category_id: str,
    paired_human_episode_indices: np.ndarray,
    processing: ProcessingConfig,
    human_dataset_root: Path,
) -> dict[str, Any]:
    out_dir = output_episode_dir(output_root, record.episode_index)
    ensure_output_episode_dir(out_dir, overwrite=processing.overwrite)

    arrays = {
        "frame_indices": np.asarray(record.frame_indices, dtype=np.dtype(processing.index_dtype)),
        "paired_human_episode_indices": np.asarray(paired_human_episode_indices, dtype=np.dtype(processing.index_dtype)),
    }
    np.savez_compressed(out_dir / "human_episode_pairings.npz", **arrays)

    metadata = {
        "robot_episode_index": record.episode_index,
        "robot_category_id": category_id,
        "num_frames": record.num_frames,
        "num_pairings_per_frame": int(processing.num_pairings_per_frame),
        "distinct_within_frame": True,
        "human_dataset_root": str(human_dataset_root),
        "output_npz_path": str((out_dir / "human_episode_pairings.npz").resolve()),
        "stored_fields": sorted(arrays),
    }
    (out_dir / "human_episode_pairings_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    paths = load_paths(config, args)
    processing = load_processing(config, args)
    categories = load_categories(config, args.category)

    if not paths.sim_dataset_root.exists():
        raise FileNotFoundError(paths.sim_dataset_root)
    if not paths.human_dataset_root.exists():
        raise FileNotFoundError(paths.human_dataset_root)
    if not paths.robot_frame_source_root.exists():
        raise FileNotFoundError(paths.robot_frame_source_root)

    human_data_template = dataset_data_template(paths.human_dataset_root)
    paths.output_root.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(processing.seed)

    run_summary: dict[str, Any] = {
        "sim_dataset_root": str(paths.sim_dataset_root),
        "human_dataset_root": str(paths.human_dataset_root),
        "robot_frame_source_root": str(paths.robot_frame_source_root),
        "output_root": str(paths.output_root),
        "seed": processing.seed,
        "num_pairings_per_frame": processing.num_pairings_per_frame,
        "categories": [],
    }

    print("Exporting balanced frame-to-human episode pairings")
    print(f"  robot_frame_source_root  : {paths.robot_frame_source_root}")
    print(f"  human_dataset_root       : {paths.human_dataset_root}")
    print(f"  output_root              : {paths.output_root}")
    print(f"  seed                     : {processing.seed}")
    print(f"  num_pairings_per_frame   : {processing.num_pairings_per_frame}")
    print(f"  categories               : {', '.join(item.category_id for item in categories)}")

    for category in tqdm(categories, desc="categories", unit="category"):
        human_episode_ids = available_human_episode_ids(paths.human_dataset_root, human_data_template, category)
        if not human_episode_ids:
            raise RuntimeError(f"No available human episodes for category {category.category_id}")

        robot_records = robot_episode_records(
            paths.robot_frame_source_root,
            category,
            processing.max_robot_episodes_per_category,
        )
        if not robot_records:
            raise RuntimeError(f"No available robot frame-source episodes for category {category.category_id}")

        total_robot_frames = sum(record.num_frames for record in robot_records)
        assignment = build_multi_balanced_assignment(
            human_episode_ids,
            total_robot_frames,
            processing.num_pairings_per_frame,
            rng,
        )
        usage_stats = category_usage_summary(assignment)

        cursor = 0
        category_episode_outputs: list[dict[str, Any]] = []
        for record in tqdm(
            robot_records,
            desc=f"{category.category_id}: save pairings",
            unit="episode",
            leave=False,
        ):
            next_cursor = cursor + record.num_frames
            paired_human_episode_indices = assignment[cursor:next_cursor]
            if paired_human_episode_indices.shape != (record.num_frames, processing.num_pairings_per_frame):
                raise RuntimeError(f"Assignment slice mismatch for robot episode {record.episode_index}")
            metadata = save_episode_pairing(
                paths.output_root,
                record,
                category.category_id,
                paired_human_episode_indices,
                processing,
                paths.human_dataset_root,
            )
            category_episode_outputs.append(metadata)
            cursor = next_cursor
        if cursor != assignment.shape[0]:
            raise RuntimeError(f"Assignment cursor mismatch for category {category.category_id}")

        run_summary["categories"].append(
            {
                "category_id": category.category_id,
                "robot_episode_count": len(robot_records),
                "robot_frame_count": total_robot_frames,
                "total_frame_pairings": int(total_robot_frames * processing.num_pairings_per_frame),
                "human_episode_count": len(human_episode_ids),
                "human_episode_min": min(human_episode_ids),
                "human_episode_max": max(human_episode_ids),
                "num_pairings_per_frame": processing.num_pairings_per_frame,
                "distinct_within_frame": True,
                **usage_stats,
                "episodes": category_episode_outputs,
            }
        )

    (paths.output_root / "run_summary.json").write_text(
        json.dumps(run_summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
