from __future__ import annotations

import argparse
import json
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
DEFAULT_CONFIG = ROOT / "3_frame_human_segment_progress_matches_config.yaml"


@dataclass(frozen=True)
class PathsConfig:
    sim_dataset_root: Path
    human_dataset_root: Path
    robot_local_window_root: Path
    human_pairing_root: Path
    output_root: Path


@dataclass(frozen=True)
class ProcessingConfig:
    seed: int
    num_future_knots: int
    overwrite: bool
    index_dtype: str
    progress_dtype: str
    max_robot_episodes_per_category: int | None


@dataclass(frozen=True)
class CategoryConfig:
    category_id: str
    sim_start: int
    sim_end_exclusive: int
    human_start: int
    human_end_exclusive: int


@dataclass(frozen=True)
class RobotEpisodeInputs:
    episode_index: int
    category_id: str
    frame_indices: np.ndarray
    local_start_frame_index: np.ndarray
    local_end_frame_index: np.ndarray
    paired_human_episode_indices: np.ndarray


@dataclass(frozen=True)
class AnnotationIndex:
    episode_index: int
    category_id: str
    num_frames: int
    frame_to_segment_id: np.ndarray
    frame_to_progress: np.ndarray
    segment_frames: dict[int, np.ndarray]
    segment_progress: dict[int, np.ndarray]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Step 3: for each robot frame and each paired human episode, map the robot local-window start/end "
            "frames to the closest-progress human frames within the same temporal segment."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--robot-local-window-root", type=Path, default=None)
    parser.add_argument("--human-pairing-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-future-knots", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--category", action="append", default=[], help="Repeat to limit export to named categories.")
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
    processing = config.get("processing", {})
    seed = int(args.seed) if args.seed is not None else int(processing.get("seed", 0))
    num_future_knots = (
        int(args.num_future_knots)
        if args.num_future_knots is not None
        else int(processing.get("num_future_knots", 12))
    )
    if args.robot_local_window_root is not None:
        robot_local_window_root = as_path(args.robot_local_window_root)
    else:
        template = paths.get("robot_local_window_root_template") or paths.get("robot_local_window_root")
        if not template:
            raise KeyError("Config must define paths.robot_local_window_root_template or paths.robot_local_window_root.")
        robot_local_window_root = as_path(str(template).format(num_future_knots=num_future_knots))
    if args.human_pairing_root is not None:
        human_pairing_root = as_path(args.human_pairing_root)
    else:
        template = paths.get("human_pairing_root_template") or paths.get("human_pairing_root")
        if not template:
            raise KeyError("Config must define paths.human_pairing_root_template or paths.human_pairing_root.")
        human_pairing_root = as_path(str(template).format(seed=seed))
    if args.output_root is not None:
        output_root = as_path(args.output_root)
    else:
        template = paths.get("output_root_template") or paths.get("output_root")
        if not template:
            raise KeyError("Config must define paths.output_root_template or paths.output_root.")
        output_root = as_path(str(template).format(seed=seed, num_future_knots=num_future_knots))
    return PathsConfig(
        sim_dataset_root=as_path(paths["sim_dataset_root"]),
        human_dataset_root=as_path(paths["human_dataset_root"]),
        robot_local_window_root=robot_local_window_root,
        human_pairing_root=human_pairing_root,
        output_root=output_root,
    )


def load_processing(config: dict[str, Any], args: argparse.Namespace) -> ProcessingConfig:
    processing = config.get("processing")
    if not isinstance(processing, dict):
        raise KeyError("Missing processing block in config.")
    seed = int(args.seed) if args.seed is not None else int(processing.get("seed", 0))
    num_future_knots = (
        int(args.num_future_knots)
        if args.num_future_knots is not None
        else int(processing.get("num_future_knots", 12))
    )
    if num_future_knots <= 0:
        raise ValueError("num_future_knots must be positive.")
    max_robot_episodes = (
        int(args.max_robot_episodes_per_category)
        if args.max_robot_episodes_per_category is not None
        else (int(processing["max_robot_episodes_per_category"]) if processing.get("max_robot_episodes_per_category") is not None else None)
    )
    return ProcessingConfig(
        seed=seed,
        num_future_knots=num_future_knots,
        overwrite=bool(processing.get("overwrite", False) or args.overwrite),
        index_dtype=str(processing.get("index_dtype", "int32")),
        progress_dtype=str(processing.get("progress_dtype", "float32")),
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


def checkpoint_json_path(dataset_root: Path, episode_index: int) -> Path:
    return (
        dataset_root
        / "annotations"
        / "temporal_checkpoints"
        / f"chunk-{episode_index // 1000:03d}"
        / f"episode_{episode_index:06d}"
        / "checkpoints.json"
    )


def robot_local_window_npz_path(root: Path, episode_index: int) -> Path:
    return root / f"chunk-{episode_index // 1000:03d}" / f"episode_{episode_index:06d}" / "local_raw_bspline_windows.npz"


def human_pairing_npz_path(root: Path, episode_index: int) -> Path:
    return root / f"chunk-{episode_index // 1000:03d}" / f"episode_{episode_index:06d}" / "human_episode_pairings.npz"


def output_episode_dir(output_root: Path, episode_index: int) -> Path:
    return output_root / f"chunk-{episode_index // 1000:03d}" / f"episode_{episode_index:06d}"


def ensure_output_episode_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory already contains files, pass overwrite to replace it: {path}")
    path.mkdir(parents=True, exist_ok=True)


def empty_float_array(shape: tuple[int, ...], dtype_name: str) -> np.ndarray:
    array = np.empty(shape, dtype=np.dtype(dtype_name))
    array.fill(np.nan)
    return array


def load_annotation_index(checkpoint_path: Path, episode_index: int) -> AnnotationIndex:
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    category_id = canonical_category_id(payload.get("category_id", ""))
    frame_indexing = payload.get("frame_indexing") or {}
    end_frame_exclusive = int(frame_indexing.get("end_frame_exclusive", 0))
    segments = payload.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ValueError(f"No segments found in {checkpoint_path}")
    if end_frame_exclusive <= 0:
        end_frame_exclusive = max(int(item["end_frame_exclusive"]) for item in segments)

    frame_to_segment_id = np.full((end_frame_exclusive,), fill_value=-1, dtype=np.int32)
    frame_to_progress = np.empty((end_frame_exclusive,), dtype=np.float32)
    frame_to_progress.fill(np.nan)
    segment_frames: dict[int, np.ndarray] = {}
    segment_progress: dict[int, np.ndarray] = {}

    for segment in segments:
        segment_id = int(segment["segment_id"])
        progress = segment.get("progress") or {}
        per_frame = progress.get("per_frame")
        if not isinstance(per_frame, list) or not per_frame:
            raise ValueError(f"Segment {segment_id} in {checkpoint_path} has no progress.per_frame entries.")
        frames = np.asarray([int(item["frame"]) for item in per_frame], dtype=np.int32)
        progresses = np.asarray([float(item["progress"]) for item in per_frame], dtype=np.float32)
        if np.any(frames < 0) or np.any(frames >= end_frame_exclusive):
            raise ValueError(f"Segment {segment_id} in {checkpoint_path} has out-of-range frame indices.")
        frame_to_segment_id[frames] = segment_id
        frame_to_progress[frames] = progresses
        segment_frames[segment_id] = frames
        segment_progress[segment_id] = progresses

    if np.any(frame_to_segment_id < 0):
        missing_count = int(np.count_nonzero(frame_to_segment_id < 0))
        raise ValueError(f"{checkpoint_path} leaves {missing_count} frames without segment IDs.")
    if np.any(np.isnan(frame_to_progress)):
        missing_count = int(np.count_nonzero(np.isnan(frame_to_progress)))
        raise ValueError(f"{checkpoint_path} leaves {missing_count} frames without progress values.")

    return AnnotationIndex(
        episode_index=episode_index,
        category_id=category_id,
        num_frames=end_frame_exclusive,
        frame_to_segment_id=frame_to_segment_id,
        frame_to_progress=frame_to_progress,
        segment_frames=segment_frames,
        segment_progress=segment_progress,
    )


def match_frame_by_segment_progress(
    annotation: AnnotationIndex,
    segment_id: int,
    target_progress: float,
) -> tuple[int, float, float, bool]:
    frames = annotation.segment_frames.get(int(segment_id))
    progresses = annotation.segment_progress.get(int(segment_id))
    if frames is None or progresses is None or frames.size == 0:
        return -1, np.nan, np.nan, False
    deltas = np.abs(progresses.astype(np.float64) - float(target_progress))
    best_idx = int(np.argmin(deltas))
    return int(frames[best_idx]), float(progresses[best_idx]), float(deltas[best_idx]), True


def load_robot_episode_inputs(
    category_id: str,
    episode_index: int,
    robot_local_window_root: Path,
    human_pairing_root: Path,
) -> RobotEpisodeInputs:
    local_window_path = robot_local_window_npz_path(robot_local_window_root, episode_index)
    pairing_path = human_pairing_npz_path(human_pairing_root, episode_index)
    if not local_window_path.exists():
        raise FileNotFoundError(local_window_path)
    if not pairing_path.exists():
        raise FileNotFoundError(pairing_path)

    with np.load(local_window_path, allow_pickle=False) as archive:
        frame_indices = np.asarray(archive["frame_indices"], dtype=np.int64)
        local_start_frame_index = np.asarray(archive["local_start_frame_index"], dtype=np.int64)
        local_end_frame_index = np.asarray(archive["local_end_frame_index"], dtype=np.int64)
    with np.load(pairing_path, allow_pickle=False) as archive:
        pairing_frame_indices = np.asarray(archive["frame_indices"], dtype=np.int64)
        paired_human_episode_indices = np.asarray(archive["paired_human_episode_indices"], dtype=np.int64)

    if frame_indices.ndim != 1 or local_start_frame_index.ndim != 1 or local_end_frame_index.ndim != 1:
        raise ValueError(f"Unexpected local-window array ranks for robot episode {episode_index}")
    if paired_human_episode_indices.ndim != 2:
        raise ValueError(f"paired_human_episode_indices must be rank-2 for robot episode {episode_index}")
    if not np.array_equal(frame_indices, pairing_frame_indices):
        raise ValueError(f"frame_indices mismatch between step 1 and step 2 for robot episode {episode_index}")
    if not np.array_equal(frame_indices, local_start_frame_index):
        raise ValueError(f"Expected local_start_frame_index to equal frame_indices for robot episode {episode_index}")
    if local_end_frame_index.shape[0] != frame_indices.shape[0]:
        raise ValueError(f"local_end_frame_index length mismatch for robot episode {episode_index}")

    return RobotEpisodeInputs(
        episode_index=episode_index,
        category_id=category_id,
        frame_indices=frame_indices,
        local_start_frame_index=local_start_frame_index,
        local_end_frame_index=local_end_frame_index,
        paired_human_episode_indices=paired_human_episode_indices,
    )


def robot_episode_inputs_for_category(
    category: CategoryConfig,
    robot_local_window_root: Path,
    human_pairing_root: Path,
    max_robot_episodes_per_category: int | None,
) -> list[RobotEpisodeInputs]:
    episode_ids = list(range(category.sim_start, category.sim_end_exclusive))
    if max_robot_episodes_per_category is not None:
        episode_ids = episode_ids[:max_robot_episodes_per_category]
    out: list[RobotEpisodeInputs] = []
    for episode_index in tqdm(
        episode_ids,
        desc=f"{category.category_id}: robot episodes",
        unit="episode",
        leave=False,
    ):
        local_window_path = robot_local_window_npz_path(robot_local_window_root, episode_index)
        pairing_path = human_pairing_npz_path(human_pairing_root, episode_index)
        if not local_window_path.exists() or not pairing_path.exists():
            continue
        out.append(load_robot_episode_inputs(category.category_id, episode_index, robot_local_window_root, human_pairing_root))
    return out


def build_episode_matches(
    inputs: RobotEpisodeInputs,
    robot_annotation: AnnotationIndex,
    human_annotation_cache: dict[int, AnnotationIndex | None],
    human_dataset_root: Path,
    processing: ProcessingConfig,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    num_frames = int(inputs.frame_indices.shape[0])
    num_pairings_per_frame = int(inputs.paired_human_episode_indices.shape[1])
    index_dtype = np.dtype(processing.index_dtype)
    progress_dtype = np.dtype(processing.progress_dtype)

    robot_start_frame_index = np.asarray(inputs.local_start_frame_index, dtype=index_dtype)
    robot_end_frame_index = np.asarray(inputs.local_end_frame_index, dtype=index_dtype)
    if np.any(inputs.local_start_frame_index < 0) or np.any(inputs.local_end_frame_index < 0):
        raise ValueError(f"Negative frame indices encountered in robot episode {inputs.episode_index}")
    if np.any(inputs.local_end_frame_index >= robot_annotation.num_frames):
        raise ValueError(f"Robot end frame exceeds annotation coverage in robot episode {inputs.episode_index}")

    robot_start_segment_id = robot_annotation.frame_to_segment_id[inputs.local_start_frame_index].astype(index_dtype, copy=False)
    robot_end_segment_id = robot_annotation.frame_to_segment_id[inputs.local_end_frame_index].astype(index_dtype, copy=False)
    robot_start_progress = robot_annotation.frame_to_progress[inputs.local_start_frame_index].astype(progress_dtype, copy=False)
    robot_end_progress = robot_annotation.frame_to_progress[inputs.local_end_frame_index].astype(progress_dtype, copy=False)

    human_start_frame_index = np.full((num_frames, num_pairings_per_frame), fill_value=-1, dtype=index_dtype)
    human_end_frame_index = np.full((num_frames, num_pairings_per_frame), fill_value=-1, dtype=index_dtype)
    human_matched_start_progress = empty_float_array((num_frames, num_pairings_per_frame), processing.progress_dtype)
    human_matched_end_progress = empty_float_array((num_frames, num_pairings_per_frame), processing.progress_dtype)
    human_start_progress_abs_error = empty_float_array((num_frames, num_pairings_per_frame), processing.progress_dtype)
    human_end_progress_abs_error = empty_float_array((num_frames, num_pairings_per_frame), processing.progress_dtype)
    start_match_valid_mask = np.zeros((num_frames, num_pairings_per_frame), dtype=bool)
    end_match_valid_mask = np.zeros((num_frames, num_pairings_per_frame), dtype=bool)
    missing_human_checkpoint_count = 0

    for frame_pos in range(num_frames):
        start_segment_id = int(robot_start_segment_id[frame_pos])
        end_segment_id = int(robot_end_segment_id[frame_pos])
        start_progress = float(robot_start_progress[frame_pos])
        end_progress = float(robot_end_progress[frame_pos])
        for pairing_slot in range(num_pairings_per_frame):
            human_episode_index = int(inputs.paired_human_episode_indices[frame_pos, pairing_slot])
            if human_episode_index not in human_annotation_cache:
                checkpoint_path = checkpoint_json_path(human_dataset_root, human_episode_index)
                if checkpoint_path.exists():
                    human_annotation_cache[human_episode_index] = load_annotation_index(checkpoint_path, human_episode_index)
                else:
                    human_annotation_cache[human_episode_index] = None
            human_annotation = human_annotation_cache[human_episode_index]
            if human_annotation is None:
                missing_human_checkpoint_count += 1
                continue

            start_frame, start_progress_matched, start_error, start_valid = match_frame_by_segment_progress(
                human_annotation,
                start_segment_id,
                start_progress,
            )
            end_frame, end_progress_matched, end_error, end_valid = match_frame_by_segment_progress(
                human_annotation,
                end_segment_id,
                end_progress,
            )
            if start_valid:
                human_start_frame_index[frame_pos, pairing_slot] = np.asarray(start_frame, dtype=index_dtype)
                human_matched_start_progress[frame_pos, pairing_slot] = np.asarray(start_progress_matched, dtype=progress_dtype)
                human_start_progress_abs_error[frame_pos, pairing_slot] = np.asarray(start_error, dtype=progress_dtype)
                start_match_valid_mask[frame_pos, pairing_slot] = True
            if end_valid:
                human_end_frame_index[frame_pos, pairing_slot] = np.asarray(end_frame, dtype=index_dtype)
                human_matched_end_progress[frame_pos, pairing_slot] = np.asarray(end_progress_matched, dtype=progress_dtype)
                human_end_progress_abs_error[frame_pos, pairing_slot] = np.asarray(end_error, dtype=progress_dtype)
                end_match_valid_mask[frame_pos, pairing_slot] = True

    arrays = {
        "frame_indices": np.asarray(inputs.frame_indices, dtype=index_dtype),
        "paired_human_episode_indices": np.asarray(inputs.paired_human_episode_indices, dtype=index_dtype),
        "robot_start_frame_index": robot_start_frame_index,
        "robot_end_frame_index": robot_end_frame_index,
        "robot_start_segment_id": robot_start_segment_id,
        "robot_end_segment_id": robot_end_segment_id,
        "robot_start_progress": robot_start_progress,
        "robot_end_progress": robot_end_progress,
        "human_start_frame_index": human_start_frame_index,
        "human_end_frame_index": human_end_frame_index,
        "human_matched_start_progress": human_matched_start_progress,
        "human_matched_end_progress": human_matched_end_progress,
        "human_start_progress_abs_error": human_start_progress_abs_error,
        "human_end_progress_abs_error": human_end_progress_abs_error,
        "start_match_valid_mask": start_match_valid_mask,
        "end_match_valid_mask": end_match_valid_mask,
    }

    start_valid_count = int(np.count_nonzero(start_match_valid_mask))
    end_valid_count = int(np.count_nonzero(end_match_valid_mask))
    total_pairings = int(num_frames * num_pairings_per_frame)
    combined_valid = start_match_valid_mask & end_match_valid_mask
    combined_valid_count = int(np.count_nonzero(combined_valid))

    def _summary_stats(values: np.ndarray, mask: np.ndarray) -> dict[str, float | int | None]:
        valid = values[mask]
        if valid.size == 0:
            return {"count": 0, "mean": None, "median": None, "max": None}
        valid = valid.astype(np.float64, copy=False)
        return {
            "count": int(valid.size),
            "mean": float(valid.mean()),
            "median": float(np.median(valid)),
            "max": float(valid.max()),
        }

    metadata = {
        "robot_episode_index": inputs.episode_index,
        "robot_category_id": inputs.category_id,
        "num_frames": num_frames,
        "num_pairings_per_frame": num_pairings_per_frame,
        "total_frame_pairings": total_pairings,
        "start_match_valid_count": start_valid_count,
        "end_match_valid_count": end_valid_count,
        "complete_match_valid_count": combined_valid_count,
        "missing_human_checkpoint_count": missing_human_checkpoint_count,
        "start_progress_abs_error": _summary_stats(human_start_progress_abs_error, start_match_valid_mask),
        "end_progress_abs_error": _summary_stats(human_end_progress_abs_error, end_match_valid_mask),
        "human_dataset_root": str(human_dataset_root),
        "stored_fields": sorted(arrays),
    }
    return arrays, metadata


def save_episode_output(
    output_root: Path,
    episode_index: int,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
    overwrite: bool,
) -> None:
    out_dir = output_episode_dir(output_root, episode_index)
    ensure_output_episode_dir(out_dir, overwrite=overwrite)
    np.savez_compressed(out_dir / "frame_human_segment_progress_matches.npz", **arrays)
    output_metadata = dict(metadata)
    output_metadata["output_npz_path"] = str((out_dir / "frame_human_segment_progress_matches.npz").resolve())
    (out_dir / "frame_human_segment_progress_matches_metadata.json").write_text(
        json.dumps(output_metadata, indent=2) + "\n",
        encoding="utf-8",
    )


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
    if not paths.robot_local_window_root.exists():
        raise FileNotFoundError(paths.robot_local_window_root)
    if not paths.human_pairing_root.exists():
        raise FileNotFoundError(paths.human_pairing_root)
    paths.output_root.mkdir(parents=True, exist_ok=True)

    run_summary: dict[str, Any] = {
        "sim_dataset_root": str(paths.sim_dataset_root),
        "human_dataset_root": str(paths.human_dataset_root),
        "robot_local_window_root": str(paths.robot_local_window_root),
        "human_pairing_root": str(paths.human_pairing_root),
        "output_root": str(paths.output_root),
        "seed": processing.seed,
        "num_future_knots": processing.num_future_knots,
        "categories": [],
    }

    human_annotation_cache: dict[int, AnnotationIndex | None] = {}

    print("Exporting frame-level robot/human segment-progress matches")
    print(f"  robot_local_window_root : {paths.robot_local_window_root}")
    print(f"  human_pairing_root      : {paths.human_pairing_root}")
    print(f"  sim_dataset_root        : {paths.sim_dataset_root}")
    print(f"  human_dataset_root      : {paths.human_dataset_root}")
    print(f"  output_root             : {paths.output_root}")
    print(f"  seed                    : {processing.seed}")
    print(f"  num_future_knots        : {processing.num_future_knots}")
    print(f"  categories              : {', '.join(item.category_id for item in categories)}")

    for category in tqdm(categories, desc="categories", unit="category"):
        robot_inputs = robot_episode_inputs_for_category(
            category,
            paths.robot_local_window_root,
            paths.human_pairing_root,
            processing.max_robot_episodes_per_category,
        )
        if not robot_inputs:
            raise RuntimeError(f"No step-1/step-2 episodes available for category {category.category_id}")

        category_episode_outputs: list[dict[str, Any]] = []
        total_frames = 0
        total_pairings = 0
        total_start_valid = 0
        total_end_valid = 0
        total_complete_valid = 0
        total_missing_human_checkpoints = 0

        for inputs in tqdm(
            robot_inputs,
            desc=f"{category.category_id}: save matches",
            unit="episode",
            leave=False,
        ):
            robot_checkpoint = checkpoint_json_path(paths.sim_dataset_root, inputs.episode_index)
            robot_annotation = load_annotation_index(robot_checkpoint, inputs.episode_index)
            if robot_annotation.category_id and robot_annotation.category_id != category.category_id:
                raise ValueError(
                    f"Robot episode {inputs.episode_index} annotation category {robot_annotation.category_id!r} "
                    f"does not match requested category {category.category_id!r}"
                )

            arrays, metadata = build_episode_matches(
                inputs,
                robot_annotation,
                human_annotation_cache,
                paths.human_dataset_root,
                processing,
            )
            save_episode_output(paths.output_root, inputs.episode_index, arrays, metadata, overwrite=processing.overwrite)
            category_episode_outputs.append(metadata)
            total_frames += int(metadata["num_frames"])
            total_pairings += int(metadata["total_frame_pairings"])
            total_start_valid += int(metadata["start_match_valid_count"])
            total_end_valid += int(metadata["end_match_valid_count"])
            total_complete_valid += int(metadata["complete_match_valid_count"])
            total_missing_human_checkpoints += int(metadata["missing_human_checkpoint_count"])

        run_summary["categories"].append(
            {
                "category_id": category.category_id,
                "robot_episode_count": len(robot_inputs),
                "robot_frame_count": total_frames,
                "total_frame_pairings": total_pairings,
                "start_match_valid_count": total_start_valid,
                "end_match_valid_count": total_end_valid,
                "complete_match_valid_count": total_complete_valid,
                "missing_human_checkpoint_count": total_missing_human_checkpoints,
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
