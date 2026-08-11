from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, Sampler
from tqdm.auto import tqdm

from .utils import EpisodeLRUCache, RunningMoments, atomic_json_dump, atomic_npz_dump


STATE_COLUMN = "observation.state"


@dataclass(frozen=True)
class CategoryConfig:
    category_id: str
    sim_start: int
    sim_end_exclusive: int
    human_start: int
    human_end_exclusive: int


@dataclass(frozen=True)
class AnnotationEpisode:
    episode_index: int
    category_id: str
    labels: list[str]
    frame_to_segment_id: np.ndarray
    frame_to_progress: np.ndarray
    segment_frames: dict[int, np.ndarray]
    segment_progress: dict[int, np.ndarray]
    segment_frames_sorted_by_progress: dict[int, np.ndarray]
    segment_progress_sorted: dict[int, np.ndarray]


@dataclass(frozen=True)
class ExactHumanIntervalCache:
    frame_indices: np.ndarray
    paired_human_episode_indices: np.ndarray
    human_gt_start_u: np.ndarray
    human_gt_end_u: np.ndarray
    interval_valid_mask: np.ndarray
    robot_start_segment_id: np.ndarray
    robot_end_segment_id: np.ndarray
    robot_start_progress: np.ndarray
    robot_end_progress: np.ndarray


@dataclass(frozen=True)
class HumanSplineEpisode:
    episode_index: int
    coefficients: np.ndarray
    knots: np.ndarray
    degree: int


@dataclass(frozen=True)
class RobotEpisodeBundle:
    episode_index: int
    frame_indices: np.ndarray
    embeddings: np.ndarray
    states: np.ndarray
    global_coefficients: np.ndarray
    global_knots: np.ndarray
    degree: int
    local_start_frame_index: np.ndarray
    local_end_frame_index: np.ndarray
    global_local_start_u: np.ndarray
    global_local_end_u: np.ndarray
    exact_local_spline_valid: np.ndarray
    exact_local_knot_local_u_flat: np.ndarray
    exact_local_knot_offsets: np.ndarray
    exact_local_num_knots: np.ndarray
    paired_human_episode_indices: np.ndarray
    predicted_human_u: np.ndarray | None
    human_gt_start_u: np.ndarray | None = None
    human_gt_end_u: np.ndarray | None = None
    interval_valid_mask: np.ndarray | None = None
    category_index: int | None = None


def canonical_category_id(value: str) -> str:
    return str(value).strip().lower()


def as_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def load_category_configs(config: dict[str, Any]) -> list[CategoryConfig]:
    categories = []
    for item in config["categories"]:
        categories.append(
            CategoryConfig(
                category_id=canonical_category_id(item["id"]),
                sim_start=int(item["sim_start"]),
                sim_end_exclusive=int(item["sim_end_exclusive"]),
                human_start=int(item["human_start"]),
                human_end_exclusive=int(item["human_end_exclusive"]),
            )
        )
    selected = {canonical_category_id(value) for value in config["data"]["selected_categories"]}
    return [item for item in categories if item.category_id in selected]


def category_for_robot_episode(episode_index: int, categories: list[CategoryConfig]) -> CategoryConfig:
    for category in categories:
        if category.sim_start <= episode_index < category.sim_end_exclusive:
            return category
    raise KeyError(f"No selected category covers robot episode {episode_index}")


def robot_embedding_npz_path(root: Path, episode_index: int) -> Path:
    return root / f"chunk-{episode_index // 1000:03d}" / f"episode_{episode_index:06d}" / "frame_embeddings.npz"


def robot_spline_npz_path(root: Path, episode_index: int) -> Path:
    return root / f"chunk-{episode_index // 1000:03d}" / f"episode_{episode_index:06d}" / "spline.npz"


def robot_local_window_npz_path(root: Path, episode_index: int) -> Path:
    return root / f"chunk-{episode_index // 1000:03d}" / f"episode_{episode_index:06d}" / "local_raw_bspline_windows.npz"


def pairing_npz_path(root: Path, episode_index: int) -> Path:
    return root / f"chunk-{episode_index // 1000:03d}" / f"episode_{episode_index:06d}" / "human_episode_pairings.npz"


def predicted_human_u_npz_path(root: Path, episode_index: int) -> Path:
    return root / f"chunk-{episode_index // 1000:03d}" / f"episode_{episode_index:06d}" / "predicted_human_u.npz"


def human_spline_npz_path(root: Path, episode_index: int) -> Path:
    return root / f"chunk-{episode_index // 1000:03d}" / f"episode_{episode_index:06d}" / "spline.npz"


def consolidated_human_spline_npz_path(cache_root: Path, episode_index: int) -> Path:
    return cache_root / f"chunk-{episode_index // 1000:03d}" / f"episode_{episode_index:06d}" / "human_spline_cache.npz"


def checkpoint_json_path(dataset_root: Path, episode_index: int) -> Path:
    return (
        dataset_root
        / "annotations"
        / "temporal_checkpoints"
        / f"chunk-{episode_index // 1000:03d}"
        / f"episode_{episode_index:06d}"
        / "checkpoints.json"
    )


def robot_parquet_path(dataset_root: Path, episode_index: int) -> Path:
    return dataset_root / "data" / f"chunk-{episode_index // 1000:03d}" / f"episode_{episode_index:06d}.parquet"


def exact_human_interval_npz_path(cache_root: Path, episode_index: int) -> Path:
    return cache_root / f"chunk-{episode_index // 1000:03d}" / f"episode_{episode_index:06d}" / "exact_human_interval_cache.npz"


def consolidated_robot_episode_npz_path(cache_root: Path, episode_index: int) -> Path:
    return cache_root / f"chunk-{episode_index // 1000:03d}" / f"episode_{episode_index:06d}" / "robot_episode_cache.npz"


def read_dataset_info(root: Path) -> dict[str, Any]:
    return json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))


def resolve_state_dims(config: dict[str, Any], info: dict[str, Any]) -> list[int]:
    state_dims = config["data"].get("state_dims", "all")
    if state_dims == "all":
        shape = info["features"][STATE_COLUMN]["shape"]
        return list(range(int(shape[0])))
    return [int(value) for value in state_dims]


def load_annotation_episode(path: Path, episode_index: int) -> AnnotationEpisode:
    payload = json.loads(path.read_text(encoding="utf-8"))
    category_id = canonical_category_id(payload.get("category_id", ""))
    labels = [str(item) for item in payload.get("labels", [])]
    segments = payload.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ValueError(f"No segments found in {path}")
    end_frame_exclusive = int((payload.get("frame_indexing") or {}).get("end_frame_exclusive", 0))
    if end_frame_exclusive <= 0:
        end_frame_exclusive = max(int(item["end_frame_exclusive"]) for item in segments)
    frame_to_segment_id = np.full((end_frame_exclusive,), fill_value=-1, dtype=np.int32)
    frame_to_progress = np.full((end_frame_exclusive,), fill_value=np.nan, dtype=np.float32)
    segment_frames: dict[int, np.ndarray] = {}
    segment_progress: dict[int, np.ndarray] = {}
    segment_frames_sorted: dict[int, np.ndarray] = {}
    segment_progress_sorted: dict[int, np.ndarray] = {}
    for segment in segments:
        segment_id = int(segment["segment_id"])
        per_frame = ((segment.get("progress") or {}).get("per_frame")) or []
        if not per_frame:
            raise ValueError(f"Segment {segment_id} in {path} does not contain progress.per_frame")
        frames = np.asarray([int(item["frame"]) for item in per_frame], dtype=np.int32)
        progress = np.asarray([float(item["progress"]) for item in per_frame], dtype=np.float32)
        frame_to_segment_id[frames] = segment_id
        frame_to_progress[frames] = progress
        order = np.argsort(progress, kind="stable")
        segment_frames[segment_id] = frames
        segment_progress[segment_id] = progress
        segment_frames_sorted[segment_id] = frames[order]
        segment_progress_sorted[segment_id] = progress[order]
    if np.any(frame_to_segment_id < 0):
        raise ValueError(f"Annotation {path} leaves unlabeled frames.")
    return AnnotationEpisode(
        episode_index=episode_index,
        category_id=category_id,
        labels=labels,
        frame_to_segment_id=frame_to_segment_id,
        frame_to_progress=frame_to_progress,
        segment_frames=segment_frames,
        segment_progress=segment_progress,
        segment_frames_sorted_by_progress=segment_frames_sorted,
        segment_progress_sorted=segment_progress_sorted,
    )


def load_frame_u_by_frame_index(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        frame_indices = np.asarray(archive["frame_indices"], dtype=np.int64)
        frame_u = np.asarray(archive["frame_u"], dtype=np.float64)
    if frame_indices.shape[0] != frame_u.shape[0]:
        raise ValueError(f"frame_indices/frame_u mismatch in {path}")
    size = int(frame_indices.max()) + 1
    mapping = np.full((size,), fill_value=np.nan, dtype=np.float64)
    mapping[frame_indices] = frame_u
    return mapping.astype(np.float32)


def interpolate_human_u(
    human_annotation: AnnotationEpisode,
    human_frame_u_by_frame_index: np.ndarray,
    segment_id: int,
    target_progress: float,
    tolerance: float = 1e-6,
) -> tuple[float, bool]:
    if segment_id not in human_annotation.segment_progress:
        return float("nan"), False
    sorted_progress = human_annotation.segment_progress_sorted[segment_id]
    sorted_frames = human_annotation.segment_frames_sorted_by_progress[segment_id]
    raw_progress = human_annotation.segment_progress[segment_id]
    raw_frames = human_annotation.segment_frames[segment_id]
    if sorted_progress.size == 0:
        return float("nan"), False
    insert = int(np.searchsorted(sorted_progress, target_progress, side="left"))
    if insert <= 0:
        frame_index = int(sorted_frames[0])
        u_value = float(human_frame_u_by_frame_index[frame_index])
        return (u_value, np.isfinite(u_value))
    if insert >= sorted_progress.size:
        frame_index = int(sorted_frames[-1])
        u_value = float(human_frame_u_by_frame_index[frame_index])
        return (u_value, np.isfinite(u_value))
    q_a = float(sorted_progress[insert - 1])
    q_b = float(sorted_progress[insert])
    frame_a = int(sorted_frames[insert - 1])
    frame_b = int(sorted_frames[insert])
    u_a = float(human_frame_u_by_frame_index[frame_a])
    u_b = float(human_frame_u_by_frame_index[frame_b])
    if not np.isfinite(u_a) or not np.isfinite(u_b):
        return float("nan"), False
    if abs(q_b - q_a) <= tolerance:
        nearest_index = int(np.argmin(np.abs(raw_progress.astype(np.float64) - float(target_progress))))
        frame_nearest = int(raw_frames[nearest_index])
        u_value = float(human_frame_u_by_frame_index[frame_nearest])
        return (u_value, np.isfinite(u_value))
    alpha = (float(target_progress) - q_a) / (q_b - q_a)
    return (float((1.0 - alpha) * u_a + alpha * u_b), True)


def discover_valid_robot_episodes(
    sim_dataset_root: Path,
    robot_embedding_root: Path,
    robot_spline_root: Path,
    robot_local_window_root: Path,
    robot_pairing_root: Path,
    categories: list[CategoryConfig],
    skip_missing: bool,
) -> dict[str, list[int]]:
    result: dict[str, list[int]] = {}
    for category in categories:
        episode_ids: list[int] = []
        for episode_index in range(category.sim_start, category.sim_end_exclusive):
            required = (
                robot_embedding_npz_path(robot_embedding_root, episode_index),
                robot_spline_npz_path(robot_spline_root, episode_index),
                robot_local_window_npz_path(robot_local_window_root, episode_index),
                pairing_npz_path(robot_pairing_root, episode_index),
                checkpoint_json_path(sim_dataset_root, episode_index),
            )
            if all(path.exists() for path in required):
                episode_ids.append(episode_index)
                continue
            if not skip_missing:
                missing = [str(path) for path in required if not path.exists()]
                raise FileNotFoundError(f"Robot episode {episode_index} is missing required files: {missing}")
        result[category.category_id] = episode_ids
    return result


def split_robot_episodes_by_category(
    category_to_episode_ids: dict[str, list[int]],
    split_seed: int,
    per_category_ratio: float,
    min_episodes_per_category: int,
    max_episodes_per_category: int | None,
) -> dict[str, dict[str, list[int]]]:
    rng = np.random.default_rng(int(split_seed))
    result: dict[str, dict[str, list[int]]] = {}
    for category_id, episode_ids in category_to_episode_ids.items():
        order = rng.permutation(len(episode_ids)).tolist()
        ordered = [episode_ids[index] for index in order]
        if max_episodes_per_category is not None:
            ordered = ordered[: int(max_episodes_per_category)]
        if len(ordered) <= 1:
            result[category_id] = {"train": ordered, "val": []}
            continue
        val_count = int(round(len(ordered) * float(per_category_ratio)))
        val_count = max(int(min_episodes_per_category), val_count)
        val_count = min(val_count, len(ordered) - 1)
        result[category_id] = {"train": ordered[val_count:], "val": ordered[:val_count]}
    return result


def collect_human_episode_ids_from_pairings(robot_episode_ids: list[int], pairing_root: Path) -> list[int]:
    human_episode_ids: set[int] = set()
    for episode_index in tqdm(robot_episode_ids, desc="pairings/human-ids", unit="episode"):
        with np.load(pairing_npz_path(pairing_root, episode_index), allow_pickle=False) as archive:
            human_episode_ids.update(int(value) for value in np.asarray(archive["paired_human_episode_indices"]).reshape(-1).tolist())
    return sorted(human_episode_ids)


def compute_state_normalization(
    sim_dataset_root: Path,
    robot_episode_ids: list[int],
    state_dims: list[int],
    output_path: Path,
) -> dict[str, Any]:
    moments = RunningMoments(len(state_dims))
    for episode_index in tqdm(robot_episode_ids, desc="precompute/state-norm", unit="episode"):
        table = pq.read_table(robot_parquet_path(sim_dataset_root, episode_index), columns=[STATE_COLUMN])
        states = np.asarray(table[STATE_COLUMN].to_pylist(), dtype=np.float32)[:, state_dims]
        moments.update(states)
    mean, std = moments.result()
    payload = {
        "state_dims": [int(value) for value in state_dims],
        "state_mean": mean.tolist(),
        "state_std": std.tolist(),
        "episode_count": len(robot_episode_ids),
    }
    atomic_json_dump(payload, output_path)
    return payload


def precompute_exact_human_interval_cache(
    robot_episode_ids: list[int],
    sim_dataset_root: Path,
    human_dataset_root: Path,
    robot_pairing_root: Path,
    robot_local_window_root: Path,
    human_bspline_root: Path,
    cache_root: Path,
    overwrite: bool,
) -> dict[str, Any]:
    cache_root.mkdir(parents=True, exist_ok=True)
    human_annotation_cache: dict[int, AnnotationEpisode] = {}
    human_frame_u_cache: dict[int, np.ndarray] = {}
    total_samples = 0
    valid_samples = 0
    invalid_samples = 0
    for episode_index in tqdm(robot_episode_ids, desc="precompute/exact-human-u", unit="episode"):
        out_path = exact_human_interval_npz_path(cache_root, episode_index)
        if out_path.exists() and not overwrite:
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with np.load(pairing_npz_path(robot_pairing_root, episode_index), allow_pickle=False) as pair_archive:
            frame_indices = np.asarray(pair_archive["frame_indices"], dtype=np.int64)
            paired_human_episode_indices = np.asarray(pair_archive["paired_human_episode_indices"], dtype=np.int64)
        with np.load(robot_local_window_npz_path(robot_local_window_root, episode_index), allow_pickle=False) as window_archive:
            local_start_frame_index = np.asarray(window_archive["local_start_frame_index"], dtype=np.int64)
            local_end_frame_index = np.asarray(window_archive["local_end_frame_index"], dtype=np.int64)
        robot_annotation = load_annotation_episode(checkpoint_json_path(sim_dataset_root, episode_index), episode_index)
        robot_start_segment_id = robot_annotation.frame_to_segment_id[local_start_frame_index].astype(np.int32, copy=False)
        robot_end_segment_id = robot_annotation.frame_to_segment_id[local_end_frame_index].astype(np.int32, copy=False)
        robot_start_progress = robot_annotation.frame_to_progress[local_start_frame_index].astype(np.float32, copy=False)
        robot_end_progress = robot_annotation.frame_to_progress[local_end_frame_index].astype(np.float32, copy=False)
        human_gt_start_u = np.full_like(paired_human_episode_indices, fill_value=np.nan, dtype=np.float32)
        human_gt_end_u = np.full_like(paired_human_episode_indices, fill_value=np.nan, dtype=np.float32)
        interval_valid_mask = np.zeros_like(paired_human_episode_indices, dtype=bool)

        for row in range(frame_indices.shape[0]):
            start_segment = int(robot_start_segment_id[row])
            end_segment = int(robot_end_segment_id[row])
            start_progress = float(robot_start_progress[row])
            end_progress = float(robot_end_progress[row])
            for slot in range(paired_human_episode_indices.shape[1]):
                human_episode_index = int(paired_human_episode_indices[row, slot])
                if human_episode_index not in human_annotation_cache:
                    human_annotation_cache[human_episode_index] = load_annotation_episode(
                        checkpoint_json_path(human_dataset_root, human_episode_index),
                        human_episode_index,
                    )
                if human_episode_index not in human_frame_u_cache:
                    human_frame_u_cache[human_episode_index] = load_frame_u_by_frame_index(
                        human_spline_npz_path(human_bspline_root, human_episode_index)
                    )
                start_u, start_valid = interpolate_human_u(
                    human_annotation_cache[human_episode_index],
                    human_frame_u_cache[human_episode_index],
                    start_segment,
                    start_progress,
                )
                end_u, end_valid = interpolate_human_u(
                    human_annotation_cache[human_episode_index],
                    human_frame_u_cache[human_episode_index],
                    end_segment,
                    end_progress,
                )
                if start_valid and end_valid and end_u > start_u:
                    human_gt_start_u[row, slot] = np.asarray(start_u, dtype=np.float32)
                    human_gt_end_u[row, slot] = np.asarray(end_u, dtype=np.float32)
                    interval_valid_mask[row, slot] = True
                    valid_samples += 1
                else:
                    invalid_samples += 1
                total_samples += 1
        atomic_npz_dump(
            out_path,
            compressed=True,
            frame_indices=frame_indices.astype(np.int32, copy=False),
            paired_human_episode_indices=paired_human_episode_indices.astype(np.int32, copy=False),
            human_gt_start_u=human_gt_start_u.astype(np.float32, copy=False),
            human_gt_end_u=human_gt_end_u.astype(np.float32, copy=False),
            interval_valid_mask=interval_valid_mask,
            robot_start_segment_id=robot_start_segment_id.astype(np.int32, copy=False),
            robot_end_segment_id=robot_end_segment_id.astype(np.int32, copy=False),
            robot_start_progress=robot_start_progress.astype(np.float32, copy=False),
            robot_end_progress=robot_end_progress.astype(np.float32, copy=False),
        )
    return {
        "total_samples": int(total_samples),
        "valid_samples": int(valid_samples),
        "invalid_samples": int(invalid_samples),
        "valid_fraction": float(valid_samples / max(1, total_samples)),
    }


def load_exact_human_interval_cache(path: Path) -> ExactHumanIntervalCache:
    with np.load(path, allow_pickle=False) as archive:
        return ExactHumanIntervalCache(
            frame_indices=np.asarray(archive["frame_indices"], dtype=np.int32),
            paired_human_episode_indices=np.asarray(archive["paired_human_episode_indices"], dtype=np.int32),
            human_gt_start_u=np.asarray(archive["human_gt_start_u"], dtype=np.float32),
            human_gt_end_u=np.asarray(archive["human_gt_end_u"], dtype=np.float32),
            interval_valid_mask=np.asarray(archive["interval_valid_mask"], dtype=bool),
            robot_start_segment_id=np.asarray(archive["robot_start_segment_id"], dtype=np.int32),
            robot_end_segment_id=np.asarray(archive["robot_end_segment_id"], dtype=np.int32),
            robot_start_progress=np.asarray(archive["robot_start_progress"], dtype=np.float32),
            robot_end_progress=np.asarray(archive["robot_end_progress"], dtype=np.float32),
        )


def prepare_consolidated_human_spline_cache(
    human_episode_ids: list[int],
    human_bspline_root: Path,
    consolidated_cache_root: Path,
    *,
    overwrite: bool,
    compressed: bool,
) -> dict[str, Any]:
    consolidated_cache_root.mkdir(parents=True, exist_ok=True)
    total_episodes = 0
    total_coefficients = 0
    total_knots = 0
    for episode_index in tqdm(human_episode_ids, desc="precompute/human-spline-cache", unit="episode"):
        out_path = consolidated_human_spline_npz_path(consolidated_cache_root, episode_index)
        if out_path.exists() and not overwrite:
            continue
        with np.load(human_spline_npz_path(human_bspline_root, episode_index), allow_pickle=False) as archive:
            coefficients = np.asarray(archive["global_coefficients"], dtype=np.float32)
            knots = np.asarray(archive["global_knots"], dtype=np.float32)
            degree = int(np.asarray(archive["global_degree"]).reshape(-1)[0])
        atomic_npz_dump(
            out_path,
            compressed=compressed,
            global_coefficients=coefficients.astype(np.float32, copy=False),
            global_knots=knots.astype(np.float32, copy=False),
            global_degree=np.asarray(degree, dtype=np.int32),
        )
        total_episodes += 1
        total_coefficients += int(coefficients.shape[0])
        total_knots += int(knots.shape[0])
    return {
        "episodes_written": int(total_episodes),
        "coefficients_written": int(total_coefficients),
        "knots_written": int(total_knots),
        "compressed": bool(compressed),
    }


def prepare_consolidated_robot_episode_cache(
    robot_episode_ids: list[int],
    sim_dataset_root: Path,
    robot_embedding_root: Path,
    robot_spline_root: Path,
    robot_local_window_root: Path,
    robot_pairing_root: Path,
    exact_human_interval_cache_root: Path,
    consolidated_cache_root: Path,
    state_dims: list[int],
    state_mean: np.ndarray,
    state_std: np.ndarray,
    robot_embedding_key: str,
    robot_state_source: str,
    predicted_human_u_root: Path | None,
    categories: list[CategoryConfig],
    category_to_index: dict[str, int],
    verify_alignment: bool,
    overwrite: bool,
    compressed: bool,
) -> dict[str, Any]:
    consolidated_cache_root.mkdir(parents=True, exist_ok=True)
    total_episodes = 0
    total_frames = 0
    total_valid_samples = 0
    total_invalid_samples = 0
    state_dims_array = np.asarray(state_dims, dtype=np.int64)
    state_mean = np.asarray(state_mean, dtype=np.float32)
    state_std = np.asarray(state_std, dtype=np.float32)

    for episode_index in tqdm(robot_episode_ids, desc="precompute/robot-episode-cache", unit="episode"):
        out_path = consolidated_robot_episode_npz_path(consolidated_cache_root, episode_index)
        if out_path.exists() and not overwrite:
            continue

        with np.load(robot_embedding_npz_path(robot_embedding_root, episode_index), allow_pickle=False) as embed_archive:
            embeddings = np.asarray(embed_archive[robot_embedding_key], dtype=np.float32)
            frame_indices = np.asarray(embed_archive["frame_indices"], dtype=np.int64)
            state_from_embedding = np.asarray(embed_archive["state"], dtype=np.float32) if "state" in embed_archive.files else None

        if robot_state_source == "embedding_file":
            if state_from_embedding is None:
                raise KeyError(f"Robot embedding file does not contain state for episode {episode_index}")
            states = np.asarray(state_from_embedding[:, state_dims_array], dtype=np.float32)
        else:
            table = pq.read_table(robot_parquet_path(sim_dataset_root, episode_index), columns=[STATE_COLUMN, "frame_index"])
            states = np.asarray(table[STATE_COLUMN].to_pylist(), dtype=np.float32)[:, state_dims_array]
            parquet_frame_indices = np.asarray(table["frame_index"].to_numpy(), dtype=np.int64)
            if verify_alignment and not np.array_equal(frame_indices, parquet_frame_indices):
                raise ValueError(f"Frame-index mismatch between embeddings and parquet for robot episode {episode_index}")
        normalized_states = (states - state_mean) / state_std

        with np.load(robot_spline_npz_path(robot_spline_root, episode_index), allow_pickle=False) as spline_archive:
            global_coefficients = np.asarray(spline_archive["global_coefficients"], dtype=np.float32)
            global_knots = np.asarray(spline_archive["global_knots"], dtype=np.float32)
            degree = int(np.asarray(spline_archive["global_degree"]).reshape(-1)[0])
            spline_frame_indices = np.asarray(spline_archive["frame_indices"], dtype=np.int64)
        if verify_alignment and not np.array_equal(frame_indices, spline_frame_indices):
            raise ValueError(f"Frame-index mismatch between embeddings and fitted spline for robot episode {episode_index}")

        with np.load(robot_local_window_npz_path(robot_local_window_root, episode_index), allow_pickle=False) as local_archive:
            local_frame_indices = np.asarray(local_archive["frame_indices"], dtype=np.int64)
            local_start_frame_index = np.asarray(local_archive["local_start_frame_index"], dtype=np.int32)
            local_end_frame_index = np.asarray(local_archive["local_end_frame_index"], dtype=np.int32)
            global_local_start_u = np.asarray(local_archive["global_local_start_u"], dtype=np.float32)
            global_local_end_u = np.asarray(local_archive["global_local_end_u"], dtype=np.float32)
            exact_local_spline_valid = np.asarray(local_archive["exact_local_spline_valid"], dtype=bool)
            exact_local_knot_local_u_flat = np.asarray(local_archive["exact_local_spline_knot_local_u"], dtype=np.float32)
            exact_local_knot_offsets = np.asarray(local_archive["exact_local_spline_knot_offsets"], dtype=np.int32)
            exact_local_num_knots = np.asarray(local_archive["exact_local_spline_num_knots"], dtype=np.int32)
        if verify_alignment and not np.array_equal(frame_indices, local_frame_indices):
            raise ValueError(f"Frame-index mismatch between embeddings and local windows for robot episode {episode_index}")

        with np.load(pairing_npz_path(robot_pairing_root, episode_index), allow_pickle=False) as pair_archive:
            pairing_frame_indices = np.asarray(pair_archive["frame_indices"], dtype=np.int64)
            paired_human_episode_indices = np.asarray(pair_archive["paired_human_episode_indices"], dtype=np.int32)
        if verify_alignment and not np.array_equal(frame_indices, pairing_frame_indices):
            raise ValueError(f"Frame-index mismatch between embeddings and pairings for robot episode {episode_index}")

        interval_cache = load_exact_human_interval_cache(exact_human_interval_npz_path(exact_human_interval_cache_root, episode_index))
        if verify_alignment and not np.array_equal(frame_indices, interval_cache.frame_indices.astype(np.int64, copy=False)):
            raise ValueError(f"Frame-index mismatch between embeddings and exact-human-interval cache for robot episode {episode_index}")
        if verify_alignment and not np.array_equal(
            paired_human_episode_indices,
            interval_cache.paired_human_episode_indices.astype(np.int32, copy=False),
        ):
            raise ValueError(f"Paired-human-episode mismatch between pairings and exact-human-interval cache for robot episode {episode_index}")

        predicted_human_u = np.full(paired_human_episode_indices.shape, fill_value=np.nan, dtype=np.float32)
        if predicted_human_u_root is not None:
            episode_predicted_path = predicted_human_u_npz_path(predicted_human_u_root, episode_index)
            if episode_predicted_path.exists():
                with np.load(episode_predicted_path, allow_pickle=False) as predicted_archive:
                    predicted_frame_indices = np.asarray(predicted_archive["frame_indices"], dtype=np.int64)
                    if verify_alignment and not np.array_equal(frame_indices, predicted_frame_indices):
                        raise ValueError(f"Frame-index mismatch between embeddings and predicted human u for robot episode {episode_index}")
                    predicted_human_u = np.asarray(predicted_archive["predicted_human_u"], dtype=np.float32)

        category = category_for_robot_episode(episode_index, categories)
        category_index = int(category_to_index[category.category_id])

        atomic_npz_dump(
            out_path,
            compressed=compressed,
            frame_indices=frame_indices.astype(np.int32, copy=False),
            embeddings=embeddings.astype(np.float32, copy=False),
            normalized_states=normalized_states.astype(np.float32, copy=False),
            global_coefficients=global_coefficients.astype(np.float32, copy=False),
            global_knots=global_knots.astype(np.float32, copy=False),
            global_degree=np.asarray(degree, dtype=np.int32),
            local_start_frame_index=local_start_frame_index.astype(np.int32, copy=False),
            local_end_frame_index=local_end_frame_index.astype(np.int32, copy=False),
            global_local_start_u=global_local_start_u.astype(np.float32, copy=False),
            global_local_end_u=global_local_end_u.astype(np.float32, copy=False),
            exact_local_spline_valid=exact_local_spline_valid,
            exact_local_spline_knot_local_u=exact_local_knot_local_u_flat.astype(np.float32, copy=False),
            exact_local_spline_knot_offsets=exact_local_knot_offsets.astype(np.int32, copy=False),
            exact_local_spline_num_knots=exact_local_num_knots.astype(np.int32, copy=False),
            paired_human_episode_indices=paired_human_episode_indices.astype(np.int32, copy=False),
            human_gt_start_u=interval_cache.human_gt_start_u.astype(np.float32, copy=False),
            human_gt_end_u=interval_cache.human_gt_end_u.astype(np.float32, copy=False),
            interval_valid_mask=interval_cache.interval_valid_mask,
            predicted_human_u=predicted_human_u.astype(np.float32, copy=False),
            category_index=np.asarray(category_index, dtype=np.int32),
        )
        total_episodes += 1
        total_frames += int(frame_indices.shape[0])
        valid_mask = interval_cache.interval_valid_mask & exact_local_spline_valid[:, None]
        total_valid_samples += int(np.count_nonzero(valid_mask))
        total_invalid_samples += int(valid_mask.size - np.count_nonzero(valid_mask))

    return {
        "episodes_written": int(total_episodes),
        "frames_written": int(total_frames),
        "valid_samples_written": int(total_valid_samples),
        "invalid_samples_written": int(total_invalid_samples),
        "compressed": bool(compressed),
    }


class EpisodeChunkBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        dataset: "TranslatorDataset",
        batch_size: int,
        *,
        shuffle: bool,
        drop_last: bool,
        seed: int,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        self.dataset = dataset
        self.batch_size = max(1, int(batch_size))
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = max(1, int(world_size))
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _build_global_batches(self) -> list[list[int]]:
        batches: list[list[int]] = []
        for start, end in self.dataset.episode_sample_ranges:
            sample_count = int(end - start)
            if sample_count <= 0:
                continue
            full_batches = sample_count // self.batch_size
            for batch_offset in range(full_batches):
                chunk_start = start + batch_offset * self.batch_size
                batches.append(list(range(chunk_start, chunk_start + self.batch_size)))
            remainder = sample_count % self.batch_size
            if remainder > 0 and not self.drop_last:
                batches.append(list(range(end - remainder, end)))
        if self.shuffle and batches:
            rng = np.random.default_rng(self.seed + self.epoch)
            rng.shuffle(batches)
        return batches

    def __iter__(self):
        batches = self._build_global_batches()
        if self.world_size > 1:
            if self.drop_last:
                usable = (len(batches) // self.world_size) * self.world_size
                batches = batches[:usable]
            elif batches:
                pad = (-len(batches)) % self.world_size
                if pad > 0:
                    batches.extend(batches[:pad])
            batches = batches[self.rank :: self.world_size]
        return iter(batches)

    def __len__(self) -> int:
        total_batches = 0
        for start, end in self.dataset.episode_sample_ranges:
            sample_count = max(0, int(end - start))
            if self.drop_last:
                total_batches += sample_count // self.batch_size
            else:
                total_batches += int(math.ceil(sample_count / float(self.batch_size))) if sample_count > 0 else 0
        if self.world_size <= 1:
            return total_batches
        if self.drop_last:
            return total_batches // self.world_size
        return int(math.ceil(total_batches / float(self.world_size)))


class TranslatorDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        robot_episode_ids: list[int],
        pairing_slot: int,
        paths: dict[str, Path],
        state_dims: list[int],
        state_mean: np.ndarray,
        state_std: np.ndarray,
        category_to_index: dict[str, int],
        categories: list[CategoryConfig],
        config: dict[str, Any],
    ) -> None:
        self.robot_episode_ids = list(robot_episode_ids)
        self.pairing_slot = int(pairing_slot)
        self.paths = paths
        self.state_dims = np.asarray(state_dims, dtype=np.int64)
        self.state_mean = np.asarray(state_mean, dtype=np.float32)
        self.state_std = np.asarray(state_std, dtype=np.float32)
        self.category_to_index = dict(category_to_index)
        self.categories = list(categories)
        self.robot_embedding_key = str(config["data"]["robot_embedding_key"])
        self.robot_state_source = str(config["data"]["robot_state_source"])
        self.history_length = int(config["data"]["history_length"])
        self.history_stride = int(config["data"]["history_stride"])
        self.verify_alignment = bool(config["data"]["verify_alignment"])
        self.use_predicted_human_u = bool(config["human_input_u"]["use_predicted_human_u"])
        self.require_predicted_human_u = bool(config["human_input_u"]["require_predicted_human_u"])
        self.use_consolidated_human_cache = bool(config["data"].get("use_consolidated_human_cache", False))
        self.use_consolidated_robot_cache = bool(config["data"].get("use_consolidated_robot_cache", False))
        self.human_cache = EpisodeLRUCache(capacity=int(config["data"].get("human_cache_capacity", 64)))
        self.robot_cache = EpisodeLRUCache(capacity=int(config["data"].get("robot_cache_capacity", 8)))
        self.interval_cache = EpisodeLRUCache(capacity=int(config["data"].get("interval_cache_capacity", 8)))
        self.sample_index: list[tuple[int, int, int]] = []
        self.episode_sample_ranges: list[tuple[int, int]] = []
        for episode_index in tqdm(self.robot_episode_ids, desc=f"dataset/slot{self.pairing_slot}", unit="episode", leave=False):
            category = category_for_robot_episode(episode_index, self.categories)
            category_index = self.category_to_index[category.category_id]
            range_start = len(self.sample_index)
            robot_episode = self._load_robot_episode(episode_index)
            if self.use_consolidated_robot_cache:
                if robot_episode.interval_valid_mask is None:
                    raise ValueError(f"Consolidated robot cache is missing interval_valid_mask for episode {episode_index}")
                valid_rows = np.flatnonzero(
                    robot_episode.interval_valid_mask[:, self.pairing_slot] & robot_episode.exact_local_spline_valid
                ).astype(np.int64)
            else:
                interval_cache = self._load_interval_cache(episode_index)
                valid_rows = np.flatnonzero(
                    interval_cache.interval_valid_mask[:, self.pairing_slot] & robot_episode.exact_local_spline_valid
                ).astype(np.int64)
            self.sample_index.extend((episode_index, int(row), int(category_index)) for row in valid_rows.tolist())
            range_end = len(self.sample_index)
            if range_end > range_start:
                self.episode_sample_ranges.append((range_start, range_end))

    def __len__(self) -> int:
        return len(self.sample_index)

    def _load_human_episode(self, episode_index: int) -> HumanSplineEpisode:
        cached = self.human_cache.get(episode_index)
        if cached is not None:
            return cached
        if self.use_consolidated_human_cache:
            consolidated_root = self.paths.get("consolidated_human_cache_root")
            if consolidated_root is None:
                raise KeyError("paths['consolidated_human_cache_root'] is required when use_consolidated_human_cache=true")
            source_path = consolidated_human_spline_npz_path(consolidated_root, episode_index)
        else:
            source_path = human_spline_npz_path(self.paths["human_bspline_root"], episode_index)
        with np.load(source_path, allow_pickle=False) as archive:
            result = HumanSplineEpisode(
                episode_index=episode_index,
                coefficients=np.asarray(archive["global_coefficients"], dtype=np.float32),
                knots=np.asarray(archive["global_knots"], dtype=np.float32),
                degree=int(np.asarray(archive["global_degree"]).reshape(-1)[0]),
            )
        self.human_cache.put(episode_index, result)
        return result

    def _load_robot_episode(self, episode_index: int) -> RobotEpisodeBundle:
        cached = self.robot_cache.get(episode_index)
        if cached is not None:
            return cached

        if self.use_consolidated_robot_cache:
            consolidated_root = self.paths.get("consolidated_robot_cache_root")
            if consolidated_root is None:
                raise KeyError("paths['consolidated_robot_cache_root'] is required when use_consolidated_robot_cache=true")
            with np.load(consolidated_robot_episode_npz_path(consolidated_root, episode_index), allow_pickle=False) as archive:
                result = RobotEpisodeBundle(
                    episode_index=episode_index,
                    frame_indices=np.asarray(archive["frame_indices"], dtype=np.int64),
                    embeddings=np.asarray(archive["embeddings"], dtype=np.float32),
                    states=np.asarray(archive["normalized_states"], dtype=np.float32),
                    global_coefficients=np.asarray(archive["global_coefficients"], dtype=np.float32),
                    global_knots=np.asarray(archive["global_knots"], dtype=np.float32),
                    degree=int(np.asarray(archive["global_degree"]).reshape(-1)[0]),
                    local_start_frame_index=np.asarray(archive["local_start_frame_index"], dtype=np.int32),
                    local_end_frame_index=np.asarray(archive["local_end_frame_index"], dtype=np.int32),
                    global_local_start_u=np.asarray(archive["global_local_start_u"], dtype=np.float32),
                    global_local_end_u=np.asarray(archive["global_local_end_u"], dtype=np.float32),
                    exact_local_spline_valid=np.asarray(archive["exact_local_spline_valid"], dtype=bool),
                    exact_local_knot_local_u_flat=np.asarray(archive["exact_local_spline_knot_local_u"], dtype=np.float32),
                    exact_local_knot_offsets=np.asarray(archive["exact_local_spline_knot_offsets"], dtype=np.int32),
                    exact_local_num_knots=np.asarray(archive["exact_local_spline_num_knots"], dtype=np.int32),
                    paired_human_episode_indices=np.asarray(archive["paired_human_episode_indices"], dtype=np.int32),
                    predicted_human_u=np.asarray(archive["predicted_human_u"], dtype=np.float32),
                    human_gt_start_u=np.asarray(archive["human_gt_start_u"], dtype=np.float32),
                    human_gt_end_u=np.asarray(archive["human_gt_end_u"], dtype=np.float32),
                    interval_valid_mask=np.asarray(archive["interval_valid_mask"], dtype=bool),
                    category_index=int(np.asarray(archive["category_index"]).reshape(-1)[0]),
                )
            self.robot_cache.put(episode_index, result)
            return result

        with np.load(robot_embedding_npz_path(self.paths["robot_embedding_root"], episode_index), allow_pickle=False) as embed_archive:
            embeddings = np.asarray(embed_archive[self.robot_embedding_key], dtype=np.float32)
            frame_indices = np.asarray(embed_archive["frame_indices"], dtype=np.int64)
            state_from_embedding = np.asarray(embed_archive["state"], dtype=np.float32) if "state" in embed_archive.files else None
        if self.robot_state_source == "embedding_file":
            if state_from_embedding is None:
                raise KeyError(f"Robot embedding file does not contain state for episode {episode_index}")
            states = state_from_embedding
        else:
            table = pq.read_table(robot_parquet_path(self.paths["sim_dataset_root"], episode_index), columns=[STATE_COLUMN, "frame_index"])
            states = np.asarray(table[STATE_COLUMN].to_pylist(), dtype=np.float32)
            parquet_frame_indices = np.asarray(table["frame_index"].to_numpy(), dtype=np.int64)
            if self.verify_alignment and not np.array_equal(frame_indices, parquet_frame_indices):
                raise ValueError(f"Frame-index mismatch between embeddings and parquet for robot episode {episode_index}")
        with np.load(robot_spline_npz_path(self.paths["robot_spline_root"], episode_index), allow_pickle=False) as spline_archive:
            global_coefficients = np.asarray(spline_archive["global_coefficients"], dtype=np.float32)
            global_knots = np.asarray(spline_archive["global_knots"], dtype=np.float32)
            degree = int(np.asarray(spline_archive["global_degree"]).reshape(-1)[0])
            spline_frame_indices = np.asarray(spline_archive["frame_indices"], dtype=np.int64)
        if self.verify_alignment and not np.array_equal(frame_indices, spline_frame_indices):
            raise ValueError(f"Frame-index mismatch between embeddings and fitted spline for robot episode {episode_index}")
        with np.load(robot_local_window_npz_path(self.paths["robot_local_window_root"], episode_index), allow_pickle=False) as local_archive:
            local_frame_indices = np.asarray(local_archive["frame_indices"], dtype=np.int64)
            local_start_frame_index = np.asarray(local_archive["local_start_frame_index"], dtype=np.int32)
            local_end_frame_index = np.asarray(local_archive["local_end_frame_index"], dtype=np.int32)
            global_local_start_u = np.asarray(local_archive["global_local_start_u"], dtype=np.float32)
            global_local_end_u = np.asarray(local_archive["global_local_end_u"], dtype=np.float32)
            exact_local_spline_valid = np.asarray(local_archive["exact_local_spline_valid"], dtype=bool)
            exact_local_knot_local_u_flat = np.asarray(local_archive["exact_local_spline_knot_local_u"], dtype=np.float32)
            exact_local_knot_offsets = np.asarray(local_archive["exact_local_spline_knot_offsets"], dtype=np.int32)
            exact_local_num_knots = np.asarray(local_archive["exact_local_spline_num_knots"], dtype=np.int32)
        if self.verify_alignment and not np.array_equal(frame_indices, local_frame_indices):
            raise ValueError(f"Frame-index mismatch between embeddings and local windows for robot episode {episode_index}")
        with np.load(pairing_npz_path(self.paths["robot_pairing_root"], episode_index), allow_pickle=False) as pair_archive:
            pairing_frame_indices = np.asarray(pair_archive["frame_indices"], dtype=np.int64)
            paired_human_episode_indices = np.asarray(pair_archive["paired_human_episode_indices"], dtype=np.int32)
        if self.verify_alignment and not np.array_equal(frame_indices, pairing_frame_indices):
            raise ValueError(f"Frame-index mismatch between embeddings and pairings for robot episode {episode_index}")
        predicted_human_u = None
        predicted_path = self.paths.get("predicted_human_u_root")
        if predicted_path is not None:
            episode_predicted_path = predicted_human_u_npz_path(predicted_path, episode_index)
            if episode_predicted_path.exists():
                with np.load(episode_predicted_path, allow_pickle=False) as predicted_archive:
                    predicted_frame_indices = np.asarray(predicted_archive["frame_indices"], dtype=np.int64)
                    if self.verify_alignment and not np.array_equal(frame_indices, predicted_frame_indices):
                        raise ValueError(f"Frame-index mismatch between embeddings and predicted human u for robot episode {episode_index}")
                    predicted_human_u = np.asarray(predicted_archive["predicted_human_u"], dtype=np.float32)
            elif self.require_predicted_human_u:
                raise FileNotFoundError(episode_predicted_path)
        result = RobotEpisodeBundle(
            episode_index=episode_index,
            frame_indices=frame_indices,
            embeddings=embeddings,
            states=states,
            global_coefficients=global_coefficients,
            global_knots=global_knots,
            degree=degree,
            local_start_frame_index=local_start_frame_index,
            local_end_frame_index=local_end_frame_index,
            global_local_start_u=global_local_start_u,
            global_local_end_u=global_local_end_u,
            exact_local_spline_valid=exact_local_spline_valid,
            exact_local_knot_local_u_flat=exact_local_knot_local_u_flat,
            exact_local_knot_offsets=exact_local_knot_offsets,
            exact_local_num_knots=exact_local_num_knots,
            paired_human_episode_indices=paired_human_episode_indices,
            predicted_human_u=predicted_human_u,
        )
        self.robot_cache.put(episode_index, result)
        return result

    def _load_interval_cache(self, episode_index: int) -> ExactHumanIntervalCache:
        cached = self.interval_cache.get(episode_index)
        if cached is not None:
            return cached
        result = load_exact_human_interval_cache(exact_human_interval_npz_path(self.paths["exact_human_interval_cache_root"], episode_index))
        self.interval_cache.put(episode_index, result)
        return result

    def _history_positions_and_mask(self, row_index: int) -> tuple[np.ndarray, np.ndarray]:
        offsets = np.arange(self.history_length - 1, -1, -1, dtype=np.int64) * self.history_stride
        positions = row_index - offsets
        valid = positions >= 0
        positions = np.maximum(positions, 0)
        return positions.astype(np.int64), valid.astype(bool)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        episode_index, row_index, category_index = self.sample_index[index]
        robot_episode = self._load_robot_episode(episode_index)
        if self.use_consolidated_robot_cache:
            if robot_episode.human_gt_start_u is None or robot_episode.human_gt_end_u is None:
                raise ValueError(f"Consolidated robot cache missing human GT u arrays for episode {episode_index}")
            human_gt_start_u = float(robot_episode.human_gt_start_u[row_index, self.pairing_slot])
            human_gt_end_u = float(robot_episode.human_gt_end_u[row_index, self.pairing_slot])
        else:
            interval_cache = self._load_interval_cache(episode_index)
            human_gt_start_u = float(interval_cache.human_gt_start_u[row_index, self.pairing_slot])
            human_gt_end_u = float(interval_cache.human_gt_end_u[row_index, self.pairing_slot])

        human_episode_index = int(robot_episode.paired_human_episode_indices[row_index, self.pairing_slot])
        human_episode = self._load_human_episode(human_episode_index)
        history_positions, history_valid_mask = self._history_positions_and_mask(row_index)
        robot_history_embeddings = robot_episode.embeddings[history_positions].astype(np.float32)
        robot_history_states = robot_episode.states[history_positions][:, self.state_dims].astype(np.float32) if not self.use_consolidated_robot_cache else robot_episode.states[history_positions].astype(np.float32)
        if not self.use_consolidated_robot_cache:
            robot_history_states = (robot_history_states - self.state_mean) / self.state_std
        robot_current_embedding = robot_episode.embeddings[row_index].astype(np.float32)
        exact_knot_offset = int(robot_episode.exact_local_knot_offsets[row_index])
        exact_knot_count = int(robot_episode.exact_local_num_knots[row_index])
        exact_local_knot_local_u = robot_episode.exact_local_knot_local_u_flat[exact_knot_offset : exact_knot_offset + exact_knot_count]
        predicted_human_start_u = np.asarray(np.nan, dtype=np.float32)
        if self.use_predicted_human_u and robot_episode.predicted_human_u is not None:
            predicted_human_start_u = np.asarray(robot_episode.predicted_human_u[row_index, self.pairing_slot], dtype=np.float32)
        return {
            "robot_history_embeddings": torch.from_numpy(robot_history_embeddings),
            "robot_history_states": torch.from_numpy(robot_history_states),
            "robot_history_mask": torch.from_numpy(history_valid_mask.astype(np.bool_)),
            "robot_current_embedding": torch.from_numpy(robot_current_embedding),
            "human_global_coefficients": torch.from_numpy(human_episode.coefficients),
            "human_global_knots": torch.from_numpy(human_episode.knots),
            "human_gt_start_u": torch.tensor(human_gt_start_u, dtype=torch.float32),
            "human_gt_end_u": torch.tensor(human_gt_end_u, dtype=torch.float32),
            "predicted_human_start_u": torch.tensor(float(predicted_human_start_u), dtype=torch.float32),
            "robot_global_coefficients": torch.from_numpy(robot_episode.global_coefficients),
            "robot_global_knots": torch.from_numpy(robot_episode.global_knots),
            "robot_gt_start_u": torch.tensor(float(robot_episode.global_local_start_u[row_index]), dtype=torch.float32),
            "robot_gt_end_u": torch.tensor(float(robot_episode.global_local_end_u[row_index]), dtype=torch.float32),
            "robot_exact_local_knot_local_u": torch.from_numpy(exact_local_knot_local_u.astype(np.float32, copy=False)),
            "robot_exact_local_num_knots": torch.tensor(int(exact_knot_count), dtype=torch.int32),
            "category_index": torch.tensor(category_index, dtype=torch.int64),
            "robot_episode_index": torch.tensor(episode_index, dtype=torch.int64),
            "robot_frame_row": torch.tensor(row_index, dtype=torch.int64),
            "human_episode_index": torch.tensor(human_episode_index, dtype=torch.int64),
            "pairing_slot": torch.tensor(self.pairing_slot, dtype=torch.int64),
        }


def translator_collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not batch:
        raise ValueError("Cannot collate an empty batch")
    batch_size = len(batch)
    max_human_coefficients = max(int(item["human_global_coefficients"].shape[0]) for item in batch)
    human_feature_dim = int(batch[0]["human_global_coefficients"].shape[1])
    max_human_knots = max(int(item["human_global_knots"].shape[0]) for item in batch)
    max_robot_coefficients = max(int(item["robot_global_coefficients"].shape[0]) for item in batch)
    robot_feature_dim = int(batch[0]["robot_global_coefficients"].shape[1])
    max_robot_knots = max(int(item["robot_global_knots"].shape[0]) for item in batch)
    max_robot_exact_local_knots = max(int(item["robot_exact_local_knot_local_u"].shape[0]) for item in batch)

    human_global_coefficients = torch.zeros((batch_size, max_human_coefficients, human_feature_dim), dtype=torch.float32)
    human_global_knots = torch.zeros((batch_size, max_human_knots), dtype=torch.float32)
    human_global_coeff_counts = torch.zeros((batch_size,), dtype=torch.int32)
    human_global_knot_counts = torch.zeros((batch_size,), dtype=torch.int32)
    robot_global_coefficients = torch.zeros((batch_size, max_robot_coefficients, robot_feature_dim), dtype=torch.float32)
    robot_global_knots = torch.zeros((batch_size, max_robot_knots), dtype=torch.float32)
    robot_global_coeff_counts = torch.zeros((batch_size,), dtype=torch.int32)
    robot_global_knot_counts = torch.zeros((batch_size,), dtype=torch.int32)
    robot_exact_local_knot_local_u = torch.zeros((batch_size, max_robot_exact_local_knots), dtype=torch.float32)

    for batch_index, item in enumerate(batch):
        human_coeff_count = int(item["human_global_coefficients"].shape[0])
        human_knot_count = int(item["human_global_knots"].shape[0])
        robot_coeff_count = int(item["robot_global_coefficients"].shape[0])
        robot_knot_count = int(item["robot_global_knots"].shape[0])
        robot_exact_knot_count = int(item["robot_exact_local_knot_local_u"].shape[0])
        human_global_coefficients[batch_index, :human_coeff_count] = item["human_global_coefficients"]
        human_global_knots[batch_index, :human_knot_count] = item["human_global_knots"]
        human_global_coeff_counts[batch_index] = human_coeff_count
        human_global_knot_counts[batch_index] = human_knot_count
        robot_global_coefficients[batch_index, :robot_coeff_count] = item["robot_global_coefficients"]
        robot_global_knots[batch_index, :robot_knot_count] = item["robot_global_knots"]
        robot_global_coeff_counts[batch_index] = robot_coeff_count
        robot_global_knot_counts[batch_index] = robot_knot_count
        robot_exact_local_knot_local_u[batch_index, :robot_exact_knot_count] = item["robot_exact_local_knot_local_u"]

    return {
        "robot_history_embeddings": torch.stack([item["robot_history_embeddings"] for item in batch], dim=0),
        "robot_history_states": torch.stack([item["robot_history_states"] for item in batch], dim=0),
        "robot_history_mask": torch.stack([item["robot_history_mask"] for item in batch], dim=0),
        "robot_current_embedding": torch.stack([item["robot_current_embedding"] for item in batch], dim=0),
        "human_global_coefficients": human_global_coefficients,
        "human_global_knots": human_global_knots,
        "human_global_coeff_counts": human_global_coeff_counts,
        "human_global_knot_counts": human_global_knot_counts,
        "human_gt_start_u": torch.stack([item["human_gt_start_u"] for item in batch], dim=0),
        "human_gt_end_u": torch.stack([item["human_gt_end_u"] for item in batch], dim=0),
        "predicted_human_start_u": torch.stack([item["predicted_human_start_u"] for item in batch], dim=0),
        "robot_global_coefficients": robot_global_coefficients,
        "robot_global_knots": robot_global_knots,
        "robot_global_coeff_counts": robot_global_coeff_counts,
        "robot_global_knot_counts": robot_global_knot_counts,
        "robot_gt_start_u": torch.stack([item["robot_gt_start_u"] for item in batch], dim=0),
        "robot_gt_end_u": torch.stack([item["robot_gt_end_u"] for item in batch], dim=0),
        "robot_exact_local_knot_local_u": robot_exact_local_knot_local_u,
        "robot_exact_local_num_knots": torch.stack([item["robot_exact_local_num_knots"] for item in batch], dim=0),
        "category_index": torch.stack([item["category_index"] for item in batch], dim=0),
        "robot_episode_index": torch.stack([item["robot_episode_index"] for item in batch], dim=0),
        "robot_frame_row": torch.stack([item["robot_frame_row"] for item in batch], dim=0),
        "human_episode_index": torch.stack([item["human_episode_index"] for item in batch], dim=0),
        "pairing_slot": torch.stack([item["pairing_slot"] for item in batch], dim=0),
    }
