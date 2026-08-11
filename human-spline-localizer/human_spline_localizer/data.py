from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from .spline import compute_coefficient_geometry, evaluate_bspline_basis_matrix, normalize_knots_to_unit_domain
from .utils import EpisodeLRUCache, RunningMoments, atomic_json_dump


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
class RobotEpisodeFeatures:
    episode_index: int
    embeddings: np.ndarray
    states: np.ndarray
    frame_indices: np.ndarray


@dataclass(frozen=True)
class HumanEpisodeFeatures:
    episode_index: int
    coefficients: np.ndarray
    left_support: np.ndarray
    right_support: np.ndarray
    support_midpoint: np.ndarray
    support_width: np.ndarray
    greville_phase: np.ndarray
    basis_200: np.ndarray


@dataclass(frozen=True)
class RobotTargetCache:
    frame_indices: np.ndarray
    paired_human_episode_indices: np.ndarray
    target_u: np.ndarray
    target_valid_mask: np.ndarray
    robot_segment_id: np.ndarray
    robot_progress: np.ndarray


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


def pairing_npz_path(root: Path, episode_index: int) -> Path:
    return root / f"chunk-{episode_index // 1000:03d}" / f"episode_{episode_index:06d}" / "human_episode_pairings.npz"


def human_spline_npz_path(root: Path, episode_index: int) -> Path:
    return root / f"chunk-{episode_index // 1000:03d}" / f"episode_{episode_index:06d}" / "spline.npz"


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


def human_cache_npz_path(cache_root: Path, episode_index: int) -> Path:
    return cache_root / f"chunk-{episode_index // 1000:03d}" / f"episode_{episode_index:06d}" / "human_spline_cache.npz"


def robot_target_cache_npz_path(cache_root: Path, episode_index: int) -> Path:
    return cache_root / f"chunk-{episode_index // 1000:03d}" / f"episode_{episode_index:06d}" / "robot_alignment_targets.npz"


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


def load_human_frame_u_by_frame_index(path: Path) -> np.ndarray:
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
    robot_pairing_root: Path,
    categories: list[CategoryConfig],
    skip_missing: bool,
) -> dict[str, list[int]]:
    result: dict[str, list[int]] = {}
    for category in categories:
        episode_ids: list[int] = []
        for episode_index in range(category.sim_start, category.sim_end_exclusive):
            embedding_path = robot_embedding_npz_path(robot_embedding_root, episode_index)
            pairing_path = pairing_npz_path(robot_pairing_root, episode_index)
            annotation_path = checkpoint_json_path(sim_dataset_root, episode_index)
            if embedding_path.exists() and pairing_path.exists() and annotation_path.exists():
                episode_ids.append(episode_index)
                continue
            if not skip_missing:
                missing = [str(path) for path in (embedding_path, pairing_path, annotation_path) if not path.exists()]
                raise FileNotFoundError(f"Robot episode {episode_index} is missing required files: {missing}")
        result[category.category_id] = episode_ids
    return result


def split_robot_episodes_by_category(
    category_to_episode_ids: dict[str, list[int]],
    split_seed: int,
    per_category_ratio: float,
    min_episodes_per_category: int,
) -> dict[str, dict[str, list[int]]]:
    rng = np.random.default_rng(int(split_seed))
    result: dict[str, dict[str, list[int]]] = {}
    for category_id, episode_ids in category_to_episode_ids.items():
        order = rng.permutation(len(episode_ids)).tolist()
        ordered = [episode_ids[index] for index in order]
        if len(ordered) <= 1:
            result[category_id] = {"train": ordered, "val": []}
            continue
        val_count = int(round(len(ordered) * float(per_category_ratio)))
        val_count = max(int(min_episodes_per_category), val_count)
        val_count = min(val_count, len(ordered) - 1)
        result[category_id] = {
            "train": ordered[val_count:],
            "val": ordered[:val_count],
        }
    return result


def collect_human_episode_ids_from_pairings(
    robot_episode_ids: list[int],
    pairing_root: Path,
) -> list[int]:
    human_episode_ids: set[int] = set()
    for episode_index in tqdm(robot_episode_ids, desc="pairings/human-ids", unit="episode"):
        with np.load(pairing_npz_path(pairing_root, episode_index), allow_pickle=False) as archive:
            human_episode_ids.update(int(value) for value in np.asarray(archive["paired_human_episode_indices"]).reshape(-1).tolist())
    return sorted(human_episode_ids)


def prepare_human_spline_cache(
    human_episode_ids: list[int],
    human_bspline_root: Path,
    cache_root: Path,
    phase_bin_count: int,
    overwrite: bool,
) -> None:
    cache_root.mkdir(parents=True, exist_ok=True)
    bin_centers = ((np.arange(int(phase_bin_count), dtype=np.float64) + 0.5) / float(phase_bin_count)).astype(np.float64)
    for episode_index in tqdm(human_episode_ids, desc="precompute/human-splines", unit="episode"):
        out_path = human_cache_npz_path(cache_root, episode_index)
        if out_path.exists() and not overwrite:
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        source_path = human_spline_npz_path(human_bspline_root, episode_index)
        with np.load(source_path, allow_pickle=False) as archive:
            coefficients = np.asarray(archive["global_coefficients"], dtype=np.float32)
            global_knots = np.asarray(archive["global_knots"], dtype=np.float64)
            degree = int(np.asarray(archive["global_degree"]).reshape(-1)[0])
        coefficient_count = int(coefficients.shape[0])
        normalized_knots = normalize_knots_to_unit_domain(global_knots, degree)
        geometry = compute_coefficient_geometry(normalized_knots, degree, coefficient_count)
        basis_200 = evaluate_bspline_basis_matrix(normalized_knots, degree, bin_centers, coefficient_count)
        np.savez_compressed(
            out_path,
            coefficient_count=np.asarray(coefficient_count, dtype=np.int32),
            normalized_knots=normalized_knots.astype(np.float32),
            basis_200=basis_200.astype(np.float32),
            **geometry,
        )


def prepare_robot_alignment_target_cache(
    robot_episode_ids: list[int],
    sim_dataset_root: Path,
    human_dataset_root: Path,
    robot_pairing_root: Path,
    human_bspline_root: Path,
    cache_root: Path,
    overwrite: bool,
) -> None:
    cache_root.mkdir(parents=True, exist_ok=True)
    human_annotation_cache: dict[int, AnnotationEpisode] = {}
    human_frame_u_cache: dict[int, np.ndarray] = {}
    for episode_index in tqdm(robot_episode_ids, desc="precompute/robot-targets", unit="episode"):
        out_path = robot_target_cache_npz_path(cache_root, episode_index)
        if out_path.exists() and not overwrite:
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pairing_path = pairing_npz_path(robot_pairing_root, episode_index)
        with np.load(pairing_path, allow_pickle=False) as archive:
            frame_indices = np.asarray(archive["frame_indices"], dtype=np.int64)
            paired_human_episode_indices = np.asarray(archive["paired_human_episode_indices"], dtype=np.int64)
        robot_annotation = load_annotation_episode(checkpoint_json_path(sim_dataset_root, episode_index), episode_index)
        robot_segment_id = robot_annotation.frame_to_segment_id[frame_indices].astype(np.int32, copy=False)
        robot_progress = robot_annotation.frame_to_progress[frame_indices].astype(np.float32, copy=False)
        target_u = np.full(paired_human_episode_indices.shape, fill_value=np.nan, dtype=np.float32)
        target_valid_mask = np.zeros(paired_human_episode_indices.shape, dtype=bool)
        for row in range(frame_indices.shape[0]):
            segment_id = int(robot_segment_id[row])
            progress = float(robot_progress[row])
            for slot in range(paired_human_episode_indices.shape[1]):
                human_episode_index = int(paired_human_episode_indices[row, slot])
                if human_episode_index not in human_annotation_cache:
                    human_annotation_cache[human_episode_index] = load_annotation_episode(
                        checkpoint_json_path(human_dataset_root, human_episode_index),
                        human_episode_index,
                    )
                if human_episode_index not in human_frame_u_cache:
                    human_frame_u_cache[human_episode_index] = load_human_frame_u_by_frame_index(
                        human_spline_npz_path(human_bspline_root, human_episode_index)
                    )
                human_u, valid = interpolate_human_u(
                    human_annotation_cache[human_episode_index],
                    human_frame_u_cache[human_episode_index],
                    segment_id,
                    progress,
                )
                if valid:
                    target_u[row, slot] = np.asarray(human_u, dtype=np.float32)
                    target_valid_mask[row, slot] = True
        np.savez_compressed(
            out_path,
            frame_indices=frame_indices.astype(np.int32),
            paired_human_episode_indices=paired_human_episode_indices.astype(np.int32),
            target_u=target_u.astype(np.float32),
            target_valid_mask=target_valid_mask,
            robot_segment_id=robot_segment_id.astype(np.int32),
            robot_progress=robot_progress.astype(np.float32),
        )


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


def load_robot_target_cache(path: Path) -> RobotTargetCache:
    with np.load(path, allow_pickle=False) as archive:
        return RobotTargetCache(
            frame_indices=np.asarray(archive["frame_indices"], dtype=np.int32),
            paired_human_episode_indices=np.asarray(archive["paired_human_episode_indices"], dtype=np.int32),
            target_u=np.asarray(archive["target_u"], dtype=np.float32),
            target_valid_mask=np.asarray(archive["target_valid_mask"], dtype=bool),
            robot_segment_id=np.asarray(archive["robot_segment_id"], dtype=np.int32),
            robot_progress=np.asarray(archive["robot_progress"], dtype=np.float32),
        )


class LocalizerDataset(Dataset[dict[str, torch.Tensor]]):
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
        self.robot_embedding_root = paths["robot_embedding_root"]
        self.robot_pairing_root = paths["robot_pairing_root"]
        self.sim_dataset_root = paths["sim_dataset_root"]
        self.human_bspline_root = paths["human_bspline_root"]
        self.human_cache_root = paths["human_cache_root"]
        self.robot_target_cache_root = paths["robot_target_cache_root"]
        self.state_dims = np.asarray(state_dims, dtype=np.int64)
        self.state_mean = np.asarray(state_mean, dtype=np.float32)
        self.state_std = np.asarray(state_std, dtype=np.float32)
        self.category_to_index = dict(category_to_index)
        self.categories = list(categories)
        self.robot_embedding_key = str(config["data"]["robot_embedding_key"])
        self.robot_state_source = str(config["data"]["robot_state_source"])
        self.history_length = int(config["data"]["history_length"])
        self.history_stride = int(config["data"]["history_stride"])
        self.phase_bin_count = int(config["data"]["phase_bin_count"])
        self.target_sigma = float(config["data"]["target_sigma"])
        self.verify_alignment = bool(config["data"]["verify_alignment"])
        self.robot_cache = EpisodeLRUCache(capacity=8)
        self.human_cache = EpisodeLRUCache(capacity=32)
        self.target_cache = EpisodeLRUCache(capacity=16)
        self.bin_centers = ((np.arange(self.phase_bin_count, dtype=np.float32) + 0.5) / float(self.phase_bin_count)).astype(np.float32)
        self.sample_index: list[tuple[int, int, int]] = []
        for episode_index in tqdm(self.robot_episode_ids, desc=f"dataset/slot{self.pairing_slot}", unit="episode", leave=False):
            category = category_for_robot_episode(episode_index, self.categories)
            category_index = self.category_to_index[category.category_id]
            target_cache = load_robot_target_cache(robot_target_cache_npz_path(self.robot_target_cache_root, episode_index))
            valid_rows = np.flatnonzero(target_cache.target_valid_mask[:, self.pairing_slot]).astype(np.int64)
            self.sample_index.extend((episode_index, int(row), int(category_index)) for row in valid_rows.tolist())

    def __len__(self) -> int:
        return len(self.sample_index)

    def _load_robot_episode(self, episode_index: int) -> RobotEpisodeFeatures:
        cached = self.robot_cache.get(episode_index)
        if cached is not None:
            return cached
        embedding_path = robot_embedding_npz_path(self.robot_embedding_root, episode_index)
        with np.load(embedding_path, allow_pickle=False) as archive:
            embeddings = np.asarray(archive[self.robot_embedding_key], dtype=np.float32)
            frame_indices = np.asarray(archive["frame_indices"], dtype=np.int64)
            state_from_embedding = np.asarray(archive["state"], dtype=np.float32) if "state" in archive.files else None
        if self.robot_state_source == "embedding_file":
            if state_from_embedding is None:
                raise KeyError(f"{embedding_path} does not contain a state array.")
            states = state_from_embedding
        else:
            table = pq.read_table(robot_parquet_path(self.sim_dataset_root, episode_index), columns=[STATE_COLUMN, "frame_index"])
            states = np.asarray(table[STATE_COLUMN].to_pylist(), dtype=np.float32)
            parquet_frame_indices = np.asarray(table["frame_index"].to_numpy(), dtype=np.int64)
            if self.verify_alignment and not np.array_equal(frame_indices, parquet_frame_indices):
                raise ValueError(f"Frame-index mismatch between embeddings and parquet for robot episode {episode_index}")
        result = RobotEpisodeFeatures(episode_index=episode_index, embeddings=embeddings, states=states, frame_indices=frame_indices)
        self.robot_cache.put(episode_index, result)
        return result

    def _load_human_episode(self, episode_index: int) -> HumanEpisodeFeatures:
        cached = self.human_cache.get(episode_index)
        if cached is not None:
            return cached
        spline_path = human_spline_npz_path(self.human_bspline_root, episode_index)
        cache_path = human_cache_npz_path(self.human_cache_root, episode_index)
        with np.load(spline_path, allow_pickle=False) as spline_archive:
            coefficients = np.asarray(spline_archive["global_coefficients"], dtype=np.float32)
        with np.load(cache_path, allow_pickle=False) as cache_archive:
            result = HumanEpisodeFeatures(
                episode_index=episode_index,
                coefficients=coefficients,
                left_support=np.asarray(cache_archive["left_support"], dtype=np.float32),
                right_support=np.asarray(cache_archive["right_support"], dtype=np.float32),
                support_midpoint=np.asarray(cache_archive["support_midpoint"], dtype=np.float32),
                support_width=np.asarray(cache_archive["support_width"], dtype=np.float32),
                greville_phase=np.asarray(cache_archive["greville_phase"], dtype=np.float32),
                basis_200=np.asarray(cache_archive["basis_200"], dtype=np.float32),
            )
        self.human_cache.put(episode_index, result)
        return result

    def _load_target_cache(self, episode_index: int) -> RobotTargetCache:
        cached = self.target_cache.get(episode_index)
        if cached is not None:
            return cached
        result = load_robot_target_cache(robot_target_cache_npz_path(self.robot_target_cache_root, episode_index))
        self.target_cache.put(episode_index, result)
        return result

    def _history_positions_and_mask(self, row_index: int) -> tuple[np.ndarray, np.ndarray]:
        offsets = np.arange(self.history_length - 1, -1, -1, dtype=np.int64) * self.history_stride
        positions = row_index - offsets
        valid = positions >= 0
        positions = np.maximum(positions, 0)
        return positions.astype(np.int64), valid.astype(bool)

    def _soft_target(self, target_u: float) -> np.ndarray:
        delta = self.bin_centers.astype(np.float64) - float(target_u)
        logits = -0.5 * (delta / max(self.target_sigma, 1e-8)) ** 2
        logits -= logits.max()
        weights = np.exp(logits)
        weights /= np.maximum(weights.sum(), 1e-12)
        return weights.astype(np.float32)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        episode_index, row_index, category_index = self.sample_index[index]
        robot_episode = self._load_robot_episode(episode_index)
        target_cache = self._load_target_cache(episode_index)
        human_episode_index = int(target_cache.paired_human_episode_indices[row_index, self.pairing_slot])
        human_episode = self._load_human_episode(human_episode_index)
        history_positions, history_valid_mask = self._history_positions_and_mask(row_index)
        robot_history_embeddings = robot_episode.embeddings[history_positions].astype(np.float32)
        robot_history_states = robot_episode.states[history_positions][:, self.state_dims].astype(np.float32)
        robot_history_states = (robot_history_states - self.state_mean) / self.state_std
        target_u = float(target_cache.target_u[row_index, self.pairing_slot])
        soft_target = self._soft_target(target_u)
        checkpoint_target = int(target_cache.robot_segment_id[row_index])
        progress_target = float(target_cache.robot_progress[row_index])
        coefficient_count = int(human_episode.coefficients.shape[0])
        return {
            "robot_history_embeddings": torch.from_numpy(robot_history_embeddings),
            "robot_history_states": torch.from_numpy(robot_history_states),
            "robot_history_mask": torch.from_numpy(history_valid_mask.astype(np.bool_)),
            "human_coefficients": torch.from_numpy(human_episode.coefficients),
            "human_left_support": torch.from_numpy(human_episode.left_support),
            "human_right_support": torch.from_numpy(human_episode.right_support),
            "human_support_midpoint": torch.from_numpy(human_episode.support_midpoint),
            "human_support_width": torch.from_numpy(human_episode.support_width),
            "human_greville_phase": torch.from_numpy(human_episode.greville_phase),
            "human_basis_200": torch.from_numpy(human_episode.basis_200),
            "human_mask": torch.ones((coefficient_count,), dtype=torch.bool),
            "soft_target": torch.from_numpy(soft_target),
            "target_u": torch.tensor(target_u, dtype=torch.float32),
            "checkpoint_target": torch.tensor(checkpoint_target, dtype=torch.int64),
            "progress_target": torch.tensor(progress_target, dtype=torch.float32),
            "category_index": torch.tensor(category_index, dtype=torch.int64),
            "robot_episode_index": torch.tensor(episode_index, dtype=torch.int64),
            "robot_frame_row": torch.tensor(row_index, dtype=torch.int64),
            "human_episode_index": torch.tensor(human_episode_index, dtype=torch.int64),
        }


def localizer_collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not batch:
        raise ValueError("Cannot collate an empty batch")
    batch_size = len(batch)
    max_human_coefficients = max(int(item["human_coefficients"].shape[0]) for item in batch)
    embedding_dim = int(batch[0]["human_coefficients"].shape[1])
    phase_bin_count = int(batch[0]["human_basis_200"].shape[0])
    human_coefficients = torch.zeros((batch_size, max_human_coefficients, embedding_dim), dtype=torch.float32)
    human_left = torch.zeros((batch_size, max_human_coefficients), dtype=torch.float32)
    human_right = torch.zeros((batch_size, max_human_coefficients), dtype=torch.float32)
    human_mid = torch.zeros((batch_size, max_human_coefficients), dtype=torch.float32)
    human_width = torch.zeros((batch_size, max_human_coefficients), dtype=torch.float32)
    human_greville = torch.zeros((batch_size, max_human_coefficients), dtype=torch.float32)
    human_basis = torch.zeros((batch_size, phase_bin_count, max_human_coefficients), dtype=torch.float32)
    human_mask = torch.zeros((batch_size, max_human_coefficients), dtype=torch.bool)
    for batch_index, item in enumerate(batch):
        length = int(item["human_coefficients"].shape[0])
        human_coefficients[batch_index, :length] = item["human_coefficients"]
        human_left[batch_index, :length] = item["human_left_support"]
        human_right[batch_index, :length] = item["human_right_support"]
        human_mid[batch_index, :length] = item["human_support_midpoint"]
        human_width[batch_index, :length] = item["human_support_width"]
        human_greville[batch_index, :length] = item["human_greville_phase"]
        human_basis[batch_index, :, :length] = item["human_basis_200"]
        human_mask[batch_index, :length] = True
    return {
        "robot_history_embeddings": torch.stack([item["robot_history_embeddings"] for item in batch], dim=0),
        "robot_history_states": torch.stack([item["robot_history_states"] for item in batch], dim=0),
        "robot_history_mask": torch.stack([item["robot_history_mask"] for item in batch], dim=0),
        "human_coefficients": human_coefficients,
        "human_left_support": human_left,
        "human_right_support": human_right,
        "human_support_midpoint": human_mid,
        "human_support_width": human_width,
        "human_greville_phase": human_greville,
        "human_basis_200": human_basis,
        "human_mask": human_mask,
        "soft_target": torch.stack([item["soft_target"] for item in batch], dim=0),
        "target_u": torch.stack([item["target_u"] for item in batch], dim=0),
        "checkpoint_target": torch.stack([item["checkpoint_target"] for item in batch], dim=0),
        "progress_target": torch.stack([item["progress_target"] for item in batch], dim=0),
        "category_index": torch.stack([item["category_index"] for item in batch], dim=0),
        "robot_episode_index": torch.stack([item["robot_episode_index"] for item in batch], dim=0),
        "robot_frame_row": torch.stack([item["robot_frame_row"] for item in batch], dim=0),
        "human_episode_index": torch.stack([item["human_episode_index"] for item in batch], dim=0),
    }
