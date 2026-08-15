from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.amp import autocast
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from .config import load_config, save_resolved_config
from .data import as_path, translator_collate
from .losses import compute_losses
from .model import LocalHumanToRobotSplineModel
from .training import (
    _build_derivative_weight_schedules,
    _build_robot_targets,
    _effective_derivative_weights,
    _move_to_device,
)
from .utils import EpisodeLRUCache, MetricAverages, append_jsonl, atomic_json_dump, seed_everything


@dataclass(frozen=True)
class UnseenEvalPair:
    pair_id: str
    category_id: str
    robot_episode_index: int
    human_episode_index: int
    robot_episode_dir: Path
    human_episode_dir: Path
    exact_match_path: Path


@dataclass(frozen=True)
class HumanEpisode:
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
    global_local_start_u: np.ndarray
    global_local_end_u: np.ndarray
    exact_local_spline_valid: np.ndarray
    exact_local_knot_local_u_flat: np.ndarray
    exact_local_knot_offsets: np.ndarray
    exact_local_num_knots: np.ndarray


@dataclass(frozen=True)
class ExactHumanMatchCache:
    frame_indices: np.ndarray
    paired_human_episode_indices: np.ndarray
    human_start_u: np.ndarray
    human_end_u: np.ndarray
    start_valid_mask: np.ndarray
    end_valid_mask: np.ndarray


def _load_eval_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    eval_config = load_config(path, overrides)
    base_config_path = eval_config.get("base_config_path")
    if base_config_path is None:
        raise KeyError("Missing base_config_path in unseen eval config.")
    base_config = load_config(base_config_path)
    merged = copy.deepcopy(base_config)
    merged["unseen_eval"] = eval_config.get("unseen_eval", {})
    merged["base_config_path"] = str(as_path(base_config_path))
    return merged


def _load_unseen_pairs(prep_config_path: Path, exact_match_root: Path, pair_ids: list[str] | None) -> list[UnseenEvalPair]:
    payload = load_config(prep_config_path)
    pairs_payload = payload.get("pairs")
    if not isinstance(pairs_payload, list) or not pairs_payload:
        raise ValueError(f"No pairs found in unseen prep config: {prep_config_path}")
    selected_pair_ids = set(pair_ids or [])
    result: list[UnseenEvalPair] = []
    for item in pairs_payload:
        pair_id = str(item["pair_id"])
        if selected_pair_ids and pair_id not in selected_pair_ids:
            continue
        category_id = str(item["category_id"])
        robot_episode_index = int(item["robot_episode_index"])
        human_episode_index = int(item["human_episode_index"])
        exact_match_path = (
            exact_match_root
            / category_id
            / pair_id
            / f"robot_episode_{robot_episode_index:06d}"
            / "exact_frame_human_u_matches.npz"
        )
        result.append(
            UnseenEvalPair(
                pair_id=pair_id,
                category_id=category_id,
                robot_episode_index=robot_episode_index,
                human_episode_index=human_episode_index,
                robot_episode_dir=as_path(item["robot_episode_dir"]),
                human_episode_dir=as_path(item["human_episode_dir"]),
                exact_match_path=exact_match_path,
            )
        )
    if not result:
        raise RuntimeError(f"No unseen pairs selected from {prep_config_path}")
    return result


def _load_human_episode(human_dir: Path, expected_episode_index: int) -> HumanEpisode:
    path = human_dir / "spline.npz"
    with np.load(path, allow_pickle=False) as archive:
        return HumanEpisode(
            episode_index=expected_episode_index,
            coefficients=np.asarray(archive["global_coefficients"], dtype=np.float32),
            knots=np.asarray(archive["global_knots"], dtype=np.float32),
            degree=int(np.asarray(archive["global_degree"]).reshape(-1)[0]),
        )


def _load_robot_episode(robot_dir: Path, robot_embedding_key: str) -> RobotEpisodeBundle:
    with np.load(robot_dir / "frame_embeddings.npz", allow_pickle=False) as embed_archive:
        embeddings = np.asarray(embed_archive[robot_embedding_key], dtype=np.float32)
        states = np.asarray(embed_archive["state"], dtype=np.float32)
        frame_indices = np.asarray(embed_archive["frame_indices"], dtype=np.int64)
    with np.load(robot_dir / "spline.npz", allow_pickle=False) as spline_archive:
        global_coefficients = np.asarray(spline_archive["global_coefficients"], dtype=np.float32)
        global_knots = np.asarray(spline_archive["global_knots"], dtype=np.float32)
        degree = int(np.asarray(spline_archive["global_degree"]).reshape(-1)[0])
        spline_frame_indices = np.asarray(spline_archive["frame_indices"], dtype=np.int64)
    with np.load(robot_dir / "local_raw_bspline_windows.npz", allow_pickle=False) as local_archive:
        local_frame_indices = np.asarray(local_archive["frame_indices"], dtype=np.int64)
        global_local_start_u = np.asarray(local_archive["global_local_start_u"], dtype=np.float32)
        global_local_end_u = np.asarray(local_archive["global_local_end_u"], dtype=np.float32)
        exact_local_spline_valid = np.asarray(local_archive["exact_local_spline_valid"], dtype=bool)
        exact_local_knot_local_u_flat = np.asarray(local_archive["exact_local_spline_knot_local_u"], dtype=np.float32)
        exact_local_knot_offsets = np.asarray(local_archive["exact_local_spline_knot_offsets"], dtype=np.int32)
        exact_local_num_knots = np.asarray(local_archive["exact_local_spline_num_knots"], dtype=np.int32)
    if not np.array_equal(frame_indices, spline_frame_indices):
        raise ValueError(f"Frame-index mismatch between embeddings and spline in {robot_dir}")
    if not np.array_equal(frame_indices, local_frame_indices):
        raise ValueError(f"Frame-index mismatch between embeddings and local windows in {robot_dir}")
    episode_index = int(np.asarray((np.load(robot_dir / "frame_embeddings.npz", allow_pickle=False))["episode_index"]).reshape(-1)[0])
    return RobotEpisodeBundle(
        episode_index=episode_index,
        frame_indices=frame_indices,
        embeddings=embeddings,
        states=states,
        global_coefficients=global_coefficients,
        global_knots=global_knots,
        degree=degree,
        global_local_start_u=global_local_start_u,
        global_local_end_u=global_local_end_u,
        exact_local_spline_valid=exact_local_spline_valid,
        exact_local_knot_local_u_flat=exact_local_knot_local_u_flat,
        exact_local_knot_offsets=exact_local_knot_offsets,
        exact_local_num_knots=exact_local_num_knots,
    )


def _load_exact_match_cache(path: Path) -> ExactHumanMatchCache:
    with np.load(path, allow_pickle=False) as archive:
        return ExactHumanMatchCache(
            frame_indices=np.asarray(archive["frame_indices"], dtype=np.int64),
            paired_human_episode_indices=np.asarray(archive["paired_human_episode_indices"], dtype=np.int64),
            human_start_u=np.asarray(archive["human_start_u"], dtype=np.float32),
            human_end_u=np.asarray(archive["human_end_u"], dtype=np.float32),
            start_valid_mask=np.asarray(archive["start_valid_mask"], dtype=bool),
            end_valid_mask=np.asarray(archive["end_valid_mask"], dtype=bool),
        )


class UnseenTranslatorEvalDataset(Dataset[dict[str, Tensor]]):
    def __init__(
        self,
        *,
        pairs: list[UnseenEvalPair],
        config: dict[str, Any],
        state_dims: list[int],
        state_mean: np.ndarray,
        state_std: np.ndarray,
        robot_embedding_mean: np.ndarray,
        robot_embedding_std: np.ndarray,
        human_cache_capacity: int,
        robot_cache_capacity: int,
    ) -> None:
        super().__init__()
        self.pairs = list(pairs)
        self.state_dims = np.asarray(state_dims, dtype=np.int64)
        self.state_mean = np.asarray(state_mean, dtype=np.float32)
        self.state_std = np.asarray(state_std, dtype=np.float32)
        self.robot_embedding_mean = np.asarray(robot_embedding_mean, dtype=np.float32)
        self.robot_embedding_std = np.asarray(robot_embedding_std, dtype=np.float32)
        self.history_length = int(config["data"]["history_length"])
        self.history_stride = int(config["data"]["history_stride"])
        self.robot_embedding_key = str(config["data"]["robot_embedding_key"])
        self.interval_mode = str(((config.get("unseen_eval") or {}).get("interval_mode", "exact_gt"))).strip().lower()
        self.fixed_future_robot_frames = (
            int((config.get("unseen_eval") or {}).get("fixed_future_robot_frames", 0))
            if (config.get("unseen_eval") or {}).get("fixed_future_robot_frames") is not None
            else None
        )
        if self.interval_mode not in {"exact_gt", "fixed_future_robot_frames"}:
            raise ValueError(
                f"Unsupported unseen_eval.interval_mode={self.interval_mode!r}. "
                "Expected one of: exact_gt, fixed_future_robot_frames."
            )
        if self.interval_mode == "fixed_future_robot_frames":
            if self.fixed_future_robot_frames is None:
                raise ValueError("unseen_eval.fixed_future_robot_frames is required when interval_mode=fixed_future_robot_frames")
            if self.fixed_future_robot_frames < 0:
                raise ValueError("unseen_eval.fixed_future_robot_frames must be non-negative")
        self.human_cache = EpisodeLRUCache(capacity=human_cache_capacity)
        self.robot_cache = EpisodeLRUCache(capacity=robot_cache_capacity)
        self.exact_cache = EpisodeLRUCache(capacity=robot_cache_capacity)
        category_ids = sorted({pair.category_id for pair in self.pairs})
        self.category_to_index = {category_id: index for index, category_id in enumerate(category_ids)}
        self.sample_index: list[tuple[int, int]] = []
        self.end_row_index_by_pair: dict[int, np.ndarray] = {}
        self.end_u_by_pair: dict[int, np.ndarray] = {}
        self.end_valid_by_pair: dict[int, np.ndarray] = {}
        self.invalid_summary: dict[str, int] = {
            "total_rows": 0,
            "valid_rows": 0,
            "invalid_missing_start": 0,
            "invalid_missing_end": 0,
            "invalid_nonpositive_width": 0,
            "invalid_robot_local_spline": 0,
        }
        for pair_index, pair in enumerate(self.pairs):
            exact_cache = self._get_exact_cache(pair_index)
            robot_episode = self._get_robot_episode(pair_index)
            if not np.array_equal(robot_episode.frame_indices, exact_cache.frame_indices):
                raise ValueError(f"Frame-index mismatch between robot episode and exact human match cache for {pair.pair_id}")
            if exact_cache.paired_human_episode_indices.shape[1] != 1:
                raise ValueError(f"Expected exactly one pairing slot for {pair.pair_id}, got {exact_cache.paired_human_episode_indices.shape[1]}")
            self.invalid_summary["total_rows"] += int(exact_cache.frame_indices.shape[0])
            start_valid = exact_cache.start_valid_mask[:, 0]
            if self.interval_mode == "exact_gt":
                end_row_index = np.arange(exact_cache.frame_indices.shape[0], dtype=np.int64)
                end_u = exact_cache.human_end_u[:, 0]
                end_valid = exact_cache.end_valid_mask[:, 0]
            else:
                end_row_index = np.minimum(
                    np.arange(exact_cache.frame_indices.shape[0], dtype=np.int64) + int(self.fixed_future_robot_frames),
                    exact_cache.frame_indices.shape[0] - 1,
                )
                end_u = exact_cache.human_start_u[end_row_index, 0]
                end_valid = exact_cache.start_valid_mask[end_row_index, 0]
            self.end_row_index_by_pair[pair_index] = end_row_index
            self.end_u_by_pair[pair_index] = end_u.astype(np.float32, copy=False)
            self.end_valid_by_pair[pair_index] = end_valid.astype(bool, copy=False)
            positive_width = end_u > exact_cache.human_start_u[:, 0]
            robot_valid = robot_episode.exact_local_spline_valid
            valid_mask = start_valid & end_valid & positive_width & robot_valid
            self.invalid_summary["valid_rows"] += int(np.count_nonzero(valid_mask))
            self.invalid_summary["invalid_missing_start"] += int(np.count_nonzero(~start_valid))
            self.invalid_summary["invalid_missing_end"] += int(np.count_nonzero(start_valid & ~end_valid))
            self.invalid_summary["invalid_nonpositive_width"] += int(np.count_nonzero(start_valid & end_valid & ~positive_width))
            self.invalid_summary["invalid_robot_local_spline"] += int(np.count_nonzero(start_valid & end_valid & positive_width & ~robot_valid))
            valid_rows = np.flatnonzero(valid_mask).astype(np.int64)
            self.sample_index.extend((pair_index, int(row_index)) for row_index in valid_rows.tolist())

    def __len__(self) -> int:
        return len(self.sample_index)

    def _get_exact_cache(self, pair_index: int) -> ExactHumanMatchCache:
        cached = self.exact_cache.get(pair_index)
        if cached is not None:
            return cached
        result = _load_exact_match_cache(self.pairs[pair_index].exact_match_path)
        self.exact_cache.put(pair_index, result)
        return result

    def _get_robot_episode(self, pair_index: int) -> RobotEpisodeBundle:
        cached = self.robot_cache.get(pair_index)
        if cached is not None:
            return cached
        result = _load_robot_episode(self.pairs[pair_index].robot_episode_dir, self.robot_embedding_key)
        self.robot_cache.put(pair_index, result)
        return result

    def _get_human_episode(self, pair_index: int) -> HumanEpisode:
        cached = self.human_cache.get(pair_index)
        if cached is not None:
            return cached
        pair = self.pairs[pair_index]
        result = _load_human_episode(pair.human_episode_dir, pair.human_episode_index)
        self.human_cache.put(pair_index, result)
        return result

    def _history_positions_and_mask(self, row_index: int) -> tuple[np.ndarray, np.ndarray]:
        offsets = np.arange(self.history_length - 1, -1, -1, dtype=np.int64) * self.history_stride
        positions = row_index - offsets
        valid = positions >= 0
        positions = np.maximum(positions, 0)
        return positions.astype(np.int64), valid.astype(bool)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        pair_index, row_index = self.sample_index[index]
        pair = self.pairs[pair_index]
        exact_cache = self._get_exact_cache(pair_index)
        robot_episode = self._get_robot_episode(pair_index)
        human_episode = self._get_human_episode(pair_index)
        history_positions, history_valid_mask = self._history_positions_and_mask(row_index)
        robot_history_embeddings = robot_episode.embeddings[history_positions].astype(np.float32, copy=False)
        robot_history_states = robot_episode.states[history_positions][:, self.state_dims].astype(np.float32, copy=False)
        robot_history_states = (robot_history_states - self.state_mean) / self.state_std
        robot_current_embedding = robot_episode.embeddings[row_index].astype(np.float32, copy=False)
        robot_current_anchor_embedding = (
            (robot_current_embedding - self.robot_embedding_mean) / self.robot_embedding_std
        ).astype(np.float32, copy=False)
        exact_knot_offset = int(robot_episode.exact_local_knot_offsets[row_index])
        exact_knot_count = int(robot_episode.exact_local_num_knots[row_index])
        exact_local_knot_local_u = robot_episode.exact_local_knot_local_u_flat[
            exact_knot_offset : exact_knot_offset + exact_knot_count
        ]
        human_episode_index = int(exact_cache.paired_human_episode_indices[row_index, 0])
        if human_episode_index != pair.human_episode_index:
            raise ValueError(
                f"Human episode mismatch for {pair.pair_id}: "
                f"expected {pair.human_episode_index}, got {human_episode_index}"
            )
        end_row_index = int(self.end_row_index_by_pair[pair_index][row_index])
        human_end_u = float(self.end_u_by_pair[pair_index][row_index])
        human_end_source_frame_index = int(robot_episode.frame_indices[end_row_index])
        return {
            "robot_history_embeddings": torch.from_numpy(robot_history_embeddings),
            "robot_history_states": torch.from_numpy(robot_history_states),
            "robot_history_mask": torch.from_numpy(history_valid_mask.astype(np.bool_)),
            "robot_current_anchor_embedding": torch.from_numpy(robot_current_anchor_embedding),
            "human_global_coefficients": torch.from_numpy(human_episode.coefficients),
            "human_global_knots": torch.from_numpy(human_episode.knots),
            "human_gt_start_u": torch.tensor(float(exact_cache.human_start_u[row_index, 0]), dtype=torch.float32),
            "human_gt_end_u": torch.tensor(human_end_u, dtype=torch.float32),
            "predicted_human_start_u": torch.tensor(float("nan"), dtype=torch.float32),
            "robot_global_coefficients": torch.from_numpy(robot_episode.global_coefficients),
            "robot_global_knots": torch.from_numpy(robot_episode.global_knots),
            "robot_gt_start_u": torch.tensor(float(robot_episode.global_local_start_u[row_index]), dtype=torch.float32),
            "robot_gt_end_u": torch.tensor(float(robot_episode.global_local_end_u[row_index]), dtype=torch.float32),
            "robot_exact_local_knot_local_u": torch.from_numpy(exact_local_knot_local_u.astype(np.float32, copy=False)),
            "robot_exact_local_num_knots": torch.tensor(exact_knot_count, dtype=torch.int32),
            "category_index": torch.tensor(self.category_to_index[pair.category_id], dtype=torch.int64),
            "robot_episode_index": torch.tensor(pair.robot_episode_index, dtype=torch.int64),
            "robot_frame_row": torch.tensor(row_index, dtype=torch.int64),
            "robot_frame_index": torch.tensor(int(robot_episode.frame_indices[row_index]), dtype=torch.int64),
            "human_episode_index": torch.tensor(human_episode_index, dtype=torch.int64),
            "human_end_source_row": torch.tensor(end_row_index, dtype=torch.int64),
            "human_end_source_frame_index": torch.tensor(human_end_source_frame_index, dtype=torch.int64),
            "pair_index": torch.tensor(pair_index, dtype=torch.int64),
            "pairing_slot": torch.tensor(0, dtype=torch.int64),
        }


def unseen_translator_collate(batch: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    base_batch = []
    for item in batch:
        base_batch.append(
            {
                key: value
                for key, value in item.items()
                if key not in {"pair_index", "robot_frame_index", "human_end_source_row", "human_end_source_frame_index"}
            }
        )
    result = translator_collate(base_batch)
    result["pair_index"] = torch.stack([item["pair_index"] for item in batch], dim=0)
    result["robot_frame_index"] = torch.stack([item["robot_frame_index"] for item in batch], dim=0)
    result["human_end_source_row"] = torch.stack([item["human_end_source_row"] for item in batch], dim=0)
    result["human_end_source_frame_index"] = torch.stack([item["human_end_source_frame_index"] for item in batch], dim=0)
    return result


def _sample_rmse(prediction: Tensor, target: Tensor) -> Tensor:
    prediction32 = prediction.to(torch.float32)
    target32 = target.to(torch.float32)
    dims = tuple(range(1, prediction32.ndim))
    return torch.sqrt(torch.mean((prediction32 - target32) ** 2, dim=dims))


def _sample_masked_huber(prediction: Tensor, target: Tensor, delta: float, mask: Tensor | None = None) -> Tensor:
    prediction32 = prediction.to(torch.float32)
    target32 = target.to(torch.float32)
    loss = F.huber_loss(prediction32, target32, delta=float(delta), reduction="none")
    if mask is None:
        dims = tuple(range(1, loss.ndim))
        return loss.mean(dim=dims)
    while mask.ndim < loss.ndim:
        mask = mask.unsqueeze(-1)
    mask = torch.broadcast_to(mask, loss.shape).to(loss.dtype)
    numer = (loss * mask).sum(dim=tuple(range(1, loss.ndim)))
    denom = torch.clamp(mask.sum(dim=tuple(range(1, mask.ndim))), min=1.0)
    return numer / denom


def _sample_anchor_rmse(reconstructed_robot_curve: Tensor, robot_current_anchor_embedding: Tensor) -> Tensor:
    diff = reconstructed_robot_curve[:, 0, :].to(torch.float32) - robot_current_anchor_embedding.to(torch.float32)
    return torch.sqrt(torch.mean(diff * diff, dim=-1))


def _sample_projection_metrics(
    outputs: dict[str, Tensor],
    targets: dict[str, Tensor],
    batch: dict[str, Tensor],
    config: dict[str, Any],
    effective_loss_weights: dict[str, float],
) -> dict[str, Tensor]:
    dense_cfg = config["loss"]["dense"]
    spline_cfg = config["loss"]["spline"]
    dense_position_loss = _sample_masked_huber(outputs["dense_robot_pred"], targets["dense_position"], dense_cfg["position_delta"])
    dense_velocity_loss = _sample_masked_huber(outputs["dense_robot_velocity"], targets["dense_velocity"], dense_cfg["velocity_delta"])
    dense_acceleration_loss = _sample_masked_huber(outputs["dense_robot_acceleration"], targets["dense_acceleration"], dense_cfg["acceleration_delta"])
    spline_position_loss = _sample_masked_huber(outputs["reconstructed_robot_curve"], targets["dense_position"], spline_cfg["position_delta"])
    spline_velocity_loss = _sample_masked_huber(outputs["reconstructed_robot_velocity"], targets["dense_velocity"], spline_cfg["velocity_delta"])
    spline_acceleration_loss = _sample_masked_huber(outputs["reconstructed_robot_acceleration"], targets["dense_acceleration"], spline_cfg["acceleration_delta"])
    anchor_loss = _sample_masked_huber(
        outputs["reconstructed_robot_curve"][:, 0, :],
        batch["robot_current_anchor_embedding"],
        config["loss"]["anchor"]["delta"],
    )

    gt_span_valid_mask = targets["gt_span_valid_mask"]
    epsilon = float(config["loss"]["knot"]["log_epsilon"])
    target_log = torch.log(targets["gt_span_widths"] + epsilon)
    predicted_log = torch.log(outputs["predicted_span_widths"] + epsilon)
    knot_loss = _sample_masked_huber(
        predicted_log,
        target_log,
        config["loss"]["knot"]["delta"],
        gt_span_valid_mask,
    )

    dense_total = (
        dense_position_loss
        + float(effective_loss_weights["dense_velocity"]) * dense_velocity_loss
        + float(effective_loss_weights["dense_acceleration"]) * dense_acceleration_loss
    )
    spline_total = (
        spline_position_loss
        + float(effective_loss_weights["spline_velocity"]) * spline_velocity_loss
        + float(effective_loss_weights["spline_acceleration"]) * spline_acceleration_loss
    )
    total = (
        dense_total
        + float(config["loss"]["lambda_spline"]) * spline_total
        + float(config["loss"]["lambda_knot"]) * knot_loss
        + float(config["loss"]["lambda_anchor"]) * anchor_loss
    )
    return {
        "total": total,
        "dense_total": dense_total,
        "spline_total": spline_total,
        "dense_position_loss": dense_position_loss,
        "dense_velocity_loss": dense_velocity_loss,
        "dense_acceleration_loss": dense_acceleration_loss,
        "spline_position_loss": spline_position_loss,
        "spline_velocity_loss": spline_velocity_loss,
        "spline_acceleration_loss": spline_acceleration_loss,
        "knot_loss": knot_loss,
        "anchor_loss": anchor_loss,
        "dense_position_rmse": _sample_rmse(outputs["dense_robot_pred"], targets["dense_position"]),
        "dense_velocity_rmse": _sample_rmse(outputs["dense_robot_velocity"], targets["dense_velocity"]),
        "dense_acceleration_rmse": _sample_rmse(outputs["dense_robot_acceleration"], targets["dense_acceleration"]),
        "spline_position_rmse": _sample_rmse(outputs["reconstructed_robot_curve"], targets["dense_position"]),
        "spline_velocity_rmse": _sample_rmse(outputs["reconstructed_robot_velocity"], targets["dense_velocity"]),
        "spline_acceleration_rmse": _sample_rmse(outputs["reconstructed_robot_acceleration"], targets["dense_acceleration"]),
        "anchor_rmse": _sample_anchor_rmse(outputs["reconstructed_robot_curve"], batch["robot_current_anchor_embedding"]),
        "projection_condition_proxy": outputs["projection_condition_proxy"].to(torch.float32),
        "span_entropy": outputs["span_entropy"].to(torch.float32),
        "span_min": outputs["predicted_span_widths"].min(dim=-1).values.to(torch.float32),
        "span_max": outputs["predicted_span_widths"].max(dim=-1).values.to(torch.float32),
        "gt_span_valid_fraction": gt_span_valid_mask.to(torch.float32).mean(dim=-1),
    }


def _sample_metric_value(values: Tensor, sample_index: int) -> float:
    if not torch.is_tensor(values):
        return float(values)
    if values.ndim == 0:
        return float(values.item())
    return float(values[sample_index].item())


def _summarize_records(records: list[dict[str, Any]]) -> dict[str, float]:
    if not records:
        return {}
    metric_keys = [
        "total",
        "dense_total",
        "spline_total",
        "dense_position_loss",
        "dense_velocity_loss",
        "dense_acceleration_loss",
        "spline_position_loss",
        "spline_velocity_loss",
        "spline_acceleration_loss",
        "knot_loss",
        "anchor_loss",
        "dense_position_rmse",
        "dense_velocity_rmse",
        "dense_acceleration_rmse",
        "spline_position_rmse",
        "spline_velocity_rmse",
        "spline_acceleration_rmse",
        "anchor_rmse",
        "projection_condition_proxy",
        "span_entropy",
        "span_min",
        "span_max",
        "gt_span_valid_fraction",
    ]
    result: dict[str, float] = {}
    for key in metric_keys:
        values = [float(item[key]) for item in records if key in item and math.isfinite(float(item[key]))]
        if values:
            result[key] = float(sum(values) / len(values))
    result["sample_count"] = float(len(records))
    return result


def _emit_console_summary(summary: dict[str, Any]) -> None:
    overall = summary.get("overall_metrics", {})
    print("[unseen-eval] completed")
    print(
        "[unseen-eval] overall "
        f"samples={int(summary.get('sample_count', 0))} "
        f"dense_pos_rmse={overall.get('dense_position_rmse', float('nan')):.4f} "
        f"spline_pos_rmse={overall.get('spline_position_rmse', float('nan')):.4f} "
        f"dense_vel_rmse={overall.get('dense_velocity_rmse', float('nan')):.4f} "
        f"spline_vel_rmse={overall.get('spline_velocity_rmse', float('nan')):.4f} "
        f"dense_acc_rmse={overall.get('dense_acceleration_rmse', float('nan')):.4f} "
        f"spline_acc_rmse={overall.get('spline_acceleration_rmse', float('nan')):.4f} "
        f"knot_loss={overall.get('knot_loss', float('nan')):.4f} "
        f"anchor_rmse={overall.get('anchor_rmse', float('nan')):.4f}"
    )
    for category_id, metrics in sorted((summary.get("per_category_metrics") or {}).items()):
        print(
            "[unseen-eval] category "
            f"{category_id} samples={int(metrics.get('sample_count', 0))} "
            f"dense_pos_rmse={metrics.get('dense_position_rmse', float('nan')):.4f} "
            f"spline_pos_rmse={metrics.get('spline_position_rmse', float('nan')):.4f} "
            f"dense_vel_rmse={metrics.get('dense_velocity_rmse', float('nan')):.4f} "
            f"spline_vel_rmse={metrics.get('spline_velocity_rmse', float('nan')):.4f} "
            f"dense_acc_rmse={metrics.get('dense_acceleration_rmse', float('nan')):.4f} "
            f"spline_acc_rmse={metrics.get('spline_acceleration_rmse', float('nan')):.4f}"
        )


def evaluate_unseen_validation(
    config_path: str | Path,
    overrides: list[str] | None = None,
) -> Path:
    config = _load_eval_config(config_path, overrides)
    settings = config.get("unseen_eval", {})
    if not isinstance(settings, dict):
        raise KeyError("Missing unseen_eval block in config.")

    seed_everything(int(config["training"]["seed"]))

    output_root = as_path(settings["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    save_resolved_config(config, output_root / "resolved_unseen_eval_config.yaml")

    prep_config_path = as_path(settings["unseen_prep_config_path"])
    exact_match_root = as_path(settings["exact_match_root"])
    pair_ids = [str(value) for value in settings.get("pair_ids", [])] if settings.get("pair_ids") is not None else None
    pairs = _load_unseen_pairs(prep_config_path, exact_match_root, pair_ids)

    embedding_norm_path = as_path(settings["embedding_normalization_path"])
    with np.load(embedding_norm_path, allow_pickle=False) as archive:
        robot_embedding_mean = np.asarray(archive["mean"], dtype=np.float32)
        robot_embedding_std = np.asarray(archive["std"], dtype=np.float32)

    state_norm_path = as_path(settings.get("state_normalization_path", Path(config["paths"]["output_root"]) / config["preprocessing"]["state_norm_filename"]))
    state_norm = json.loads(state_norm_path.read_text(encoding="utf-8"))

    dataset = UnseenTranslatorEvalDataset(
        pairs=pairs,
        config=config,
        state_dims=[int(value) for value in state_norm["state_dims"]],
        state_mean=np.asarray(state_norm["state_mean"], dtype=np.float32),
        state_std=np.asarray(state_norm["state_std"], dtype=np.float32),
        robot_embedding_mean=robot_embedding_mean,
        robot_embedding_std=robot_embedding_std,
        human_cache_capacity=int(settings.get("human_cache_capacity", 8)),
        robot_cache_capacity=int(settings.get("robot_cache_capacity", 8)),
    )
    if len(dataset) == 0:
        raise RuntimeError("Unseen validation dataset contains zero valid samples.")

    dataloader = DataLoader(
        dataset,
        shuffle=False,
        drop_last=False,
        batch_size=int(settings.get("batch_size", 16)),
        num_workers=int(settings.get("num_workers", 0)),
        pin_memory=bool(settings.get("pin_memory", True)),
        persistent_workers=bool(settings.get("persistent_workers", False)) and int(settings.get("num_workers", 0)) > 0,
        collate_fn=unseen_translator_collate,
    )

    device = torch.device(str(settings.get("device", "cuda")))
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for unseen eval but torch.cuda.is_available() is false.")
        if device.index is not None:
            torch.cuda.set_device(device)
    amp_enabled = bool(settings.get("amp", True)) and device.type == "cuda"

    model = LocalHumanToRobotSplineModel(config=config, state_dim=len(state_norm["state_dims"]))
    checkpoint_path = as_path(settings["checkpoint_path"])
    payload = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(payload["model"])
    model.to(device)
    model.eval()

    derivative_weight_schedules = _build_derivative_weight_schedules(config)
    effective_loss_weights = _effective_derivative_weights(config, derivative_weight_schedules, progress=1.0)
    teacher_forcing_alpha = float(settings.get("teacher_forcing_alpha", 0.0))
    compressor_gradient_gamma = float(settings.get("compressor_gradient_gamma", 0.0))

    averages = MetricAverages()
    pair_records: dict[str, list[dict[str, Any]]] = {pair.pair_id: [] for pair in pairs}
    all_records: list[dict[str, Any]] = []
    pair_outputs: dict[str, dict[str, list[np.ndarray]]] = {
        pair.pair_id: {
            "robot_episode_index": [],
            "robot_frame_row": [],
            "robot_frame_index": [],
            "human_episode_index": [],
            "human_gt_start_u": [],
            "human_gt_end_u": [],
            "robot_gt_start_u": [],
            "robot_gt_end_u": [],
            "predicted_knots": [],
            "predicted_coefficients": [],
        }
        for pair in pairs
    }

    per_sample_jsonl_path = output_root / "per_sample_metrics.jsonl"
    if per_sample_jsonl_path.exists():
        per_sample_jsonl_path.unlink()

    progress = tqdm(dataloader, desc="unseen-translator-eval", unit="batch")
    with torch.no_grad():
        for batch in progress:
            batch = _move_to_device(batch, device)
            targets = _build_robot_targets(batch, model, config)
            with autocast(device_type=device.type, enabled=amp_enabled):
                outputs = model(
                    robot_history_embeddings=batch["robot_history_embeddings"],
                    robot_history_states=batch["robot_history_states"],
                    robot_history_mask=batch["robot_history_mask"],
                    human_global_coefficients=batch["human_global_coefficients"],
                    human_global_knots=batch["human_global_knots"],
                    human_global_coeff_counts=batch["human_global_coeff_counts"],
                    human_global_knot_counts=batch["human_global_knot_counts"],
                    human_input_start_u=batch["human_gt_start_u"],
                    human_input_end_u=batch["human_gt_end_u"],
                    dense_robot_teacher=targets["dense_position"],
                    teacher_forcing_alpha=teacher_forcing_alpha,
                    compressor_gradient_gamma=compressor_gradient_gamma,
                )
                _, metrics = compute_losses(
                    outputs,
                    targets,
                    batch,
                    config,
                    teacher_forcing_alpha=teacher_forcing_alpha,
                    compressor_gradient_gamma=compressor_gradient_gamma,
                    predicted_u_alpha=0.0,
                    effective_loss_weights=effective_loss_weights,
                )
            for value in list(outputs.values()) + list(targets.values()) + list(metrics.values()):
                if torch.is_tensor(value) and (value.is_floating_point() or value.is_complex()) and not torch.isfinite(value).all():
                    raise FloatingPointError("Non-finite tensor encountered during unseen translator evaluation.")
            averages.update(metrics)
            sample_metrics = _sample_projection_metrics(outputs, targets, batch, config, effective_loss_weights)
            current = averages.result()
            progress.set_postfix(
                total=f"{current.get('total', float('nan')):.4f}",
                spline=f"{current.get('spline_position_rmse', float('nan')):.4f}",
            )

            batch_size = int(batch["pair_index"].shape[0])
            for sample_index in range(batch_size):
                pair = pairs[int(batch["pair_index"][sample_index].item())]
                record = {
                    "pair_id": pair.pair_id,
                    "category_id": pair.category_id,
                    "robot_episode_index": int(batch["robot_episode_index"][sample_index].item()),
                    "robot_frame_row": int(batch["robot_frame_row"][sample_index].item()),
                    "robot_frame_index": int(batch["robot_frame_index"][sample_index].item()),
                    "human_episode_index": int(batch["human_episode_index"][sample_index].item()),
                    "human_end_source_row": int(batch["human_end_source_row"][sample_index].item()),
                    "human_end_source_frame_index": int(batch["human_end_source_frame_index"][sample_index].item()),
                    "pairing_slot": int(batch["pairing_slot"][sample_index].item()),
                    "human_gt_start_u": float(batch["human_gt_start_u"][sample_index].item()),
                    "human_gt_end_u": float(batch["human_gt_end_u"][sample_index].item()),
                    "robot_gt_start_u": float(batch["robot_gt_start_u"][sample_index].item()),
                    "robot_gt_end_u": float(batch["robot_gt_end_u"][sample_index].item()),
                }
                for key, values in sample_metrics.items():
                    record[key] = _sample_metric_value(values, sample_index)
                append_jsonl(record, per_sample_jsonl_path)
                all_records.append(record)
                pair_records[pair.pair_id].append(record)

                pair_output = pair_outputs[pair.pair_id]
                pair_output["robot_episode_index"].append(np.asarray(record["robot_episode_index"], dtype=np.int32))
                pair_output["robot_frame_row"].append(np.asarray(record["robot_frame_row"], dtype=np.int32))
                pair_output["robot_frame_index"].append(np.asarray(record["robot_frame_index"], dtype=np.int32))
                pair_output["human_episode_index"].append(np.asarray(record["human_episode_index"], dtype=np.int32))
                pair_output["human_gt_start_u"].append(np.asarray(record["human_gt_start_u"], dtype=np.float32))
                pair_output["human_gt_end_u"].append(np.asarray(record["human_gt_end_u"], dtype=np.float32))
                pair_output["robot_gt_start_u"].append(np.asarray(record["robot_gt_start_u"], dtype=np.float32))
                pair_output["robot_gt_end_u"].append(np.asarray(record["robot_gt_end_u"], dtype=np.float32))
                pair_output["predicted_knots"].append(outputs["predicted_knots"][sample_index].detach().cpu().to(torch.float32).numpy())
                pair_output["predicted_coefficients"].append(outputs["predicted_coefficients"][sample_index].detach().cpu().to(torch.float32).numpy())

    if bool(settings.get("save_pair_npz", True)):
        pair_output_root = output_root / "pair_outputs"
        pair_output_root.mkdir(parents=True, exist_ok=True)
        for pair in pairs:
            payload_dict = pair_outputs[pair.pair_id]
            np.savez_compressed(
                pair_output_root / f"{pair.pair_id}.npz",
                robot_episode_index=np.asarray(payload_dict["robot_episode_index"], dtype=np.int32),
                robot_frame_row=np.asarray(payload_dict["robot_frame_row"], dtype=np.int32),
                robot_frame_index=np.asarray(payload_dict["robot_frame_index"], dtype=np.int32),
                human_episode_index=np.asarray(payload_dict["human_episode_index"], dtype=np.int32),
                human_gt_start_u=np.asarray(payload_dict["human_gt_start_u"], dtype=np.float32),
                human_gt_end_u=np.asarray(payload_dict["human_gt_end_u"], dtype=np.float32),
                robot_gt_start_u=np.asarray(payload_dict["robot_gt_start_u"], dtype=np.float32),
                robot_gt_end_u=np.asarray(payload_dict["robot_gt_end_u"], dtype=np.float32),
                predicted_knots=np.stack(payload_dict["predicted_knots"], axis=0) if payload_dict["predicted_knots"] else np.zeros((0,), dtype=np.float32),
                predicted_coefficients=(
                    np.stack(payload_dict["predicted_coefficients"], axis=0)
                    if payload_dict["predicted_coefficients"]
                    else np.zeros((0,), dtype=np.float32)
                ),
            )

    per_pair_metrics = {pair_id: _summarize_records(records) for pair_id, records in pair_records.items()}
    per_category_records: dict[str, list[dict[str, Any]]] = {}
    for record in all_records:
        per_category_records.setdefault(str(record["category_id"]), []).append(record)
    per_category_metrics = {category_id: _summarize_records(records) for category_id, records in per_category_records.items()}

    overall_metrics = _summarize_records(all_records)
    summary = {
        "checkpoint_path": str(checkpoint_path),
        "base_config_path": str(config["base_config_path"]),
        "output_root": str(output_root),
        "sample_count": len(all_records),
        "pair_count": len(pairs),
        "overall_metrics": overall_metrics,
        "per_pair_metrics": per_pair_metrics,
        "per_category_metrics": per_category_metrics,
        "dataset_summary": dataset.invalid_summary,
        "pairs": [
            {
                "pair_id": pair.pair_id,
                "category_id": pair.category_id,
                "robot_episode_index": pair.robot_episode_index,
                "human_episode_index": pair.human_episode_index,
                "robot_episode_dir": str(pair.robot_episode_dir),
                "human_episode_dir": str(pair.human_episode_dir),
                "exact_match_path": str(pair.exact_match_path),
            }
            for pair in pairs
        ],
        "evaluation_settings": {
            "batch_size": int(settings.get("batch_size", 16)),
            "device": str(device),
            "amp": bool(amp_enabled),
            "interval_mode": str(settings.get("interval_mode", "exact_gt")),
            "fixed_future_robot_frames": settings.get("fixed_future_robot_frames"),
            "teacher_forcing_alpha": teacher_forcing_alpha,
            "compressor_gradient_gamma": compressor_gradient_gamma,
            "effective_loss_weights": {key: float(value) for key, value in effective_loss_weights.items()},
        },
    }
    atomic_json_dump(summary, output_root / "run_summary.json")
    _emit_console_summary(
        {
            "sample_count": len(all_records),
            "overall_metrics": overall_metrics,
            "per_category_metrics": per_category_metrics,
        }
    )
    return output_root
