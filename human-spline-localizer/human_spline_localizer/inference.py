from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.amp import autocast
from tqdm.auto import tqdm

from .config import save_resolved_config
from .data import (
    HumanEpisodeFeatures,
    RobotEpisodeFeatures,
    as_path,
    category_for_robot_episode,
    collect_human_episode_ids_from_pairings,
    compute_state_normalization,
    discover_valid_robot_episodes,
    human_cache_npz_path,
    human_spline_npz_path,
    load_category_configs,
    pairing_npz_path,
    prepare_human_spline_cache,
    read_dataset_info,
    resolve_state_dims,
    robot_embedding_npz_path,
    robot_parquet_path,
)
from .model import GlobalSplineLocalizer
from .utils import EpisodeLRUCache, atomic_json_dump


@dataclass(frozen=True)
class InferencePaths:
    sim_dataset_root: Path
    human_dataset_root: Path
    robot_embedding_root: Path
    robot_pairing_root: Path
    human_bspline_root: Path
    localizer_output_root: Path
    human_cache_root: Path
    state_norm_path: Path
    inference_output_root: Path
    resolved_config_path: Path
    run_summary_path: Path


@dataclass(frozen=True)
class InferenceSettings:
    checkpoint_path: Path
    device: str
    batch_size: int
    amp: bool
    overwrite: bool
    save_episode_metadata_json: bool
    max_episodes: int | None
    human_cache_capacity: int
    episode_filter: tuple[int, ...]


def _interval_prediction_enabled(config: dict[str, Any]) -> bool:
    return bool(config["model"]["auxiliary"].get("interval_prediction", {}).get("enabled", False))


def _effective_config(runtime_config: dict[str, Any], checkpoint_config: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(checkpoint_config, dict):
        return runtime_config
    effective = deepcopy(checkpoint_config)
    for key in ("paths", "preprocessing", "inference", "logging"):
        if key in runtime_config:
            effective[key] = deepcopy(runtime_config[key])
    return effective


def _resolve_paths(config: dict[str, Any]) -> InferencePaths:
    paths = config.get("paths")
    inference = config.get("inference")
    preprocessing = config.get("preprocessing")
    if not isinstance(paths, dict):
        raise KeyError("Missing paths block in config.")
    if not isinstance(inference, dict):
        raise KeyError("Missing inference block in config.")
    if not isinstance(preprocessing, dict):
        raise KeyError("Missing preprocessing block in config.")
    localizer_output_root = as_path(paths["output_root"])
    inference_output_root = as_path(inference["output_root"])
    return InferencePaths(
        sim_dataset_root=as_path(paths["sim_dataset_root"]),
        human_dataset_root=as_path(paths["human_dataset_root"]),
        robot_embedding_root=as_path(paths["robot_embedding_root"]),
        robot_pairing_root=as_path(paths["robot_pairing_root"]),
        human_bspline_root=as_path(paths["human_bspline_root"]),
        localizer_output_root=localizer_output_root,
        human_cache_root=localizer_output_root / str(preprocessing["human_cache_dirname"]),
        state_norm_path=localizer_output_root / str(preprocessing["state_norm_filename"]),
        inference_output_root=inference_output_root,
        resolved_config_path=inference_output_root / "resolved_inference_config.yaml",
        run_summary_path=inference_output_root / "run_summary.json",
    )


def _resolve_settings(config: dict[str, Any], checkpoint_path: str | Path | None, output_root: str | Path | None, batch_size: int | None, overwrite: bool, episodes: list[int] | None, max_episodes: int | None, device: str | None) -> InferenceSettings:
    inference = config.get("inference")
    if not isinstance(inference, dict):
        raise KeyError("Missing inference block in config.")
    if checkpoint_path is not None:
        resolved_checkpoint = as_path(checkpoint_path)
    else:
        raw = inference.get("checkpoint_path")
        if raw:
            resolved_checkpoint = as_path(raw)
        else:
            resolved_checkpoint = as_path(config["paths"]["output_root"]) / "checkpoints" / "best.pt"
    if output_root is not None:
        config["inference"]["output_root"] = str(as_path(output_root))
    resolved_device = str(device if device is not None else inference.get("device", "cuda"))
    resolved_batch_size = int(batch_size if batch_size is not None else inference.get("batch_size", 64))
    resolved_max_episodes = int(max_episodes) if max_episodes is not None else (
        int(inference["max_episodes"]) if inference.get("max_episodes") is not None else None
    )
    return InferenceSettings(
        checkpoint_path=resolved_checkpoint,
        device=resolved_device,
        batch_size=resolved_batch_size,
        amp=bool(inference.get("amp", True)),
        overwrite=bool(inference.get("overwrite", False) or overwrite),
        save_episode_metadata_json=bool(inference.get("save_episode_metadata_json", True)),
        max_episodes=resolved_max_episodes,
        human_cache_capacity=int(inference.get("human_cache_capacity", 32)),
        episode_filter=tuple(int(value) for value in (episodes or [])),
    )


def _load_state_norm(config: dict[str, Any], paths: InferencePaths, valid_episode_ids: list[int]) -> dict[str, Any]:
    if paths.state_norm_path.exists():
        return json.loads(paths.state_norm_path.read_text(encoding="utf-8"))
    info = read_dataset_info(paths.sim_dataset_root)
    state_dims = resolve_state_dims(config, info)
    return compute_state_normalization(paths.sim_dataset_root, valid_episode_ids, state_dims, paths.state_norm_path)


def _ensure_output_episode_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory already contains files, pass overwrite to replace it: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _load_robot_episode(
    episode_index: int,
    paths: InferencePaths,
    config: dict[str, Any],
) -> RobotEpisodeFeatures:
    embedding_path = robot_embedding_npz_path(paths.robot_embedding_root, episode_index)
    with np.load(embedding_path, allow_pickle=False) as archive:
        embeddings = np.asarray(archive[str(config["data"]["robot_embedding_key"])], dtype=np.float32)
        frame_indices = np.asarray(archive["frame_indices"], dtype=np.int64)
        state_from_embedding = np.asarray(archive["state"], dtype=np.float32) if "state" in archive.files else None
    robot_state_source = str(config["data"]["robot_state_source"])
    if robot_state_source == "embedding_file":
        if state_from_embedding is None:
            raise KeyError(f"{embedding_path} does not contain a state array.")
        states = state_from_embedding
    else:
        table = pq.read_table(robot_parquet_path(paths.sim_dataset_root, episode_index), columns=[STATE_COLUMN, "frame_index"])
        states = np.asarray(table[STATE_COLUMN].to_pylist(), dtype=np.float32)
        parquet_frame_indices = np.asarray(table["frame_index"].to_numpy(), dtype=np.int64)
        if bool(config["data"]["verify_alignment"]) and not np.array_equal(frame_indices, parquet_frame_indices):
            raise ValueError(f"Frame-index mismatch between embeddings and parquet for robot episode {episode_index}")
    return RobotEpisodeFeatures(
        episode_index=episode_index,
        embeddings=embeddings,
        states=states,
        frame_indices=frame_indices,
    )


def _load_human_episode(
    episode_index: int,
    paths: InferencePaths,
    cache: EpisodeLRUCache,
) -> HumanEpisodeFeatures:
    cached = cache.get(episode_index)
    if cached is not None:
        return cached
    spline_path = human_spline_npz_path(paths.human_bspline_root, episode_index)
    cache_path = human_cache_npz_path(paths.human_cache_root, episode_index)
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
    cache.put(episode_index, result)
    return result


def _history_positions_and_mask(length: int, history_length: int, history_stride: int) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.arange(history_length - 1, -1, -1, dtype=np.int64) * history_stride
    row_positions = np.arange(length, dtype=np.int64)[:, None]
    positions = row_positions - offsets[None, :]
    valid = positions >= 0
    positions = np.maximum(positions, 0)
    return positions.astype(np.int64), valid.astype(bool)


def _move_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}


def _episode_output_dir(output_root: Path, episode_index: int) -> Path:
    return output_root / f"chunk-{episode_index // 1000:03d}" / f"episode_{episode_index:06d}"


def _save_episode_output(
    output_root: Path,
    episode_index: int,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
    overwrite: bool,
    write_metadata_json: bool,
) -> None:
    out_dir = _episode_output_dir(output_root, episode_index)
    _ensure_output_episode_dir(out_dir, overwrite=overwrite)
    np.savez_compressed(out_dir / "predicted_human_u.npz", **arrays)
    if write_metadata_json:
        (out_dir / "predicted_human_u_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def _build_batch(
    row_indices: np.ndarray,
    slot_indices: np.ndarray,
    paired_human_episode_indices: np.ndarray,
    robot_history_embeddings_all: np.ndarray,
    robot_history_states_all: np.ndarray,
    robot_history_mask_all: np.ndarray,
    human_cache: EpisodeLRUCache,
    paths: InferencePaths,
) -> tuple[dict[str, torch.Tensor], list[int]]:
    human_items: list[HumanEpisodeFeatures] = []
    human_episode_ids: list[int] = []
    for row_index, slot_index in zip(row_indices.tolist(), slot_indices.tolist()):
        human_episode_index = int(paired_human_episode_indices[row_index, slot_index])
        human_episode_ids.append(human_episode_index)
        human_items.append(_load_human_episode(human_episode_index, paths, human_cache))

    batch_size = len(human_items)
    max_human_coefficients = max(int(item.coefficients.shape[0]) for item in human_items)
    coefficient_dim = int(human_items[0].coefficients.shape[1])
    phase_bin_count = int(human_items[0].basis_200.shape[0])

    human_coefficients = torch.zeros((batch_size, max_human_coefficients, coefficient_dim), dtype=torch.float32)
    human_left = torch.zeros((batch_size, max_human_coefficients), dtype=torch.float32)
    human_right = torch.zeros((batch_size, max_human_coefficients), dtype=torch.float32)
    human_mid = torch.zeros((batch_size, max_human_coefficients), dtype=torch.float32)
    human_width = torch.zeros((batch_size, max_human_coefficients), dtype=torch.float32)
    human_greville = torch.zeros((batch_size, max_human_coefficients), dtype=torch.float32)
    human_basis = torch.zeros((batch_size, phase_bin_count, max_human_coefficients), dtype=torch.float32)
    human_mask = torch.zeros((batch_size, max_human_coefficients), dtype=torch.bool)

    for batch_index, item in enumerate(human_items):
        length = int(item.coefficients.shape[0])
        human_coefficients[batch_index, :length] = torch.from_numpy(item.coefficients)
        human_left[batch_index, :length] = torch.from_numpy(item.left_support)
        human_right[batch_index, :length] = torch.from_numpy(item.right_support)
        human_mid[batch_index, :length] = torch.from_numpy(item.support_midpoint)
        human_width[batch_index, :length] = torch.from_numpy(item.support_width)
        human_greville[batch_index, :length] = torch.from_numpy(item.greville_phase)
        human_basis[batch_index, :, :length] = torch.from_numpy(item.basis_200)
        human_mask[batch_index, :length] = True

    batch = {
        "robot_history_embeddings": torch.from_numpy(robot_history_embeddings_all[row_indices].astype(np.float32, copy=False)),
        "robot_history_states": torch.from_numpy(robot_history_states_all[row_indices].astype(np.float32, copy=False)),
        "robot_history_mask": torch.from_numpy(robot_history_mask_all[row_indices].astype(np.bool_, copy=False)),
        "human_coefficients": human_coefficients,
        "human_left_support": human_left,
        "human_right_support": human_right,
        "human_support_midpoint": human_mid,
        "human_support_width": human_width,
        "human_greville_phase": human_greville,
        "human_basis_200": human_basis,
        "human_mask": human_mask,
    }
    return batch, human_episode_ids


def export_predicted_human_u(
    config: dict[str, Any],
    checkpoint_path: str | Path | None = None,
    output_root: str | Path | None = None,
    batch_size: int | None = None,
    overwrite: bool = False,
    episodes: list[int] | None = None,
    max_episodes: int | None = None,
    device: str | None = None,
) -> Path:
    runtime_config = deepcopy(config)
    initial_settings = _resolve_settings(runtime_config, checkpoint_path, output_root, batch_size, overwrite, episodes, max_episodes, device)
    payload = torch.load(initial_settings.checkpoint_path, map_location="cpu")
    effective_config = _effective_config(runtime_config, payload.get("config"))
    settings = _resolve_settings(effective_config, checkpoint_path, output_root, batch_size, overwrite, episodes, max_episodes, device)
    paths = _resolve_paths(effective_config)
    paths.inference_output_root.mkdir(parents=True, exist_ok=True)
    save_resolved_config(effective_config, paths.resolved_config_path)

    categories = load_category_configs(effective_config)
    valid_robot_episodes = discover_valid_robot_episodes(
        paths.sim_dataset_root,
        paths.robot_embedding_root,
        paths.robot_pairing_root,
        categories,
        skip_missing=bool(effective_config["data"]["skip_missing_robot_episodes"]),
    )
    category_order = [str(category.category_id) for category in categories]
    valid_episode_ids = [episode for category in categories for episode in valid_robot_episodes.get(category.category_id, [])]
    if settings.episode_filter:
        selected = set(settings.episode_filter)
        valid_episode_ids = [episode for episode in valid_episode_ids if episode in selected]
    if settings.max_episodes is not None:
        valid_episode_ids = valid_episode_ids[: settings.max_episodes]
    if not valid_episode_ids:
        raise RuntimeError("No robot episodes selected for inference.")

    state_norm = _load_state_norm(effective_config, paths, valid_episode_ids)
    human_episode_ids = collect_human_episode_ids_from_pairings(valid_episode_ids, paths.robot_pairing_root)
    prepare_human_spline_cache(
        human_episode_ids,
        paths.human_bspline_root,
        paths.human_cache_root,
        phase_bin_count=int(effective_config["data"]["phase_bin_count"]),
        overwrite=False,
    )

    model = GlobalSplineLocalizer(
        config=effective_config,
        state_dim=len(state_norm["state_dims"]),
    )
    model.load_state_dict(payload["model"])
    torch_device = torch.device(settings.device)
    model.to(torch_device)
    model.eval()
    amp_enabled = bool(settings.amp) and torch_device.type == "cuda"

    state_dims = np.asarray(state_norm["state_dims"], dtype=np.int64)
    state_mean = np.asarray(state_norm["state_mean"], dtype=np.float32)
    state_std = np.asarray(state_norm["state_std"], dtype=np.float32)
    history_length = int(effective_config["data"]["history_length"])
    history_stride = int(effective_config["data"]["history_stride"])
    verify_alignment = bool(effective_config["data"]["verify_alignment"])
    human_cache = EpisodeLRUCache(capacity=settings.human_cache_capacity)
    interval_enabled = _interval_prediction_enabled(effective_config)

    run_summary = {
        "checkpoint_path": str(settings.checkpoint_path),
        "inference_output_root": str(paths.inference_output_root),
        "categories": category_order,
        "num_episodes": int(len(valid_episode_ids)),
        "batch_size": int(settings.batch_size),
        "amp": bool(amp_enabled),
        "interval_prediction_enabled": bool(interval_enabled),
        "phase_bin_count": int(effective_config["data"]["phase_bin_count"]),
        "history_length": int(history_length),
        "history_stride": int(history_stride),
        "episodes": [],
        "total_frames": 0,
        "total_pair_samples": 0,
    }

    outer = tqdm(valid_episode_ids, desc="inference/episodes", unit="episode")
    with torch.no_grad():
        for episode_index in outer:
            robot_episode = _load_robot_episode(episode_index, paths, effective_config)
            pair_path = pairing_npz_path(paths.robot_pairing_root, episode_index)
            with np.load(pair_path, allow_pickle=False) as archive:
                pairing_frame_indices = np.asarray(archive["frame_indices"], dtype=np.int64)
                paired_human_episode_indices = np.asarray(archive["paired_human_episode_indices"], dtype=np.int64)
            if verify_alignment and not np.array_equal(pairing_frame_indices, robot_episode.frame_indices):
                raise ValueError(f"Frame-index mismatch between embeddings and pairings for robot episode {episode_index}")

            num_frames = int(pairing_frame_indices.shape[0])
            num_pairings = int(paired_human_episode_indices.shape[1])
            category = category_for_robot_episode(episode_index, categories)

            normalized_states = ((robot_episode.states[:, state_dims].astype(np.float32) - state_mean) / state_std).astype(np.float32, copy=False)
            history_positions, history_valid_mask = _history_positions_and_mask(num_frames, history_length, history_stride)
            robot_history_embeddings_all = robot_episode.embeddings[history_positions].astype(np.float32, copy=False)
            robot_history_states_all = normalized_states[history_positions].astype(np.float32, copy=False)
            robot_history_mask_all = history_valid_mask.astype(bool, copy=False)

            flat_rows = np.repeat(np.arange(num_frames, dtype=np.int64), num_pairings)
            flat_slots = np.tile(np.arange(num_pairings, dtype=np.int64), num_frames)

            predicted_u = np.full((num_frames, num_pairings), fill_value=np.nan, dtype=np.float32)
            predicted_bin_index = np.full((num_frames, num_pairings), fill_value=-1, dtype=np.int32)
            predicted_c_max = np.full((num_frames, num_pairings), fill_value=np.nan, dtype=np.float32)
            predicted_entropy = np.full((num_frames, num_pairings), fill_value=np.nan, dtype=np.float32)
            predicted_margin = np.full((num_frames, num_pairings), fill_value=np.nan, dtype=np.float32)
            predicted_human_end_u = np.full((num_frames, num_pairings), fill_value=np.nan, dtype=np.float32)
            predicted_delta_u = np.full((num_frames, num_pairings), fill_value=np.nan, dtype=np.float32)

            inner = tqdm(range(0, flat_rows.shape[0], settings.batch_size), desc=f"episode_{episode_index:06d}", unit="batch", leave=False)
            for start in inner:
                stop = min(start + settings.batch_size, flat_rows.shape[0])
                batch_rows = flat_rows[start:stop]
                batch_slots = flat_slots[start:stop]
                batch, human_episode_ids_batch = _build_batch(
                    batch_rows,
                    batch_slots,
                    paired_human_episode_indices,
                    robot_history_embeddings_all,
                    robot_history_states_all,
                    robot_history_mask_all,
                    human_cache,
                    paths,
                )
                batch = _move_to_device(batch, torch_device)
                with autocast(device_type=torch_device.type, enabled=amp_enabled):
                    outputs = model(**batch)
                u_hat = outputs["u_hat"].detach().cpu().to(torch.float32).numpy()
                c_max = outputs["c_max"].detach().cpu().to(torch.float32).numpy()
                entropy = outputs["entropy"].detach().cpu().to(torch.float32).numpy()
                margin = outputs["margin"].detach().cpu().to(torch.float32).numpy()
                bin_index = outputs["logits"].argmax(dim=-1).detach().cpu().to(torch.int32).numpy()

                predicted_u[batch_rows, batch_slots] = u_hat
                predicted_bin_index[batch_rows, batch_slots] = bin_index
                predicted_c_max[batch_rows, batch_slots] = c_max
                predicted_entropy[batch_rows, batch_slots] = entropy
                predicted_margin[batch_rows, batch_slots] = margin

                if interval_enabled and "u_end_hat" in outputs and "delta_u_hat" in outputs:
                    predicted_human_end_u[batch_rows, batch_slots] = outputs["u_end_hat"].detach().cpu().to(torch.float32).numpy()
                    predicted_delta_u[batch_rows, batch_slots] = outputs["delta_u_hat"].detach().cpu().to(torch.float32).numpy()

            arrays = {
                "frame_indices": pairing_frame_indices.astype(np.int32, copy=False),
                "paired_human_episode_indices": paired_human_episode_indices.astype(np.int32, copy=False),
                "predicted_human_u": predicted_u,
                "predicted_human_bin_index": predicted_bin_index,
                "predicted_c_max": predicted_c_max,
                "predicted_entropy": predicted_entropy,
                "predicted_margin": predicted_margin,
            }
            if interval_enabled:
                arrays["predicted_human_end_u"] = predicted_human_end_u
                arrays["predicted_delta_u"] = predicted_delta_u
            metadata = {
                "episode_index": int(episode_index),
                "category_id": category.category_id,
                "num_frames": int(num_frames),
                "num_pairings_per_frame": int(num_pairings),
                "checkpoint_path": str(settings.checkpoint_path),
                "interval_prediction_enabled": bool(interval_enabled),
                "phase_bin_count": int(effective_config["data"]["phase_bin_count"]),
            }
            _save_episode_output(
                paths.inference_output_root,
                episode_index,
                arrays,
                metadata,
                overwrite=settings.overwrite,
                write_metadata_json=settings.save_episode_metadata_json,
            )
            run_summary["episodes"].append(
                {
                    "episode_index": int(episode_index),
                    "category_id": category.category_id,
                    "num_frames": int(num_frames),
                    "num_pair_samples": int(num_frames * num_pairings),
                }
            )
            run_summary["total_frames"] += int(num_frames)
            run_summary["total_pair_samples"] += int(num_frames * num_pairings)

    atomic_json_dump(run_summary, paths.run_summary_path)
    return paths.inference_output_root
