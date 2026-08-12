from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.multiprocessing as mp
from torch.amp import autocast
from tqdm.auto import tqdm

from .config import save_resolved_config
from .data import (
    STATE_COLUMN,
    ExactHumanIntervalCache,
    HumanSplineEpisode,
    as_path,
    category_for_robot_episode,
    collect_human_episode_ids_from_pairings,
    compute_state_normalization,
    discover_valid_robot_episodes,
    exact_human_interval_npz_path,
    human_spline_npz_path,
    load_category_configs,
    load_exact_human_interval_cache,
    pairing_npz_path,
    predicted_human_u_npz_path,
    precompute_exact_human_interval_cache,
    read_dataset_info,
    resolve_state_dims,
    robot_embedding_npz_path,
    robot_local_window_npz_path,
    robot_parquet_path,
)
from .model import LocalHumanToRobotSplineModel
from .utils import EpisodeLRUCache, atomic_json_dump


@dataclass(frozen=True)
class InferencePaths:
    sim_dataset_root: Path
    human_dataset_root: Path
    sim_embedding_root: Path
    human_bspline_root: Path
    robot_embedding_root: Path
    robot_spline_root: Path
    robot_local_window_root: Path
    robot_pairing_root: Path
    predicted_human_u_root: Path | None
    training_output_root: Path
    exact_human_interval_cache_root: Path
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
    episode_filter: tuple[int, ...]
    human_cache_capacity: int
    predicted_u_alpha: float
    teacher_forcing_alpha: float
    compressor_gradient_gamma: float


@dataclass(frozen=True)
class RobotInferenceEpisode:
    episode_index: int
    frame_indices: np.ndarray
    embeddings: np.ndarray
    normalized_states: np.ndarray
    paired_human_episode_indices: np.ndarray
    human_gt_start_u: np.ndarray
    human_gt_end_u: np.ndarray
    interval_valid_mask: np.ndarray
    predicted_human_u: np.ndarray | None


def _inference_distributed_settings(config: dict[str, Any]) -> dict[str, Any]:
    payload = config.get("inference", {}).get("distributed", {})
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError("inference.distributed must be a mapping when provided.")
    return payload


def _normalize_gpu_ids(raw_value: Any) -> list[int] | None:
    if raw_value is None:
        return None
    if isinstance(raw_value, str):
        lowered = raw_value.strip().lower()
        if lowered in {"", "auto", "all", "none", "null"}:
            return None
        raise ValueError(f"Unsupported GPU selector string: {raw_value!r}")
    if not isinstance(raw_value, (list, tuple)):
        raise ValueError(f"inference.distributed.gpu_ids must be null or a list of integers, got {type(raw_value)!r}")
    gpu_ids = [int(value) for value in raw_value]
    if len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError(f"inference.distributed.gpu_ids contains duplicates: {gpu_ids}")
    if any(value < 0 for value in gpu_ids):
        raise ValueError(f"inference.distributed.gpu_ids must be non-negative: {gpu_ids}")
    return gpu_ids


def _parse_explicit_cuda_device_index(device_raw: str) -> int | None:
    lowered = device_raw.strip().lower()
    if not lowered.startswith("cuda:"):
        return None
    suffix = lowered.split(":", 1)[1].strip()
    if not suffix:
        return None
    try:
        return int(suffix)
    except ValueError as exc:
        raise ValueError(f"Unsupported CUDA device selector: {device_raw!r}") from exc


def _resolve_requested_gpu_ids(config: dict[str, Any], device_override: str | None) -> list[int]:
    inference_cfg = config.get("inference", {})
    requested_device = str(device_override if device_override is not None else inference_cfg.get("device", "cuda")).strip()
    requested_device_lower = requested_device.lower()
    if not requested_device_lower.startswith("cuda"):
        return []
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for inference but torch.cuda.is_available() is false.")
    available_gpu_count = torch.cuda.device_count()
    if available_gpu_count <= 0:
        return []
    explicit_device_index = _parse_explicit_cuda_device_index(requested_device_lower)
    if explicit_device_index is not None:
        if explicit_device_index >= available_gpu_count:
            raise ValueError(
                f"Requested CUDA device index {explicit_device_index} is out of range for "
                f"torch.cuda.device_count()={available_gpu_count}."
            )
        return [explicit_device_index]
    explicit_gpu_ids = _normalize_gpu_ids(_inference_distributed_settings(config).get("gpu_ids"))
    if explicit_gpu_ids is None:
        return list(range(available_gpu_count))
    invalid = [gpu_id for gpu_id in explicit_gpu_ids if gpu_id >= available_gpu_count]
    if invalid:
        raise ValueError(
            f"Requested inference GPU ids {invalid} are out of range for torch.cuda.device_count()={available_gpu_count}."
        )
    return explicit_gpu_ids


def _parallel_inference_requested(config: dict[str, Any], gpu_ids: list[int], device_override: str | None) -> bool:
    settings = _inference_distributed_settings(config)
    raw_value = settings.get("enabled", "auto")
    if isinstance(raw_value, bool):
        return bool(raw_value) and len(gpu_ids) > 1
    mode = str(raw_value).strip().lower()
    if mode == "auto":
        requested_device = str(device_override if device_override is not None else config.get("inference", {}).get("device", "cuda")).strip().lower()
        return requested_device.startswith("cuda") and len(gpu_ids) > 1
    if mode in {"true", "1", "yes", "y", "on"}:
        return len(gpu_ids) > 1
    if mode in {"false", "0", "no", "n", "off"}:
        return False
    raise ValueError(f"Unsupported inference.distributed.enabled value: {raw_value!r}")


def _shard_episode_ids(episode_ids: list[int], shard_count: int) -> list[list[int]]:
    if shard_count <= 1:
        return [list(episode_ids)]
    return [list(episode_ids[offset::shard_count]) for offset in range(shard_count)]


def _worker_summary_path(output_root: Path, worker_rank: int) -> Path:
    return output_root / "_worker_summaries" / f"worker_{worker_rank:03d}.json"


def _effective_config(runtime_config: dict[str, Any], checkpoint_config: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(checkpoint_config, dict):
        return runtime_config
    effective = deepcopy(checkpoint_config)
    for key in ("paths", "categories", "data", "human_input_u", "preprocessing", "inference", "logging"):
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
    sim_embedding_root = as_path(paths["sim_embedding_root"])
    training_output_root = as_path(paths["output_root"])
    predicted_u_root_raw = paths.get("predicted_human_u_root")
    predicted_u_root = as_path(predicted_u_root_raw) if predicted_u_root_raw else None
    inference_output_root = as_path(inference["output_root"])
    return InferencePaths(
        sim_dataset_root=as_path(paths["sim_dataset_root"]),
        human_dataset_root=as_path(paths["human_dataset_root"]),
        sim_embedding_root=sim_embedding_root,
        human_bspline_root=as_path(paths["human_bspline_root"]),
        robot_embedding_root=sim_embedding_root / "frame_embeddings",
        robot_spline_root=sim_embedding_root / "fitted_bspline_splines",
        robot_local_window_root=sim_embedding_root / str(paths["robot_local_window_dirname"]),
        robot_pairing_root=sim_embedding_root / str(paths["robot_pairing_dirname"]),
        predicted_human_u_root=predicted_u_root,
        training_output_root=training_output_root,
        exact_human_interval_cache_root=training_output_root / str(preprocessing["exact_human_interval_cache_dirname"]),
        state_norm_path=training_output_root / str(preprocessing["state_norm_filename"]),
        inference_output_root=inference_output_root,
        resolved_config_path=inference_output_root / "resolved_inference_config.yaml",
        run_summary_path=inference_output_root / "run_summary.json",
    )


def _resolve_settings(
    config: dict[str, Any],
    checkpoint_path: str | Path | None,
    output_root: str | Path | None,
    batch_size: int | None,
    overwrite: bool,
    episodes: list[int] | None,
    max_episodes: int | None,
    device: str | None,
) -> InferenceSettings:
    inference = config.get("inference")
    if not isinstance(inference, dict):
        raise KeyError("Missing inference block in config.")
    if output_root is not None:
        config["inference"]["output_root"] = str(as_path(output_root))
    if checkpoint_path is not None:
        resolved_checkpoint = as_path(checkpoint_path)
    else:
        raw_checkpoint = inference.get("checkpoint_path")
        if raw_checkpoint:
            resolved_checkpoint = as_path(raw_checkpoint)
        else:
            resolved_checkpoint = as_path(config["paths"]["output_root"]) / "checkpoints" / "best.pt"
    resolved_batch_size = int(batch_size if batch_size is not None else inference.get("batch_size", 32))
    resolved_max_episodes = int(max_episodes) if max_episodes is not None else (
        int(inference["max_episodes"]) if inference.get("max_episodes") is not None else None
    )
    resolved_device = str(device if device is not None else inference.get("device", "cuda"))
    return InferenceSettings(
        checkpoint_path=resolved_checkpoint,
        device=resolved_device,
        batch_size=resolved_batch_size,
        amp=bool(inference.get("amp", True)),
        overwrite=bool(inference.get("overwrite", False) or overwrite),
        save_episode_metadata_json=bool(inference.get("save_episode_metadata_json", True)),
        max_episodes=resolved_max_episodes,
        episode_filter=tuple(int(value) for value in (episodes or [])),
        human_cache_capacity=int(inference.get("human_cache_capacity", 32)),
        predicted_u_alpha=float(inference.get("predicted_u_alpha", 1.0)),
        teacher_forcing_alpha=float(inference.get("teacher_forcing_alpha", 0.0)),
        compressor_gradient_gamma=float(inference.get("compressor_gradient_gamma", 0.0)),
    )


def _load_state_norm(config: dict[str, Any], paths: InferencePaths, valid_episode_ids: list[int]) -> dict[str, Any]:
    if paths.state_norm_path.exists():
        return json.loads(paths.state_norm_path.read_text(encoding="utf-8"))
    info = read_dataset_info(paths.sim_dataset_root)
    state_dims = resolve_state_dims(config, info)
    return compute_state_normalization(paths.sim_dataset_root, valid_episode_ids, state_dims, paths.state_norm_path)


def _ensure_exact_human_interval_cache(
    robot_episode_ids: list[int],
    paths: InferencePaths,
    overwrite: bool,
) -> dict[str, Any]:
    missing = [
        episode_index
        for episode_index in robot_episode_ids
        if overwrite or not exact_human_interval_npz_path(paths.exact_human_interval_cache_root, episode_index).exists()
    ]
    if not missing:
        return {
            "episodes_processed": 0,
            "total_samples": 0,
            "valid_samples": 0,
            "invalid_samples": 0,
            "valid_fraction": 1.0,
        }
    return precompute_exact_human_interval_cache(
        missing,
        paths.sim_dataset_root,
        paths.human_dataset_root,
        paths.robot_pairing_root,
        paths.robot_local_window_root,
        paths.human_bspline_root,
        paths.exact_human_interval_cache_root,
        overwrite=overwrite,
    ) | {"episodes_processed": int(len(missing))}


def _ensure_output_episode_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory already contains files, pass overwrite to replace it: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _load_human_episode(episode_index: int, root: Path, cache: EpisodeLRUCache) -> HumanSplineEpisode:
    cached = cache.get(episode_index)
    if cached is not None:
        return cached
    spline_path = human_spline_npz_path(root, episode_index)
    with np.load(spline_path, allow_pickle=False) as archive:
        result = HumanSplineEpisode(
            episode_index=episode_index,
            coefficients=np.asarray(archive["global_coefficients"], dtype=np.float32),
            knots=np.asarray(archive["global_knots"], dtype=np.float32),
            degree=int(np.asarray(archive["global_degree"]).reshape(-1)[0]),
        )
    cache.put(episode_index, result)
    return result


def _load_robot_episode(
    episode_index: int,
    paths: InferencePaths,
    config: dict[str, Any],
    state_dims: np.ndarray,
    state_mean: np.ndarray,
    state_std: np.ndarray,
) -> RobotInferenceEpisode:
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

    with np.load(pairing_npz_path(paths.robot_pairing_root, episode_index), allow_pickle=False) as archive:
        pairing_frame_indices = np.asarray(archive["frame_indices"], dtype=np.int64)
        paired_human_episode_indices = np.asarray(archive["paired_human_episode_indices"], dtype=np.int32)
    if bool(config["data"]["verify_alignment"]) and not np.array_equal(frame_indices, pairing_frame_indices):
        raise ValueError(f"Frame-index mismatch between embeddings and pairings for robot episode {episode_index}")

    interval_cache = load_exact_human_interval_cache(exact_human_interval_npz_path(paths.exact_human_interval_cache_root, episode_index))
    if bool(config["data"]["verify_alignment"]) and not np.array_equal(frame_indices, interval_cache.frame_indices.astype(np.int64, copy=False)):
        raise ValueError(f"Frame-index mismatch between embeddings and exact human interval cache for robot episode {episode_index}")

    predicted_human_u = None
    if paths.predicted_human_u_root is not None:
        predicted_path = predicted_human_u_npz_path(paths.predicted_human_u_root, episode_index)
        if predicted_path.exists():
            with np.load(predicted_path, allow_pickle=False) as archive:
                predicted_frame_indices = np.asarray(archive["frame_indices"], dtype=np.int64)
                if bool(config["data"]["verify_alignment"]) and not np.array_equal(frame_indices, predicted_frame_indices):
                    raise ValueError(f"Frame-index mismatch between embeddings and predicted human u for robot episode {episode_index}")
                predicted_human_u = np.asarray(archive["predicted_human_u"], dtype=np.float32)

    normalized_states = ((states[:, state_dims].astype(np.float32) - state_mean) / state_std).astype(np.float32, copy=False)
    return RobotInferenceEpisode(
        episode_index=episode_index,
        frame_indices=frame_indices,
        embeddings=embeddings.astype(np.float32, copy=False),
        normalized_states=normalized_states,
        paired_human_episode_indices=paired_human_episode_indices,
        human_gt_start_u=interval_cache.human_gt_start_u.astype(np.float32, copy=False),
        human_gt_end_u=interval_cache.human_gt_end_u.astype(np.float32, copy=False),
        interval_valid_mask=interval_cache.interval_valid_mask.astype(bool, copy=False),
        predicted_human_u=predicted_human_u,
    )


def _history_positions_and_mask(length: int, history_length: int, history_stride: int) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.arange(history_length - 1, -1, -1, dtype=np.int64) * history_stride
    row_positions = np.arange(length, dtype=np.int64)[:, None]
    positions = row_positions - offsets[None, :]
    valid = positions >= 0
    positions = np.maximum(positions, 0)
    return positions.astype(np.int64), valid.astype(bool)


def _select_human_input_interval(
    gt_start_u: np.ndarray,
    gt_end_u: np.ndarray,
    predicted_start_u: np.ndarray,
    predicted_u_alpha: float,
    min_width: float,
) -> tuple[np.ndarray, np.ndarray]:
    gt_width = np.maximum(gt_end_u - gt_start_u, float(min_width))
    has_prediction = np.isfinite(predicted_start_u)
    blended = (1.0 - float(predicted_u_alpha)) * gt_start_u + float(predicted_u_alpha) * predicted_start_u
    start_u = np.where(has_prediction, blended, gt_start_u)
    start_u = np.clip(start_u, 0.0, 1.0 - float(min_width))
    end_u = np.minimum(start_u + gt_width, 1.0)
    min_end = np.minimum(start_u + float(min_width), 1.0)
    end_u = np.maximum(end_u, min_end)
    return start_u.astype(np.float32, copy=False), end_u.astype(np.float32, copy=False)


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
    np.savez_compressed(out_dir / "predicted_robot_local_splines.npz", **arrays)
    if write_metadata_json:
        (out_dir / "predicted_robot_local_splines_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def _build_batch(
    row_indices: np.ndarray,
    slot_indices: np.ndarray,
    robot_episode: RobotInferenceEpisode,
    robot_history_embeddings_all: np.ndarray,
    robot_history_states_all: np.ndarray,
    robot_history_mask_all: np.ndarray,
    human_input_start_u: np.ndarray,
    human_input_end_u: np.ndarray,
    human_cache: EpisodeLRUCache,
    human_bspline_root: Path,
) -> dict[str, torch.Tensor]:
    human_items: list[HumanSplineEpisode] = []
    for row_index, slot_index in zip(row_indices.tolist(), slot_indices.tolist()):
        human_episode_index = int(robot_episode.paired_human_episode_indices[row_index, slot_index])
        human_items.append(_load_human_episode(human_episode_index, human_bspline_root, human_cache))

    batch_size = len(human_items)
    max_human_coefficients = max(int(item.coefficients.shape[0]) for item in human_items)
    coeff_dim = int(human_items[0].coefficients.shape[1])
    max_human_knots = max(int(item.knots.shape[0]) for item in human_items)

    human_global_coefficients = torch.zeros((batch_size, max_human_coefficients, coeff_dim), dtype=torch.float32)
    human_global_knots = torch.zeros((batch_size, max_human_knots), dtype=torch.float32)
    human_global_coeff_counts = torch.zeros((batch_size,), dtype=torch.int32)
    human_global_knot_counts = torch.zeros((batch_size,), dtype=torch.int32)

    for batch_index, item in enumerate(human_items):
        coeff_count = int(item.coefficients.shape[0])
        knot_count = int(item.knots.shape[0])
        human_global_coefficients[batch_index, :coeff_count] = torch.from_numpy(item.coefficients)
        human_global_knots[batch_index, :knot_count] = torch.from_numpy(item.knots)
        human_global_coeff_counts[batch_index] = coeff_count
        human_global_knot_counts[batch_index] = knot_count

    return {
        "robot_history_embeddings": torch.from_numpy(robot_history_embeddings_all[row_indices].astype(np.float32, copy=False)),
        "robot_history_states": torch.from_numpy(robot_history_states_all[row_indices].astype(np.float32, copy=False)),
        "robot_history_mask": torch.from_numpy(robot_history_mask_all[row_indices].astype(np.bool_, copy=False)),
        "human_global_coefficients": human_global_coefficients,
        "human_global_knots": human_global_knots,
        "human_global_coeff_counts": human_global_coeff_counts,
        "human_global_knot_counts": human_global_knot_counts,
        "human_input_start_u": torch.from_numpy(human_input_start_u.astype(np.float32, copy=False)),
        "human_input_end_u": torch.from_numpy(human_input_end_u.astype(np.float32, copy=False)),
    }


def _sample_finite_mask(outputs: dict[str, torch.Tensor]) -> np.ndarray:
    checks = [
        torch.isfinite(outputs["dense_robot_pred"]).reshape(outputs["dense_robot_pred"].shape[0], -1).all(dim=-1),
        torch.isfinite(outputs["predicted_span_widths"]).all(dim=-1),
        torch.isfinite(outputs["predicted_knots"]).all(dim=-1),
        torch.isfinite(outputs["predicted_coefficients"]).reshape(outputs["predicted_coefficients"].shape[0], -1).all(dim=-1),
        torch.isfinite(outputs["projection_condition_proxy"]),
        torch.isfinite(outputs["span_entropy"]),
    ]
    finite = checks[0]
    for mask in checks[1:]:
        finite = finite & mask
    return finite.detach().cpu().numpy().astype(bool, copy=False)


def _export_episode_subset(
    effective_config: dict[str, Any],
    settings: InferenceSettings,
    paths: InferencePaths,
    valid_episode_ids: list[int],
    state_norm: dict[str, Any],
    cache_summary: dict[str, Any],
    *,
    worker_rank: int,
    world_size: int,
    gpu_id: int | None,
) -> dict[str, Any]:
    payload = torch.load(settings.checkpoint_path, map_location="cpu")
    model = LocalHumanToRobotSplineModel(
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
    min_input_width = float(effective_config["human_input_u"]["min_interval_width"])
    use_predicted_human_u = bool(effective_config["human_input_u"]["use_predicted_human_u"])
    human_cache = EpisodeLRUCache(capacity=settings.human_cache_capacity)

    control_count = int(effective_config["model"]["span_decoder"]["num_output_spans"]) + int(effective_config["model"]["degree"])
    knot_count = control_count + int(effective_config["model"]["degree"]) + 1
    span_count = int(effective_config["model"]["span_decoder"]["num_output_spans"])
    coeff_dim = int(effective_config["model"]["dense_head"]["output_dim"])

    run_summary: dict[str, Any] = {
        "worker_rank": int(worker_rank),
        "world_size": int(world_size),
        "gpu_id": int(gpu_id) if gpu_id is not None else None,
        "checkpoint_path": str(settings.checkpoint_path),
        "inference_output_root": str(paths.inference_output_root),
        "device": str(settings.device),
        "num_episodes": int(len(valid_episode_ids)),
        "batch_size": int(settings.batch_size),
        "amp": bool(amp_enabled),
        "predicted_u_alpha": float(settings.predicted_u_alpha),
        "teacher_forcing_alpha": float(settings.teacher_forcing_alpha),
        "compressor_gradient_gamma": float(settings.compressor_gradient_gamma),
        "history_length": int(history_length),
        "history_stride": int(history_stride),
        "span_count": int(span_count),
        "control_count": int(control_count),
        "knot_count": int(knot_count),
        "coefficient_dim": int(coeff_dim),
        "exact_human_interval_cache": cache_summary,
        "episodes": [],
        "total_frames": 0,
        "total_pairs": 0,
        "total_interval_valid_pairs": 0,
        "total_predicted_valid_pairs": 0,
        "total_nonfinite_pairs": 0,
    }

    outer = tqdm(
        valid_episode_ids,
        desc=f"inference/gpu{gpu_id}" if gpu_id is not None else "inference/episodes",
        unit="episode",
        position=int(worker_rank),
        leave=True,
    )
    with torch.no_grad():
        for episode_index in outer:
            robot_episode = _load_robot_episode(
                episode_index,
                paths,
                effective_config,
                state_dims=state_dims,
                state_mean=state_mean,
                state_std=state_std,
            )
            num_frames = int(robot_episode.frame_indices.shape[0])
            num_pairings = int(robot_episode.paired_human_episode_indices.shape[1])
            category = category_for_robot_episode(episode_index, categories)

            history_positions, history_valid_mask = _history_positions_and_mask(num_frames, history_length, history_stride)
            robot_history_embeddings_all = robot_episode.embeddings[history_positions].astype(np.float32, copy=False)
            robot_history_states_all = robot_episode.normalized_states[history_positions].astype(np.float32, copy=False)
            robot_history_mask_all = history_valid_mask.astype(bool, copy=False)

            interval_valid_mask = robot_episode.interval_valid_mask.astype(bool, copy=False)
            flat_rows, flat_slots = np.nonzero(interval_valid_mask)

            predicted_human_start_u = np.full((num_frames, num_pairings), fill_value=np.nan, dtype=np.float32)
            if use_predicted_human_u and robot_episode.predicted_human_u is not None:
                predicted_human_start_u[:, :] = robot_episode.predicted_human_u.astype(np.float32, copy=False)

            human_input_start_u = np.full((num_frames, num_pairings), fill_value=np.nan, dtype=np.float32)
            human_input_end_u = np.full((num_frames, num_pairings), fill_value=np.nan, dtype=np.float32)
            predicted_robot_knots = np.full((num_frames, num_pairings, knot_count), fill_value=np.nan, dtype=np.float32)
            predicted_robot_coefficients = np.full((num_frames, num_pairings, control_count, coeff_dim), fill_value=np.nan, dtype=np.float32)
            predicted_robot_span_widths = np.full((num_frames, num_pairings, span_count), fill_value=np.nan, dtype=np.float32)
            projection_condition_proxy = np.full((num_frames, num_pairings), fill_value=np.nan, dtype=np.float32)
            span_entropy = np.full((num_frames, num_pairings), fill_value=np.nan, dtype=np.float32)
            prediction_valid_mask = np.zeros((num_frames, num_pairings), dtype=bool)

            inner = tqdm(
                range(0, flat_rows.shape[0], settings.batch_size),
                desc=f"episode_{episode_index:06d}",
                unit="batch",
                leave=False,
                disable=world_size > 1,
            )
            for start in inner:
                stop = min(start + settings.batch_size, flat_rows.shape[0])
                batch_rows = flat_rows[start:stop]
                batch_slots = flat_slots[start:stop]
                batch_gt_start = robot_episode.human_gt_start_u[batch_rows, batch_slots]
                batch_gt_end = robot_episode.human_gt_end_u[batch_rows, batch_slots]
                batch_predicted_start = predicted_human_start_u[batch_rows, batch_slots]
                batch_input_start, batch_input_end = _select_human_input_interval(
                    batch_gt_start,
                    batch_gt_end,
                    batch_predicted_start,
                    predicted_u_alpha=settings.predicted_u_alpha,
                    min_width=min_input_width,
                )
                human_input_start_u[batch_rows, batch_slots] = batch_input_start
                human_input_end_u[batch_rows, batch_slots] = batch_input_end
                batch = _build_batch(
                    batch_rows,
                    batch_slots,
                    robot_episode,
                    robot_history_embeddings_all,
                    robot_history_states_all,
                    robot_history_mask_all,
                    batch_input_start,
                    batch_input_end,
                    human_cache,
                    paths.human_bspline_root,
                )
                batch = _move_to_device(batch, torch_device)
                with autocast(device_type=torch_device.type, enabled=amp_enabled):
                    outputs = model(
                        robot_history_embeddings=batch["robot_history_embeddings"],
                        robot_history_states=batch["robot_history_states"],
                        robot_history_mask=batch["robot_history_mask"],
                        human_global_coefficients=batch["human_global_coefficients"],
                        human_global_knots=batch["human_global_knots"],
                        human_global_coeff_counts=batch["human_global_coeff_counts"],
                        human_global_knot_counts=batch["human_global_knot_counts"],
                        human_input_start_u=batch["human_input_start_u"],
                        human_input_end_u=batch["human_input_end_u"],
                        dense_robot_teacher=None,
                        teacher_forcing_alpha=settings.teacher_forcing_alpha,
                        compressor_gradient_gamma=settings.compressor_gradient_gamma,
                    )

                finite_mask = _sample_finite_mask(outputs)
                if not np.any(finite_mask):
                    continue
                valid_rows = batch_rows[finite_mask]
                valid_slots = batch_slots[finite_mask]
                prediction_valid_mask[valid_rows, valid_slots] = True
                predicted_robot_knots[valid_rows, valid_slots] = outputs["predicted_knots"][finite_mask].detach().cpu().to(torch.float32).numpy()
                predicted_robot_coefficients[valid_rows, valid_slots] = outputs["predicted_coefficients"][finite_mask].detach().cpu().to(torch.float32).numpy()
                predicted_robot_span_widths[valid_rows, valid_slots] = outputs["predicted_span_widths"][finite_mask].detach().cpu().to(torch.float32).numpy()
                projection_condition_proxy[valid_rows, valid_slots] = outputs["projection_condition_proxy"][finite_mask].detach().cpu().to(torch.float32).numpy()
                span_entropy[valid_rows, valid_slots] = outputs["span_entropy"][finite_mask].detach().cpu().to(torch.float32).numpy()

            episode_total_pairs = int(num_frames * num_pairings)
            episode_interval_valid_pairs = int(interval_valid_mask.sum())
            episode_predicted_valid_pairs = int(prediction_valid_mask.sum())
            episode_nonfinite_pairs = int(episode_interval_valid_pairs - episode_predicted_valid_pairs)

            arrays = {
                "frame_indices": robot_episode.frame_indices.astype(np.int32, copy=False),
                "paired_human_episode_indices": robot_episode.paired_human_episode_indices.astype(np.int32, copy=False),
                "human_interval_valid_mask": interval_valid_mask,
                "prediction_valid_mask": prediction_valid_mask,
                "human_gt_start_u": robot_episode.human_gt_start_u.astype(np.float32, copy=False),
                "human_gt_end_u": robot_episode.human_gt_end_u.astype(np.float32, copy=False),
                "predicted_human_start_u": predicted_human_start_u,
                "human_input_start_u": human_input_start_u,
                "human_input_end_u": human_input_end_u,
                "predicted_robot_knots": predicted_robot_knots,
                "predicted_robot_coefficients": predicted_robot_coefficients,
                "predicted_robot_span_widths": predicted_robot_span_widths,
                "projection_condition_proxy": projection_condition_proxy,
                "span_entropy": span_entropy,
            }
            metadata = {
                "episode_index": int(episode_index),
                "category_id": category.category_id,
                "num_frames": int(num_frames),
                "num_pairings_per_frame": int(num_pairings),
                "checkpoint_path": str(settings.checkpoint_path),
                "predicted_u_alpha": float(settings.predicted_u_alpha),
                "teacher_forcing_alpha": float(settings.teacher_forcing_alpha),
                "compressor_gradient_gamma": float(settings.compressor_gradient_gamma),
                "degree": int(effective_config["model"]["degree"]),
                "span_count": int(span_count),
                "control_count": int(control_count),
                "knot_count": int(knot_count),
                "coefficient_dim": int(coeff_dim),
                "interval_valid_pairs": int(episode_interval_valid_pairs),
                "predicted_valid_pairs": int(episode_predicted_valid_pairs),
                "nonfinite_pairs": int(episode_nonfinite_pairs),
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
                    "num_pairs": int(episode_total_pairs),
                    "interval_valid_pairs": int(episode_interval_valid_pairs),
                    "predicted_valid_pairs": int(episode_predicted_valid_pairs),
                    "nonfinite_pairs": int(episode_nonfinite_pairs),
                }
            )
            run_summary["total_frames"] += int(num_frames)
            run_summary["total_pairs"] += int(episode_total_pairs)
            run_summary["total_interval_valid_pairs"] += int(episode_interval_valid_pairs)
            run_summary["total_predicted_valid_pairs"] += int(episode_predicted_valid_pairs)
            run_summary["total_nonfinite_pairs"] += int(episode_nonfinite_pairs)

    return run_summary


def _write_worker_summary(output_root: Path, worker_rank: int, summary: dict[str, Any]) -> None:
    summary_path = _worker_summary_path(output_root, worker_rank)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(summary, summary_path)


def _aggregate_worker_summaries(
    worker_summaries: list[dict[str, Any]],
    cache_summary: dict[str, Any],
    gpu_ids: list[int],
) -> dict[str, Any]:
    if not worker_summaries:
        raise RuntimeError("No worker summaries were produced during inference export.")
    first = worker_summaries[0]
    aggregated: dict[str, Any] = {
        "checkpoint_path": first["checkpoint_path"],
        "inference_output_root": first["inference_output_root"],
        "num_episodes": 0,
        "batch_size": first["batch_size"],
        "amp": first["amp"],
        "predicted_u_alpha": first["predicted_u_alpha"],
        "teacher_forcing_alpha": first["teacher_forcing_alpha"],
        "compressor_gradient_gamma": first["compressor_gradient_gamma"],
        "history_length": first["history_length"],
        "history_stride": first["history_stride"],
        "span_count": first["span_count"],
        "control_count": first["control_count"],
        "knot_count": first["knot_count"],
        "coefficient_dim": first["coefficient_dim"],
        "exact_human_interval_cache": cache_summary,
        "distributed": {
            "enabled": bool(len(gpu_ids) > 1),
            "world_size": int(len(worker_summaries)),
            "gpu_ids": [int(value) for value in gpu_ids],
        },
        "workers": [],
        "episodes": [],
        "total_frames": 0,
        "total_pairs": 0,
        "total_interval_valid_pairs": 0,
        "total_predicted_valid_pairs": 0,
        "total_nonfinite_pairs": 0,
    }
    for summary in worker_summaries:
        aggregated["num_episodes"] += int(summary["num_episodes"])
        aggregated["workers"].append(
            {
                "worker_rank": int(summary["worker_rank"]),
                "gpu_id": int(summary["gpu_id"]) if summary["gpu_id"] is not None else None,
                "device": str(summary["device"]),
                "num_episodes": int(summary["num_episodes"]),
                "total_frames": int(summary["total_frames"]),
                "total_pairs": int(summary["total_pairs"]),
                "total_interval_valid_pairs": int(summary["total_interval_valid_pairs"]),
                "total_predicted_valid_pairs": int(summary["total_predicted_valid_pairs"]),
                "total_nonfinite_pairs": int(summary["total_nonfinite_pairs"]),
            }
        )
        aggregated["episodes"].extend(summary["episodes"])
        aggregated["total_frames"] += int(summary["total_frames"])
        aggregated["total_pairs"] += int(summary["total_pairs"])
        aggregated["total_interval_valid_pairs"] += int(summary["total_interval_valid_pairs"])
        aggregated["total_predicted_valid_pairs"] += int(summary["total_predicted_valid_pairs"])
        aggregated["total_nonfinite_pairs"] += int(summary["total_nonfinite_pairs"])
    aggregated["episodes"].sort(key=lambda item: int(item["episode_index"]))
    aggregated["workers"].sort(key=lambda item: int(item["worker_rank"]))
    return aggregated


def _inference_worker(
    worker_rank: int,
    gpu_ids: tuple[int, ...],
    episode_shards: tuple[tuple[int, ...], ...],
    effective_config: dict[str, Any],
    settings: InferenceSettings,
    cache_summary: dict[str, Any],
) -> None:
    gpu_id = int(gpu_ids[worker_rank])
    torch.cuda.set_device(gpu_id)
    worker_settings = InferenceSettings(
        checkpoint_path=settings.checkpoint_path,
        device=f"cuda:{gpu_id}",
        batch_size=settings.batch_size,
        amp=settings.amp,
        overwrite=settings.overwrite,
        save_episode_metadata_json=settings.save_episode_metadata_json,
        max_episodes=None,
        episode_filter=tuple(int(value) for value in episode_shards[worker_rank]),
        human_cache_capacity=settings.human_cache_capacity,
        predicted_u_alpha=settings.predicted_u_alpha,
        teacher_forcing_alpha=settings.teacher_forcing_alpha,
        compressor_gradient_gamma=settings.compressor_gradient_gamma,
    )
    paths = _resolve_paths(effective_config)
    episode_ids = list(worker_settings.episode_filter)
    state_norm = _load_state_norm(effective_config, paths, episode_ids)
    summary = _export_episode_subset(
        effective_config,
        worker_settings,
        paths,
        episode_ids,
        state_norm,
        cache_summary,
        worker_rank=worker_rank,
        world_size=len(gpu_ids),
        gpu_id=gpu_id,
    )
    _write_worker_summary(paths.inference_output_root, worker_rank, summary)


def export_predicted_robot_splines(
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
    valid_robot_episodes_by_category = discover_valid_robot_episodes(
        paths.sim_dataset_root,
        paths.robot_embedding_root,
        paths.robot_spline_root,
        paths.robot_local_window_root,
        paths.robot_pairing_root,
        categories,
        skip_missing=bool(effective_config["data"]["skip_missing_robot_episodes"]),
    )
    valid_episode_ids = [episode for category in categories for episode in valid_robot_episodes_by_category.get(category.category_id, [])]
    if settings.episode_filter:
        selected = set(settings.episode_filter)
        valid_episode_ids = [episode for episode in valid_episode_ids if episode in selected]
    if settings.max_episodes is not None:
        valid_episode_ids = valid_episode_ids[: settings.max_episodes]
    if not valid_episode_ids:
        raise RuntimeError("No robot episodes selected for inference.")

    state_norm = _load_state_norm(effective_config, paths, valid_episode_ids)
    cache_summary = _ensure_exact_human_interval_cache(valid_episode_ids, paths, overwrite=False)

    gpu_ids = _resolve_requested_gpu_ids(effective_config, settings.device)
    if _parallel_inference_requested(effective_config, gpu_ids, settings.device):
        raw_shards = _shard_episode_ids(valid_episode_ids, len(gpu_ids))
        active_pairs = [(int(gpu_id), tuple(int(value) for value in shard)) for gpu_id, shard in zip(gpu_ids, raw_shards) if shard]
        if len(active_pairs) > 1:
            active_gpu_ids = tuple(gpu_id for gpu_id, _ in active_pairs)
            active_episode_shards = tuple(shard for _, shard in active_pairs)
            worker_summary_dir = _worker_summary_path(paths.inference_output_root, 0).parent
            worker_summary_dir.mkdir(parents=True, exist_ok=True)
            for rank in range(len(active_gpu_ids)):
                summary_path = _worker_summary_path(paths.inference_output_root, rank)
                if summary_path.exists():
                    summary_path.unlink()
            mp.spawn(
                _inference_worker,
                nprocs=len(active_gpu_ids),
                args=(active_gpu_ids, active_episode_shards, effective_config, settings, cache_summary),
                join=True,
            )
            worker_summaries = [
                json.loads(_worker_summary_path(paths.inference_output_root, rank).read_text(encoding="utf-8"))
                for rank in range(len(active_gpu_ids))
            ]
            run_summary = _aggregate_worker_summaries(worker_summaries, cache_summary, list(active_gpu_ids))
            atomic_json_dump(run_summary, paths.run_summary_path)
            for rank in range(len(active_gpu_ids)):
                summary_path = _worker_summary_path(paths.inference_output_root, rank)
                if summary_path.exists():
                    summary_path.unlink()
            if worker_summary_dir.exists():
                try:
                    worker_summary_dir.rmdir()
                except OSError:
                    pass
            return paths.inference_output_root

    summary = _export_episode_subset(
        effective_config,
        settings,
        paths,
        valid_episode_ids,
        state_norm,
        cache_summary,
        worker_rank=0,
        world_size=1,
        gpu_id=int(gpu_ids[0]) if gpu_ids else None,
    )
    run_summary = _aggregate_worker_summaries([summary], cache_summary, gpu_ids[:1] if gpu_ids else [])
    atomic_json_dump(run_summary, paths.run_summary_path)
    return paths.inference_output_root
