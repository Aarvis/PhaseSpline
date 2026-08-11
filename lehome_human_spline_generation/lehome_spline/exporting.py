from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.amp import autocast
from tqdm.auto import tqdm

from .data import ACTION_COLUMN, IMAGE_COLUMN, STATE_COLUMN, STATE_DIMS, TopViewTransform, discover_episodes
from .model import VisualSplineVAE
from .utils import RunningMoments, atomic_json_dump, atomic_npz


def _episode_stem(episode_index: int) -> str:
    return f"episode_{episode_index:06d}"


def _episode_chunk(episode_index: int) -> str:
    return f"chunk-{episode_index // 1000:03d}"


def _episode_embedding_path(output_root: Path, episode_index: int) -> Path:
    return output_root / _episode_chunk(episode_index) / _episode_stem(episode_index) / "frame_embeddings.npz"


def _frame_u(timestamps: np.ndarray) -> np.ndarray:
    timestamps = np.asarray(timestamps, dtype=np.float64)
    if len(timestamps) < 2:
        return np.zeros(len(timestamps), dtype=np.float32)
    duration = float(timestamps[-1] - timestamps[0])
    if duration <= 0:
        return np.linspace(0.0, 1.0, len(timestamps), dtype=np.float32)
    return ((timestamps - timestamps[0]) / duration).astype(np.float32)


def _load_model(config: dict[str, Any], checkpoint_path: str | Path, device: torch.device) -> tuple[VisualSplineVAE, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = VisualSplineVAE(config)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    return model, checkpoint


def _validate_embedding_file(path: Path, expected_frames: int, latent_size: int) -> None:
    with np.load(path, allow_pickle=False) as data:
        expected = (expected_frames, latent_size)
        if data["mean"].shape != expected or data["log_variance"].shape != expected:
            raise ValueError(f"Invalid embedding shape in {path}: expected {expected}")
        if not np.isfinite(data["mean"]).all() or not np.isfinite(data["log_variance"]).all():
            raise ValueError(f"Non-finite embedding values in {path}")


def export_embeddings(
    config: dict[str, Any],
    checkpoint_path: str | Path,
    overwrite: bool = False,
    max_episodes: int | None = None,
) -> Path:
    output_root = Path(config["output"]["embeddings_dir"]).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(config["export"]["device"] if torch.cuda.is_available() else "cpu")
    model, checkpoint = _load_model(config, checkpoint_path, device)
    episodes = discover_episodes(config["dataset"]["root"])
    if max_episodes:
        episodes = episodes[:max_episodes]
    transform = TopViewTransform(int(config["model"]["dino"]["image_size"]))
    batch_size = int(config["export"]["batch_size"])
    amp_enabled = bool(config["export"]["amp"]) and device.type == "cuda"
    train_indices = set(int(value) for value in checkpoint.get("split_indices", {}).get("train", []))
    moments = RunningMoments(model.latent_size)
    fallback_moments = RunningMoments(model.latent_size)
    completed = 0

    for record in tqdm(episodes, desc="export/episodes", unit="episode"):
        destination = _episode_embedding_path(output_root, record.episode_index)
        if destination.is_file() and not overwrite:
            _validate_embedding_file(destination, record.length, model.latent_size)
            if record.episode_index in train_indices:
                with np.load(destination, allow_pickle=False) as existing:
                    moments.update(existing["mean"])
                    fallback_moments.update(existing["mean"])
            else:
                with np.load(destination, allow_pickle=False) as existing:
                    fallback_moments.update(existing["mean"])
            completed += 1
            continue

        table = pq.read_table(
            record.path,
            columns=[IMAGE_COLUMN, STATE_COLUMN, ACTION_COLUMN, "timestamp", "frame_index"],
        )
        encoded_images = table[IMAGE_COLUMN].to_pylist()
        means: list[np.ndarray] = []
        log_variances: list[np.ndarray] = []
        frame_progress = tqdm(
            range(0, record.length, batch_size),
            desc=f"export/episode-{record.episode_index:06d}",
            unit="frame-batch",
            leave=False,
        )
        with torch.inference_mode():
            for start in frame_progress:
                stop = min(record.length, start + batch_size)
                images = torch.stack([transform(value) for value in encoded_images[start:stop]]).to(
                    device, non_blocking=True
                )
                with autocast(device_type=device.type, enabled=amp_enabled, dtype=torch.bfloat16):
                    dino = model.encode_dino(images)
                    posterior = model.posterior_from_patches(dino.patches)
                means.append(posterior.mean.flatten(1).float().cpu().numpy())
                log_variances.append(posterior.log_variance.flatten(1).float().cpu().numpy())

        mean = np.concatenate(means).astype(np.float16)
        log_variance = np.concatenate(log_variances).astype(np.float16)
        states = np.asarray(table[STATE_COLUMN].to_pylist(), dtype=np.float32)[:, STATE_DIMS]
        actions = np.asarray(table[ACTION_COLUMN].to_pylist(), dtype=np.float32)[:, STATE_DIMS]
        timestamps = np.asarray(table["timestamp"].to_numpy(), dtype=np.float64)
        frame_indices = np.asarray(table["frame_index"].to_numpy(), dtype=np.int64)
        frame_u = _frame_u(timestamps)
        atomic_npz(
            destination,
            compressed=bool(config["export"]["compressed"]),
            mean=mean,
            log_variance=log_variance,
            state=states,
            action=actions,
            timestamps=timestamps,
            frame_indices=frame_indices,
            frame_u=frame_u,
            u=frame_u,
            episode_index=np.asarray(record.episode_index, dtype=np.int64),
        )
        _validate_embedding_file(destination, record.length, model.latent_size)
        if record.episode_index in train_indices:
            moments.update(mean)
        fallback_moments.update(mean)
        completed += 1

    selected_moments = moments if moments.count else fallback_moments
    if selected_moments.count:
        embedding_mean, embedding_std = selected_moments.result()
        atomic_npz(
            output_root / "embedding_normalization.npz",
            compressed=True,
            mean=embedding_mean,
            std=embedding_std,
            count=np.asarray(selected_moments.count, dtype=np.int64),
        )
    atomic_json_dump(
        {
            "checkpoint": str(Path(checkpoint_path).resolve()),
            "dataset": str(Path(config["dataset"]["root"]).resolve()),
            "episodes": completed,
            "layout": "chunk-XXX/episode_XXXXXX/frame_embeddings.npz",
            "latent_shape_per_frame": [model.num_latents, model.model_dim],
            "flattened_latent_size": model.latent_size,
            "dtype": "float16",
            "spline_source": "posterior mean",
            "parameterization": "frame_u = normalized episode timestamp in [0, 1]",
            "state_dimensions": list(STATE_DIMS),
        },
        output_root / "metadata.json",
    )
    return output_root
