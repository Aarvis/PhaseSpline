from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from .config import save_resolved_config
from .data import (
    TemporalWindowDataset,
    compute_state_normalization,
    discover_episodes,
    save_splits,
    split_episodes,
)
from .losses import (
    diagonal_gaussian_kl,
    feature_reconstruction_loss,
    mean_alignment_loss,
    normalized_horizon_weights,
    standard_normal_kl,
    variance_floor_loss,
)
from .model import VisualSplineVAE
from .utils import atomic_json_dump, move_to_device, seed_everything


LOGGER = logging.getLogger("lehome_spline.training")


@dataclass
class Stage:
    name: str
    epochs: int
    learning_rate: float
    stochastic: bool
    temporal: bool
    kl_max: float
    kl_warmup_fraction: float
    fraction: float = 1.0


class MetricAverages:
    def __init__(self) -> None:
        self.sums: dict[str, float] = {}
        self.count = 0

    def update(self, metrics: dict[str, Tensor | float]) -> None:
        self.count += 1
        for key, value in metrics.items():
            scalar = float(value.detach().item()) if torch.is_tensor(value) else float(value)
            self.sums[key] = self.sums.get(key, 0.0) + scalar

    def result(self) -> dict[str, float]:
        return {key: value / max(1, self.count) for key, value in self.sums.items()}


def _append_metric_event(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")
        handle.flush()


def _metric_console_line(event: str, stage: str, global_step: int, metrics: dict[str, float]) -> str:
    fields = [f"[{event}]", f"stage={stage}", f"step={global_step}"]
    for key in (
        "total",
        "current_patch",
        "current_global",
        "current_state",
        "standard_kl",
        "future_mean",
        "conditional_kl",
        "future_patch",
        "future_global",
        "spatial_total",
        "temporal_total",
    ):
        if key in metrics:
            fields.append(f"{key}={metrics[key]:.6f}")
    return " ".join(fields)


def _stages(config: dict[str, Any]) -> list[Stage]:
    result: list[Stage] = []
    for item in config["training"]["stages"]:
        result.append(
            Stage(
                name=str(item["name"]),
                epochs=1,
                learning_rate=float(item["learning_rate"]),
                stochastic=bool(item.get("stochastic", False)),
                temporal=bool(item.get("temporal", False)),
                kl_max=float(item.get("kl_max", 0.0)),
                kl_warmup_fraction=float(item.get("kl_warmup_fraction", 0.0)),
                fraction=float(item["fraction"]),
            )
        )
    return result


def allocate_stage_steps(total_steps: int, fractions: list[float]) -> list[int]:
    """Allocate one epoch's optimizer steps with largest-remainder rounding."""
    if total_steps <= 0:
        raise ValueError("The training split does not contain a complete batch")
    if not fractions or any(value <= 0 for value in fractions):
        raise ValueError("Every training stage fraction must be positive")
    fraction_sum = sum(fractions)
    if not math.isclose(fraction_sum, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"Training stage fractions must sum to 1.0, got {fraction_sum:.8f}")

    exact = [total_steps * value for value in fractions]
    allocated = [math.floor(value) for value in exact]
    remaining = total_steps - sum(allocated)
    order = sorted(
        range(len(fractions)),
        key=lambda index: (exact[index] - allocated[index], fractions[index]),
        reverse=True,
    )
    for index in order[:remaining]:
        allocated[index] += 1
    if any(value == 0 for value in allocated):
        raise ValueError(
            f"One epoch has only {total_steps} optimizer steps, which is insufficient "
            f"for {len(fractions)} positive training stages"
        )
    return allocated


def episode_aware_stage_indices(
    dataset: TemporalWindowDataset,
    stage_sample_counts: list[int],
    seed: int,
) -> list[list[int]]:
    """Partition one shuffled pass while keeping each stage's episode reads locally grouped."""
    usable_samples = sum(stage_sample_counts)
    if usable_samples > len(dataset):
        raise ValueError(
            f"Requested {usable_samples} scheduled samples from a dataset of length {len(dataset)}"
        )
    if not stage_sample_counts or any(value <= 0 for value in stage_sample_counts):
        raise ValueError("Every training stage must receive at least one sample")

    generator = torch.Generator().manual_seed(int(seed))
    episode_groups: list[list[int]] = [[] for _ in dataset.episodes]
    for window_index, (episode_position, _) in enumerate(dataset.windows):
        episode_groups[episode_position].append(window_index)

    labels = torch.cat(
        [
            torch.full((count,), stage_index, dtype=torch.int64)
            for stage_index, count in enumerate(stage_sample_counts)
        ]
    )
    labels = labels[torch.randperm(usable_samples, generator=generator)].tolist()
    episode_order = torch.randperm(len(episode_groups), generator=generator).tolist()
    result: list[list[int]] = [[] for _ in stage_sample_counts]
    label_position = 0
    for episode_position in episode_order:
        group = episode_groups[episode_position]
        if not group:
            continue
        local_order = torch.randperm(len(group), generator=generator).tolist()
        for local_position in local_order:
            if label_position >= usable_samples:
                break
            stage_index = labels[label_position]
            result[stage_index].append(group[local_position])
            label_position += 1
        if label_position >= usable_samples:
            break

    if [len(indices) for indices in result] != stage_sample_counts:
        raise RuntimeError("Episode-aware stage partition did not satisfy the requested sample counts")
    return result


def _configure_stage(model: VisualSplineVAE, stage: Stage) -> None:
    model.unfreeze_spatial_vae()
    if stage.temporal:
        model.temporal_prior.requires_grad_(True)
        model.temporal_prior.train()
    else:
        model.temporal_prior.requires_grad_(False)
        model.temporal_prior.eval()


def _reshape_future(value: Tensor, batch_size: int, horizon_count: int) -> Tensor:
    return value.reshape(batch_size, horizon_count, *value.shape[1:])


def calculate_batch_loss(
    model: VisualSplineVAE,
    batch: dict[str, Tensor],
    stage: Stage,
    config: dict[str, Any],
    kl_weight: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    weights = config["loss"]
    images = batch["images"]
    batch_size, image_count = images.shape[:2]
    horizon_count = image_count - 1
    flat_images = images.flatten(0, 1)

    dino = model.encode_dino(flat_images)
    grouped_patches = dino.patches.reshape(batch_size, image_count, *dino.patches.shape[1:])
    grouped_global = dino.global_token.reshape(batch_size, image_count, *dino.global_token.shape[1:])
    current_patches = grouped_patches[:, 0]
    current_global = grouped_global[:, 0]

    current_posterior = model.posterior_from_patches(current_patches)
    current_latent = current_posterior.sample(stage.stochastic)
    current_reconstruction = model.reconstruct(current_latent, current_patches.shape[1])
    patch_loss = feature_reconstruction_loss(
        current_reconstruction.patches, current_patches, float(weights["cosine_weight"])
    )
    global_loss = feature_reconstruction_loss(
        current_reconstruction.global_token, current_global, float(weights["cosine_weight"])
    )
    state_prediction = model.state_probe(current_posterior.mean)
    state_loss = F.huber_loss(state_prediction, batch["state"], delta=1.0)
    standard_kl = standard_normal_kl(current_posterior.mean, current_posterior.log_variance)
    variance_loss = variance_floor_loss(current_posterior.mean, float(weights["variance_floor"]))

    metrics: dict[str, Tensor] = {
        "current_patch": patch_loss.total,
        "current_global": global_loss.total,
        "current_state": state_loss,
        "standard_kl": standard_kl,
        "variance": variance_loss,
    }

    spatial_total = (
        float(weights["current_patch"]) * patch_loss.total
        + float(weights["current_global"]) * global_loss.total
        + float(weights["current_state"]) * state_loss
        + float(kl_weight) * standard_kl
        + float(weights["variance"]) * variance_loss
    )
    metrics["kl_weight"] = torch.as_tensor(kl_weight, device=spatial_total.device)
    metrics["spatial_total"] = spatial_total

    if not stage.temporal:
        metrics["total"] = spatial_total
        return spatial_total, metrics

    if horizon_count != len(model.horizons):
        raise ValueError(
            f"Temporal batch has {horizon_count} future images, expected {len(model.horizons)}"
        )
    future_patches = grouped_patches[:, 1:]
    future_global = grouped_global[:, 1:]

    with torch.no_grad():
        flat_future_patches = future_patches.flatten(0, 1)
        future_posterior_flat = model.posterior_from_patches(flat_future_patches)
        future_mean = _reshape_future(future_posterior_flat.mean, batch_size, horizon_count)
        future_log_variance = _reshape_future(
            future_posterior_flat.log_variance, batch_size, horizon_count
        )

    prior_mean, prior_log_variance = model.temporal_prior(
        current_posterior.mean.detach(), batch["state"], batch["actions"]
    )
    horizon_weights = normalized_horizon_weights(model.horizons, images.device)
    temporal_total = torch.zeros((), device=images.device)
    mean_total = torch.zeros_like(temporal_total)
    conditional_kl_total = torch.zeros_like(temporal_total)
    future_patch_total = torch.zeros_like(temporal_total)
    future_global_total = torch.zeros_like(temporal_total)

    for horizon_index, horizon_weight in enumerate(horizon_weights):
        mean_loss = mean_alignment_loss(
            prior_mean[:, horizon_index],
            future_mean[:, horizon_index],
            float(weights["cosine_weight"]),
        ).total
        conditional_kl = diagonal_gaussian_kl(
            future_mean[:, horizon_index],
            future_log_variance[:, horizon_index],
            prior_mean[:, horizon_index],
            prior_log_variance[:, horizon_index],
        )
        predicted_future = model.reconstruct(prior_mean[:, horizon_index], current_patches.shape[1])
        future_patch_loss = feature_reconstruction_loss(
            predicted_future.patches,
            future_patches[:, horizon_index],
            float(weights["cosine_weight"]),
        ).total
        future_global_loss = feature_reconstruction_loss(
            predicted_future.global_token,
            future_global[:, horizon_index],
            float(weights["cosine_weight"]),
        ).total

        horizon_loss = (
            float(weights["future_mean"]) * mean_loss
            + float(weights["conditional_kl"]) * conditional_kl
            + float(weights["future_patch"]) * future_patch_loss
            + float(weights["future_global"]) * future_global_loss
        )
        temporal_total = temporal_total + horizon_weight * horizon_loss
        mean_total = mean_total + horizon_weight * mean_loss
        conditional_kl_total = conditional_kl_total + horizon_weight * conditional_kl
        future_patch_total = future_patch_total + horizon_weight * future_patch_loss
        future_global_total = future_global_total + horizon_weight * future_global_loss

    joint_total = (
        float(weights.get("joint_spatial", 1.0)) * spatial_total
        + float(weights.get("joint_temporal", 1.0)) * temporal_total
    )
    metrics.update(
        {
            "future_mean": mean_total,
            "conditional_kl": conditional_kl_total,
            "future_patch": future_patch_total,
            "future_global": future_global_total,
            "temporal_total": temporal_total,
            "total": joint_total,
        }
    )
    return joint_total, metrics


def _run_epoch(
    model: VisualSplineVAE,
    loader: DataLoader,
    optimizer: AdamW | None,
    scaler: GradScaler,
    stage: Stage,
    config: dict[str, Any],
    device: torch.device,
    epoch: int,
    global_step: int,
    validation_frequency: int | None = None,
    validation_callback: Callable[[int], None] | None = None,
    log_callback: Callable[[int, dict[str, float]], None] | None = None,
) -> tuple[dict[str, float], int]:
    training = optimizer is not None
    if training:
        model.train()
        model.backbone.eval()
    else:
        model.eval()

    description = f"{stage.name}/{'train' if training else 'val'}/epoch-{epoch + 1}"
    progress = tqdm(loader, desc=description, unit="batch", leave=False)
    averages = MetricAverages()
    log_interval_averages = MetricAverages()
    total_stage_steps = max(1, stage.epochs * len(loader))
    warmup_steps = max(1, round(total_stage_steps * stage.kl_warmup_fraction))
    amp_enabled = bool(config["training"]["amp"]) and device.type == "cuda"
    log_frequency = int(config["training"].get("log_frequency", 1))

    for batch_index, batch in enumerate(progress):
        batch = move_to_device(batch, device)
        stage_step = epoch * len(loader) + batch_index
        kl_weight = stage.kl_max * min(1.0, (stage_step + 1) / warmup_steps) if stage.kl_max else 0.0
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with autocast(device_type=device.type, enabled=amp_enabled, dtype=torch.bfloat16):
                loss, metrics = calculate_batch_loss(model, batch, stage, config, kl_weight)
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad],
                    float(config["training"]["gradient_clip"]),
                )
                scaler.step(optimizer)
                scaler.update()
                global_step += 1
        averages.update(metrics)
        log_interval_averages.update(metrics)
        current = averages.result()
        completed_batches = batch_index + 1
        if not training or global_step % log_frequency == 0 or completed_batches == len(loader):
            progress.set_postfix(
                total=f"{current.get('total', math.nan):.4f}",
                patch=f"{current.get('current_patch', math.nan):.4f}",
                future=f"{current.get('future_patch', math.nan):.4f}",
            )
            if training and log_callback is not None:
                log_callback(global_step, log_interval_averages.result())
                log_interval_averages = MetricAverages()
        if (
            training
            and validation_callback is not None
            and validation_frequency is not None
            and global_step % validation_frequency == 0
        ):
            validation_callback(global_step)
            model.train()
            model.backbone.eval()
    return averages.result(), global_step


def _save_checkpoint(
    path: Path,
    model: VisualSplineVAE,
    optimizer: AdamW,
    config: dict[str, Any],
    state_normalization: dict[str, Any],
    split_indices: dict[str, list[int]],
    stage_index: int,
    epoch: int,
    global_step: int,
    metrics: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": {key: value for key, value in config.items() if not key.startswith("_")},
            "state_normalization": state_normalization,
            "split_indices": split_indices,
            "stage_index": stage_index,
            "epoch": epoch,
            "global_step": global_step,
            "metrics": metrics,
        },
        partial,
    )
    os.replace(partial, path)


def _episode_split_indices(splits: dict[str, list[Any]]) -> dict[str, list[int]]:
    return {
        key: [int(record.episode_index) for record in records]
        for key, records in splits.items()
    }


def _validate_initialization_split(
    checkpoint: dict[str, Any],
    current_split_indices: dict[str, list[int]],
) -> None:
    checkpoint_splits = checkpoint.get("split_indices")
    if not isinstance(checkpoint_splits, dict):
        raise ValueError("Initialization checkpoint does not contain dataset split indices")
    normalized_checkpoint_splits = {
        str(key): [int(value) for value in values]
        for key, values in checkpoint_splits.items()
    }
    if normalized_checkpoint_splits != current_split_indices:
        raise ValueError(
            "Initialization checkpoint dataset split does not match the current config. "
            "Use the same dataset root, train/validation ratios, and seed as the source run."
        )


def _checkpoint_state_normalization(checkpoint: dict[str, Any]) -> dict[str, Any]:
    normalization = checkpoint.get("state_normalization")
    if not isinstance(normalization, dict) or "mean" not in normalization or "std" not in normalization:
        raise ValueError("Initialization checkpoint does not contain valid state normalization")
    return normalization


def train(
    config: dict[str, Any],
    resume: str | None = None,
    init_checkpoint: str | None = None,
) -> Path:
    if resume and init_checkpoint:
        raise ValueError("--resume and --init-checkpoint are mutually exclusive")
    seed = int(config["training"]["seed"])
    seed_everything(seed)
    output_dir = Path(config["output"]["root"]).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    save_resolved_config(config, output_dir / "resolved_config.yaml")

    initialization_checkpoint: dict[str, Any] | None = None
    initialization_path: Path | None = None
    if init_checkpoint:
        initialization_path = Path(init_checkpoint).expanduser().resolve()
        initialization_checkpoint = torch.load(
            initialization_path,
            map_location="cpu",
            weights_only=False,
        )
        if not isinstance(initialization_checkpoint, dict) or "model" not in initialization_checkpoint:
            raise ValueError("Initialization checkpoint does not contain model weights")

    episodes = discover_episodes(config["dataset"]["root"])
    splits = split_episodes(
        episodes,
        seed,
        float(config["dataset"]["train_ratio"]),
        float(config["dataset"]["val_ratio"]),
    )
    save_splits(splits, output_dir / "episode_splits.json")
    split_indices = _episode_split_indices(splits)
    if initialization_checkpoint is not None:
        _validate_initialization_split(initialization_checkpoint, split_indices)
    normalization_path = output_dir / "state_normalization.json"
    if initialization_checkpoint is not None:
        state_normalization = _checkpoint_state_normalization(initialization_checkpoint)
        atomic_json_dump(state_normalization, normalization_path)
    elif normalization_path.is_file():
        state_normalization = json.loads(normalization_path.read_text(encoding="utf-8"))
    else:
        state_normalization = compute_state_normalization(splits["train"], normalization_path)

    dataset_arguments = dict(
        horizons=list(config["dataset"]["horizons"]),
        image_size=int(config["model"]["dino"]["image_size"]),
        state_mean=state_normalization["mean"],
        state_std=state_normalization["std"],
        cache_episodes=int(config["dataset"]["cache_episodes_per_worker"]),
    )
    spatial_train_dataset = TemporalWindowDataset(
        splits["train"],
        stride=int(config["dataset"]["window_stride"]),
        include_future_images=False,
        **dataset_arguments,
    )
    spatial_val_dataset = TemporalWindowDataset(
        splits["val"],
        stride=int(config["dataset"].get("val_window_stride", config["dataset"]["window_stride"])),
        include_future_images=False,
        **dataset_arguments,
    )
    temporal_train_dataset = TemporalWindowDataset(
        splits["train"],
        stride=int(config["dataset"]["window_stride"]),
        include_future_images=True,
        **dataset_arguments,
    )
    temporal_val_dataset = TemporalWindowDataset(
        splits["val"],
        stride=int(config["dataset"].get("val_window_stride", config["dataset"]["window_stride"])),
        include_future_images=True,
        **dataset_arguments,
    )
    loader_arguments = dict(
        batch_size=int(config["training"]["batch_size"]),
        num_workers=int(config["training"]["workers"]),
        pin_memory=bool(config["training"]["pin_memory"]),
        persistent_workers=bool(config["training"]["workers"]),
    )
    spatial_val_loader = DataLoader(
        spatial_val_dataset, shuffle=False, drop_last=False, **loader_arguments
    )
    temporal_val_loader = DataLoader(
        temporal_val_dataset, shuffle=False, drop_last=False, **loader_arguments
    )

    device = torch.device(config["training"]["device"] if torch.cuda.is_available() else "cpu")
    model = VisualSplineVAE(config).to(device)
    if initialization_checkpoint is not None:
        model.load_state_dict(initialization_checkpoint["model"], strict=True)
        del initialization_checkpoint
        tqdm.write(f"Initialized model weights from: {initialization_path}")
    stages = _stages(config)
    if len(spatial_train_dataset) != len(temporal_train_dataset):
        raise RuntimeError("Spatial and temporal training windows are not aligned")
    batch_size = int(config["training"]["batch_size"])
    total_train_steps = len(spatial_train_dataset) // batch_size
    stage_step_counts = allocate_stage_steps(
        total_train_steps, [stage.fraction for stage in stages]
    )
    usable_train_samples = total_train_steps * batch_size
    stage_sample_counts = [stage_steps * batch_size for stage_steps in stage_step_counts]
    partitioned_indices = episode_aware_stage_indices(
        spatial_train_dataset, stage_sample_counts, seed
    )
    stage_train_loaders: list[DataLoader] = []
    stage_schedule: list[dict[str, Any]] = []
    stage_loader_arguments = dict(loader_arguments)
    stage_loader_arguments["persistent_workers"] = False
    for stage, stage_steps, stage_indices in zip(
        stages, stage_step_counts, partitioned_indices
    ):
        stage_samples = stage_steps * batch_size
        stage_dataset = temporal_train_dataset if stage.temporal else spatial_train_dataset
        stage_train_loaders.append(
            DataLoader(
                Subset(stage_dataset, stage_indices),
                shuffle=False,
                drop_last=False,
                **stage_loader_arguments,
            )
        )
        stage_schedule.append(
            {
                "name": stage.name,
                "fraction": stage.fraction,
                "steps": stage_steps,
                "samples": stage_samples,
            }
        )
    atomic_json_dump(
        {
            "epochs": 1,
            "train_windows": len(spatial_train_dataset),
            "usable_train_windows": usable_train_samples,
            "dropped_train_windows": len(spatial_train_dataset) - usable_train_samples,
            "batch_size": batch_size,
            "total_steps": total_train_steps,
            "sampling": "episode_aware_disjoint_stage_partition",
            "validation_windows": len(spatial_val_dataset),
            "validation_stride": int(
                config["dataset"].get("val_window_stride", config["dataset"]["window_stride"])
            ),
            "stages": stage_schedule,
        },
        output_dir / "training_schedule.json",
    )
    resume_stage = 0
    resume_epoch = -1
    resume_optimizer: dict[str, Any] | None = None
    global_step = 0
    if resume:
        checkpoint = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        resume_stage = int(checkpoint["stage_index"])
        resume_epoch = int(checkpoint["epoch"])
        global_step = int(checkpoint.get("global_step", 0))
        resume_optimizer = checkpoint.get("optimizer")

    history: list[dict[str, Any]] = []
    step_validations: list[dict[str, Any]] = []
    best_validation = math.inf
    final_checkpoint = output_dir / "checkpoints" / "final.pt"
    scaler = GradScaler(device.type, enabled=bool(config["training"]["amp"]) and device.type == "cuda")
    log_frequency = int(config["training"].get("log_frequency", 1))
    validation_frequency = int(config["training"].get("val_frequency", 1))
    if log_frequency <= 0:
        raise ValueError("training.log_frequency must be a positive number of optimizer steps")
    if validation_frequency <= 0:
        raise ValueError("training.val_frequency must be a positive number of optimizer steps")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    metric_log_path = output_dir / "training_metrics.jsonl"
    _append_metric_event(
        metric_log_path,
        {
            "event": "run_start",
            "run_id": run_id,
            "global_step": global_step,
            "config": str(config.get("_config_path", "")),
            "resume": str(Path(resume).expanduser().resolve()) if resume else None,
            "init_checkpoint": str(initialization_path) if initialization_path else None,
        },
    )
    tqdm.write(f"Training metrics log: {metric_log_path}")

    for stage_index, stage in enumerate(tqdm(stages, desc="training/stages", unit="stage")):
        if stage_index < resume_stage:
            continue
        _configure_stage(model, stage)
        train_loader = stage_train_loaders[stage_index]
        val_loader = temporal_val_loader if stage.temporal else spatial_val_loader
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = AdamW(
            parameters,
            lr=stage.learning_rate,
            weight_decay=float(config["training"]["weight_decay"]),
        )
        first_epoch = resume_epoch + 1 if stage_index == resume_stage else 0
        if stage_index == resume_stage and first_epoch < stage.epochs and resume_optimizer:
            optimizer.load_state_dict(resume_optimizer)

        for epoch in tqdm(
            range(first_epoch, stage.epochs),
            desc=f"{stage.name}/epochs",
            unit="epoch",
            leave=False,
        ):
            latest_step_validation: tuple[int, dict[str, float]] | None = None

            def log_training_step(step: int, metrics: dict[str, float]) -> None:
                event = {
                    "event": "train",
                    "run_id": run_id,
                    "stage": stage.name,
                    "stage_index": stage_index,
                    "epoch": epoch,
                    "global_step": step,
                    "metrics": metrics,
                }
                _append_metric_event(metric_log_path, event)
                tqdm.write(_metric_console_line("train", stage.name, step, metrics))

            def validate_step(step: int) -> None:
                nonlocal latest_step_validation
                with torch.no_grad():
                    metrics, _ = _run_epoch(
                        model, val_loader, None, scaler, stage, config, device, epoch, step
                    )
                latest_step_validation = (step, metrics)
                _append_metric_event(
                    metric_log_path,
                    {
                        "event": "validation",
                        "trigger": "step_frequency",
                        "run_id": run_id,
                        "stage": stage.name,
                        "stage_index": stage_index,
                        "epoch": epoch,
                        "global_step": step,
                        "metrics": metrics,
                    },
                )
                tqdm.write(_metric_console_line("validation", stage.name, step, metrics))
                step_validations.append(
                    {
                        "stage": stage.name,
                        "stage_index": stage_index,
                        "epoch": epoch,
                        "global_step": step,
                        "validation": metrics,
                    }
                )
                atomic_json_dump(
                    {"epochs": history, "step_validations": step_validations},
                    output_dir / "metrics.json",
                )

            train_metrics, global_step = _run_epoch(
                model,
                train_loader,
                optimizer,
                scaler,
                stage,
                config,
                device,
                epoch,
                global_step,
                validation_frequency=validation_frequency,
                validation_callback=validate_step,
                log_callback=log_training_step,
            )
            if latest_step_validation is not None and latest_step_validation[0] == global_step:
                val_metrics = latest_step_validation[1]
            else:
                with torch.no_grad():
                    val_metrics, _ = _run_epoch(
                        model, val_loader, None, scaler, stage, config, device, epoch, global_step
                    )
                _append_metric_event(
                    metric_log_path,
                    {
                        "event": "validation",
                        "trigger": "stage_end",
                        "run_id": run_id,
                        "stage": stage.name,
                        "stage_index": stage_index,
                        "epoch": epoch,
                        "global_step": global_step,
                        "metrics": val_metrics,
                    },
                )
                tqdm.write(_metric_console_line("validation", stage.name, global_step, val_metrics))
            record = {
                "stage": stage.name,
                "stage_index": stage_index,
                "epoch": epoch,
                "global_step": global_step,
                "train": train_metrics,
                "validation": val_metrics,
            }
            history.append(record)
            atomic_json_dump(
                {"epochs": history, "step_validations": step_validations},
                output_dir / "metrics.json",
            )
            checkpoint_path = output_dir / "checkpoints" / f"{stage.name}_epoch_{epoch + 1:03d}.pt"
            _save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                config,
                state_normalization,
                split_indices,
                stage_index,
                epoch,
                global_step,
                val_metrics,
            )
            if val_metrics["total"] < best_validation:
                best_validation = val_metrics["total"]
                _save_checkpoint(
                    output_dir / "checkpoints" / "best.pt",
                    model,
                    optimizer,
                    config,
                    state_normalization,
                    split_indices,
                    stage_index,
                    epoch,
                    global_step,
                    val_metrics,
                )

        resume_epoch = -1
        resume_optimizer = None

    if history:
        last = history[-1]
        _save_checkpoint(
            final_checkpoint,
            model,
            optimizer,
            config,
            state_normalization,
            split_indices,
            int(last["stage_index"]),
            int(last["epoch"]),
            int(last["global_step"]),
            last["validation"],
        )
    return final_checkpoint
