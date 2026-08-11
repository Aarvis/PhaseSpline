from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .config import save_resolved_config
from .data import (
    LocalizerDataset,
    as_path,
    checkpoint_json_path,
    collect_human_episode_ids_from_pairings,
    compute_state_normalization,
    discover_valid_robot_episodes,
    load_annotation_episode,
    load_category_configs,
    localizer_collate,
    load_robot_target_cache,
    prepare_human_spline_cache,
    prepare_robot_alignment_target_cache,
    read_dataset_info,
    resolve_state_dims,
    robot_target_cache_npz_path,
    split_robot_episodes_by_category,
)
from .model import GlobalSplineLocalizer
from .utils import MetricAverages, atomic_json_dump, seed_everything


LOGGER = logging.getLogger("human_spline_localizer.training")

MODEL_INPUT_KEYS = (
    "robot_history_embeddings",
    "robot_history_states",
    "robot_history_mask",
    "human_coefficients",
    "human_left_support",
    "human_right_support",
    "human_support_midpoint",
    "human_support_width",
    "human_greville_phase",
    "human_basis_200",
    "human_mask",
    "category_index",
)


def _move_to_device(batch: dict[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}


def _format_metric_summary(metrics: dict[str, float]) -> str:
    ordered_keys = (
        "total",
        "loc_loss",
        "checkpoint_loss",
        "progress_loss",
        "checkpoint_accuracy",
        "phase_mae",
        "c_max",
        "entropy",
        "margin",
    )
    parts: list[str] = []
    for key in ordered_keys:
        if key in metrics:
            parts.append(f"{key}={float(metrics[key]):.4f}")
    for key, value in metrics.items():
        if key not in ordered_keys:
            parts.append(f"{key}={float(value):.4f}")
    return " ".join(parts)


def _model_inputs(batch: dict[str, Tensor]) -> dict[str, Tensor]:
    missing = [key for key in MODEL_INPUT_KEYS if key not in batch]
    if missing:
        raise KeyError(f"Batch is missing required model inputs: {missing}")
    return {key: batch[key] for key in MODEL_INPUT_KEYS}


def _paths(config: dict[str, Any]) -> dict[str, Path]:
    output_root = as_path(config["paths"]["output_root"])
    preprocessing = config["preprocessing"]
    return {
        "sim_dataset_root": as_path(config["paths"]["sim_dataset_root"]),
        "human_dataset_root": as_path(config["paths"]["human_dataset_root"]),
        "robot_embedding_root": as_path(config["paths"]["robot_embedding_root"]),
        "robot_pairing_root": as_path(config["paths"]["robot_pairing_root"]),
        "human_bspline_root": as_path(config["paths"]["human_bspline_root"]),
        "output_root": output_root,
        "human_cache_root": output_root / str(preprocessing["human_cache_dirname"]),
        "robot_target_cache_root": output_root / str(preprocessing["robot_target_cache_dirname"]),
        "checkpoints_root": output_root / "checkpoints",
        "logs_root": output_root / "logs",
        "splits_path": output_root / str(preprocessing["splits_filename"]),
        "state_norm_path": output_root / str(preprocessing["state_norm_filename"]),
        "category_metadata_path": output_root / str(preprocessing["category_metadata_filename"]),
        "resolved_config_path": output_root / "resolved_config.yaml",
    }


def _count_checkpoint_classes(
    sim_dataset_root: Path,
    categories: list,
    split_map: dict[str, dict[str, list[int]]],
) -> tuple[list[int], dict[str, list[str]]]:
    checkpoint_counts: list[int] = []
    label_names: dict[str, list[str]] = {}
    for category in categories:
        candidate_episodes = split_map[category.category_id]["train"] + split_map[category.category_id]["val"]
        if not candidate_episodes:
            raise RuntimeError(f"No robot episodes available for selected category {category.category_id}")
        annotation = load_annotation_episode(checkpoint_json_path(sim_dataset_root, candidate_episodes[0]), candidate_episodes[0])
        count = max(len(annotation.labels), int(annotation.frame_to_segment_id.max()) + 1)
        checkpoint_counts.append(count)
        label_names[category.category_id] = annotation.labels
    return checkpoint_counts, label_names


def _state_norm_payload(config: dict[str, Any], paths: dict[str, Path], train_episode_ids: list[int]) -> dict[str, Any]:
    if paths["state_norm_path"].exists():
        return json.loads(paths["state_norm_path"].read_text(encoding="utf-8"))
    info = read_dataset_info(paths["sim_dataset_root"])
    state_dims = resolve_state_dims(config, info)
    return compute_state_normalization(paths["sim_dataset_root"], train_episode_ids, state_dims, paths["state_norm_path"])


def _count_samples_for_slot(robot_episode_ids: list[int], target_cache_root: Path, slot: int) -> int:
    total = 0
    for episode_index in robot_episode_ids:
        cache = load_robot_target_cache(robot_target_cache_npz_path(target_cache_root, episode_index))
        total += int(cache.target_valid_mask[:, int(slot)].sum())
    return total


def _make_scheduler(optimizer: AdamW, config: dict[str, Any], total_optimizer_steps: int) -> LambdaLR:
    training_cfg = config["training"]
    scheduler_name = str(training_cfg.get("scheduler", "cosine")).lower()
    warmup_steps = int(training_cfg.get("warmup_steps", 0))
    if scheduler_name == "none":
        return LambdaLR(optimizer, lr_lambda=lambda _step: 1.0)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        if scheduler_name == "cosine":
            progress = (step - warmup_steps) / float(max(1, total_optimizer_steps - warmup_steps))
            progress = min(max(progress, 0.0), 1.0)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def _maybe_init_wandb(config: dict[str, Any], output_root: Path):
    wandb_cfg = config.get("logging", {}).get("wandb", {})
    if not bool(wandb_cfg.get("enabled", False)):
        return None
    token = wandb_cfg.get("api_key")
    if token:
        os.environ["WANDB_API_KEY"] = str(token)
    try:
        import wandb
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Weights & Biases logging was requested but wandb is not installed.") from exc
    run = wandb.init(
        project=str(wandb_cfg.get("project", "human-spline-localizer")),
        entity=wandb_cfg.get("entity"),
        name=wandb_cfg.get("run_name"),
        dir=str(output_root),
        config=config,
    )
    return run


def _save_checkpoint(
    path: Path,
    model: GlobalSplineLocalizer,
    optimizer: AdamW,
    scheduler: LambdaLR,
    scaler: GradScaler,
    epoch_index: int,
    global_step: int,
    best_metric: float,
    config: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch_index": int(epoch_index),
            "global_step": int(global_step),
            "best_metric": float(best_metric),
            "config": config,
        },
        path,
    )


def _load_checkpoint(path: Path, model: GlobalSplineLocalizer, optimizer: AdamW, scheduler: LambdaLR, scaler: GradScaler) -> tuple[int, int, float]:
    payload = torch.load(path, map_location="cpu")
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    scaler.load_state_dict(payload["scaler"])
    return int(payload["epoch_index"]), int(payload["global_step"]), float(payload["best_metric"])


def _compute_losses(
    outputs: dict[str, Tensor],
    batch: dict[str, Tensor],
    checkpoint_class_counts: list[int],
    config: dict[str, Any],
) -> tuple[Tensor, dict[str, Tensor]]:
    loc_loss = -(batch["soft_target"] * torch.log_softmax(outputs["logits"], dim=-1)).sum(dim=-1).mean()
    checkpoint_loss_sum = outputs["logits"].new_zeros(())
    checkpoint_sample_count = 0
    checkpoint_correct = 0
    for category_index, class_count in enumerate(checkpoint_class_counts):
        mask = batch["category_index"] == category_index
        if not torch.any(mask):
            continue
        logits = outputs["checkpoint_logits"][mask, :class_count]
        targets = batch["checkpoint_target"][mask]
        checkpoint_loss_sum = checkpoint_loss_sum + F.cross_entropy(logits, targets, reduction="sum")
        checkpoint_sample_count += int(targets.numel())
        checkpoint_correct += int((logits.argmax(dim=-1) == targets).sum().item())
    checkpoint_loss = checkpoint_loss_sum / max(1, checkpoint_sample_count)
    progress_loss = F.huber_loss(
        outputs["progress"],
        batch["progress_target"],
        delta=float(config["loss"]["progress_delta"]),
    )
    total = (
        loc_loss
        + float(config["loss"]["lambda_checkpoint"]) * checkpoint_loss
        + float(config["loss"]["lambda_progress"]) * progress_loss
    )
    phase_mae = torch.abs(outputs["u_hat"] - batch["target_u"]).mean()
    metrics = {
        "total": total.detach(),
        "loc_loss": loc_loss.detach(),
        "checkpoint_loss": checkpoint_loss.detach(),
        "progress_loss": progress_loss.detach(),
        "checkpoint_accuracy": outputs["logits"].new_tensor(checkpoint_correct / max(1, checkpoint_sample_count)),
        "phase_mae": phase_mae.detach(),
        "c_max": outputs["c_max"].mean().detach(),
        "entropy": outputs["entropy"].mean().detach(),
        "margin": outputs["margin"].mean().detach(),
    }
    return total, metrics


def _evaluate(
    model: GlobalSplineLocalizer,
    dataloader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    checkpoint_class_counts: list[int],
    config: dict[str, Any],
    log_frequency_steps: int,
) -> dict[str, float]:
    model.eval()
    averages = MetricAverages()
    with torch.no_grad():
        iterator = tqdm(dataloader, desc="validation", unit="batch", leave=False)
        for batch_index, batch in enumerate(iterator, start=1):
            batch = _move_to_device(batch, device)
            with autocast(device_type=device.type, enabled=amp_enabled):
                outputs = model(**_model_inputs(batch))
                _, metrics = _compute_losses(outputs, batch, checkpoint_class_counts, config)
            averages.update(metrics)
            if batch_index % max(1, log_frequency_steps) == 0:
                current = averages.result()
                iterator.set_postfix(val_total=f"{current.get('total', float('nan')):.4f}", val_mae=f"{current.get('phase_mae', float('nan')):.4f}")
    return averages.result()


def train(config: dict[str, Any], resume: str | None = None) -> Path:
    seed_everything(int(config["training"]["seed"]))
    paths = _paths(config)
    for path in (paths["output_root"], paths["checkpoints_root"], paths["logs_root"]):
        path.mkdir(parents=True, exist_ok=True)
    save_resolved_config(config, paths["resolved_config_path"])

    categories = load_category_configs(config)
    category_to_index = {category.category_id: index for index, category in enumerate(categories)}
    valid_robot_episodes = discover_valid_robot_episodes(
        paths["sim_dataset_root"],
        paths["robot_embedding_root"],
        paths["robot_pairing_root"],
        categories,
        skip_missing=bool(config["data"]["skip_missing_robot_episodes"]),
    )
    split_map = split_robot_episodes_by_category(
        valid_robot_episodes,
        split_seed=int(config["validation"]["split_seed"]),
        per_category_ratio=float(config["validation"]["per_category_ratio"]),
        min_episodes_per_category=int(config["validation"]["min_episodes_per_category"]),
    )
    atomic_json_dump(split_map, paths["splits_path"])

    train_episode_ids = [episode for category in categories for episode in split_map[category.category_id]["train"]]
    val_episode_ids = [episode for category in categories for episode in split_map[category.category_id]["val"]]
    human_episode_ids = collect_human_episode_ids_from_pairings(train_episode_ids + val_episode_ids, paths["robot_pairing_root"])
    prepare_human_spline_cache(
        human_episode_ids,
        paths["human_bspline_root"],
        paths["human_cache_root"],
        phase_bin_count=int(config["data"]["phase_bin_count"]),
        overwrite=bool(config["preprocessing"]["overwrite"]),
    )
    prepare_robot_alignment_target_cache(
        train_episode_ids + val_episode_ids,
        paths["sim_dataset_root"],
        paths["human_dataset_root"],
        paths["robot_pairing_root"],
        paths["human_bspline_root"],
        paths["robot_target_cache_root"],
        overwrite=bool(config["preprocessing"]["overwrite"]),
    )
    state_norm = _state_norm_payload(config, paths, train_episode_ids)
    checkpoint_class_counts, label_names = _count_checkpoint_classes(paths["sim_dataset_root"], categories, split_map)
    category_metadata = {
        "category_order": [category.category_id for category in categories],
        "checkpoint_class_counts": checkpoint_class_counts,
        "label_names": label_names,
    }
    atomic_json_dump(category_metadata, paths["category_metadata_path"])

    model = GlobalSplineLocalizer(
        config=config,
        state_dim=len(state_norm["state_dims"]),
        checkpoint_class_counts=checkpoint_class_counts,
    )
    device = torch.device(str(config["training"]["device"]))
    model.to(device)

    pairing_slot_schedule = [int(value) for value in config["training"]["pairing_slot_schedule"]]
    if int(config["training"]["epochs"]) != len(pairing_slot_schedule):
        raise ValueError("training.epochs must match the length of training.pairing_slot_schedule")
    batch_size = int(config["training"]["batch_size"])
    grad_accum_steps = max(1, int(config["training"]["gradient_accumulation_steps"]))
    total_optimizer_steps = 0
    for slot in pairing_slot_schedule:
        slot_samples = _count_samples_for_slot(train_episode_ids, paths["robot_target_cache_root"], slot)
        total_optimizer_steps += math.ceil(max(1, slot_samples) / batch_size / grad_accum_steps)
    optimizer = AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
        betas=(float(config["training"]["adam_beta1"]), float(config["training"]["adam_beta2"])),
    )
    scheduler = _make_scheduler(optimizer, config, total_optimizer_steps)
    amp_enabled = bool(config["training"]["amp"]) and device.type == "cuda"
    scaler = GradScaler(enabled=amp_enabled)
    wandb_run = _maybe_init_wandb(config, paths["output_root"])

    start_epoch = 0
    global_step = 0
    best_metric = float("inf")
    if resume is not None:
        start_epoch, global_step, best_metric = _load_checkpoint(Path(resume), model, optimizer, scheduler, scaler)
        start_epoch += 1

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": int(config["training"]["num_workers"]),
        "pin_memory": bool(config["training"]["pin_memory"]) and device.type == "cuda",
        "persistent_workers": bool(config["training"]["persistent_workers"]) and int(config["training"]["num_workers"]) > 0,
        "collate_fn": localizer_collate,
    }

    for epoch_index in range(start_epoch, int(config["training"]["epochs"])):
        pairing_slot = pairing_slot_schedule[epoch_index]
        train_dataset = LocalizerDataset(
            train_episode_ids,
            pairing_slot=pairing_slot,
            paths=paths,
            state_dims=state_norm["state_dims"],
            state_mean=np.asarray(state_norm["state_mean"], dtype=np.float32),
            state_std=np.asarray(state_norm["state_std"], dtype=np.float32),
            category_to_index=category_to_index,
            categories=categories,
            config=config,
        )
        val_dataset = LocalizerDataset(
            val_episode_ids,
            pairing_slot=pairing_slot,
            paths=paths,
            state_dims=state_norm["state_dims"],
            state_mean=np.asarray(state_norm["state_mean"], dtype=np.float32),
            state_std=np.asarray(state_norm["state_std"], dtype=np.float32),
            category_to_index=category_to_index,
            categories=categories,
            config=config,
        )
        train_loader = DataLoader(train_dataset, shuffle=True, drop_last=False, **loader_kwargs)
        val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False, **loader_kwargs) if len(val_dataset) > 0 else None

        model.train()
        averages = MetricAverages()
        optimizer.zero_grad(set_to_none=True)
        iterator = tqdm(train_loader, desc=f"epoch {epoch_index + 1}/{config['training']['epochs']} slot={pairing_slot}", unit="batch")
        for batch_index, batch in enumerate(iterator, start=1):
            batch = _move_to_device(batch, device)
            with autocast(device_type=device.type, enabled=amp_enabled):
                outputs = model(**_model_inputs(batch))
                loss, metrics = _compute_losses(outputs, batch, checkpoint_class_counts, config)
                loss = loss / grad_accum_steps
            scaler.scale(loss).backward()
            averages.update(metrics)
            if batch_index % grad_accum_steps == 0 or batch_index == len(train_loader):
                if float(config["training"]["gradient_clip_norm"]) > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"]["gradient_clip_norm"]))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

                if global_step % max(1, int(config["training"]["train_log_frequency_steps"])) == 0:
                    current = averages.result()
                    log_payload = {f"train/{key}": value for key, value in current.items()}
                    log_payload["train/epoch"] = float(epoch_index + 1)
                    log_payload["train/pairing_slot"] = float(pairing_slot)
                    log_payload["train/lr"] = float(optimizer.param_groups[0]["lr"])
                    if wandb_run is not None:
                        wandb_run.log(log_payload, step=global_step)
                    tqdm.write(
                        f"[train] epoch={epoch_index + 1}/{config['training']['epochs']} "
                        f"slot={pairing_slot} step={global_step} lr={float(optimizer.param_groups[0]['lr']):.6e} "
                        f"{_format_metric_summary(current)}"
                    )
                    iterator.set_postfix(
                        total=f"{current.get('total', float('nan')):.4f}",
                        mae=f"{current.get('phase_mae', float('nan')):.4f}",
                        cp=f"{current.get('checkpoint_accuracy', float('nan')):.4f}",
                    )

                if (
                    val_loader is not None
                    and int(config["training"]["validation_frequency_steps"]) > 0
                    and global_step % int(config["training"]["validation_frequency_steps"]) == 0
                ):
                    validation_metrics = _evaluate(
                        model,
                        val_loader,
                        device,
                        amp_enabled,
                        checkpoint_class_counts,
                        config,
                        log_frequency_steps=int(config["training"]["validation_log_frequency_steps"]),
                    )
                    if wandb_run is not None:
                        wandb_run.log({f"val/{key}": value for key, value in validation_metrics.items()}, step=global_step)
                    tqdm.write(
                        f"[val] epoch={epoch_index + 1}/{config['training']['epochs']} "
                        f"slot={pairing_slot} step={global_step} "
                        f"{_format_metric_summary(validation_metrics)}"
                    )
                    metric_name = str(config["training"]["checkpoint_metric"]).replace("val_", "")
                    monitored = float(validation_metrics.get(metric_name, float("inf")))
                    if monitored < best_metric:
                        best_metric = monitored
                        _save_checkpoint(paths["checkpoints_root"] / "best.pt", model, optimizer, scheduler, scaler, epoch_index, global_step, best_metric, config)
                        tqdm.write(
                            f"[checkpoint] saved best checkpoint at step={global_step} "
                            f"metric={metric_name} value={best_metric:.4f}"
                        )
                    model.train()

        if val_loader is not None:
            validation_metrics = _evaluate(
                model,
                val_loader,
                device,
                amp_enabled,
                checkpoint_class_counts,
                config,
                log_frequency_steps=int(config["training"]["validation_log_frequency_steps"]),
            )
            if wandb_run is not None:
                wandb_run.log({f"val_epoch/{key}": value for key, value in validation_metrics.items()}, step=global_step)
            tqdm.write(
                f"[val-epoch] epoch={epoch_index + 1}/{config['training']['epochs']} "
                f"slot={pairing_slot} step={global_step} "
                f"{_format_metric_summary(validation_metrics)}"
            )
            metric_name = str(config["training"]["checkpoint_metric"]).replace("val_", "")
            monitored = float(validation_metrics.get(metric_name, float("inf")))
            if monitored < best_metric:
                best_metric = monitored
                _save_checkpoint(paths["checkpoints_root"] / "best.pt", model, optimizer, scheduler, scaler, epoch_index, global_step, best_metric, config)
                tqdm.write(
                    f"[checkpoint] saved best checkpoint at epoch_end step={global_step} "
                    f"metric={metric_name} value={best_metric:.4f}"
                )
            model.train()
        _save_checkpoint(paths["checkpoints_root"] / "last.pt", model, optimizer, scheduler, scaler, epoch_index, global_step, best_metric, config)
        tqdm.write(
            f"[epoch] completed epoch={epoch_index + 1}/{config['training']['epochs']} "
            f"slot={pairing_slot} global_step={global_step} best_{str(config['training']['checkpoint_metric']).replace('val_', '')}={best_metric:.4f}"
        )
    _save_checkpoint(paths["checkpoints_root"] / "final.pt", model, optimizer, scheduler, scaler, int(config["training"]["epochs"]) - 1, global_step, best_metric, config)
    if wandb_run is not None:
        wandb_run.finish()
    return paths["checkpoints_root"] / "final.pt"
