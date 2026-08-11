from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .config import save_resolved_config
from .data import (
    TranslatorDataset,
    as_path,
    collect_human_episode_ids_from_pairings,
    compute_state_normalization,
    discover_valid_robot_episodes,
    exact_human_interval_npz_path,
    load_category_configs,
    load_exact_human_interval_cache,
    precompute_exact_human_interval_cache,
    read_dataset_info,
    resolve_state_dims,
    robot_local_window_npz_path,
    split_robot_episodes_by_category,
    translator_collate,
)
from .losses import compute_losses
from .model import LocalHumanToRobotSplineModel
from .schedule import PiecewiseLinearSchedule
from .spline_math import derive_gt_span_widths_batch, evaluate_global_spline_interval_batch
from .utils import MetricAverages, atomic_json_dump, seed_everything


def _move_to_device(batch: dict[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}


def _constant_schedule(value: float = 1.0) -> PiecewiseLinearSchedule:
    scalar = float(value)
    return PiecewiseLinearSchedule(milestones=(0.0, 1.0), values=(scalar, scalar))


def _optional_schedule(payload: dict[str, Any], key: str, *, default_value: float = 1.0) -> PiecewiseLinearSchedule:
    schedule_payload = payload.get(key)
    if schedule_payload is None:
        return _constant_schedule(default_value)
    return PiecewiseLinearSchedule.from_config(schedule_payload)


def _build_derivative_weight_schedules(config: dict[str, Any]) -> dict[str, PiecewiseLinearSchedule]:
    dense_cfg = config["loss"]["dense"]
    spline_cfg = config["loss"]["spline"]
    return {
        "dense_velocity_scale": _optional_schedule(dense_cfg, "velocity_weight_scale_schedule"),
        "dense_acceleration_scale": _optional_schedule(dense_cfg, "acceleration_weight_scale_schedule"),
        "spline_velocity_scale": _optional_schedule(spline_cfg, "velocity_weight_scale_schedule"),
        "spline_acceleration_scale": _optional_schedule(spline_cfg, "acceleration_weight_scale_schedule"),
    }


def _effective_derivative_weights(
    config: dict[str, Any],
    schedules: dict[str, PiecewiseLinearSchedule],
    progress: float,
) -> dict[str, float]:
    dense_cfg = config["loss"]["dense"]
    spline_cfg = config["loss"]["spline"]
    return {
        "dense_velocity": float(dense_cfg["lambda_velocity"]) * schedules["dense_velocity_scale"].value(progress),
        "dense_acceleration": float(dense_cfg["lambda_acceleration"]) * schedules["dense_acceleration_scale"].value(progress),
        "spline_velocity": float(spline_cfg["lambda_velocity"]) * schedules["spline_velocity_scale"].value(progress),
        "spline_acceleration": float(spline_cfg["lambda_acceleration"]) * schedules["spline_acceleration_scale"].value(progress),
    }


def _format_metric_summary(metrics: dict[str, float]) -> str:
    ordered_keys = (
        "total",
        "dense_total",
        "spline_total",
        "dense_position_rmse",
        "dense_velocity_rmse",
        "dense_acceleration_rmse",
        "spline_position_rmse",
        "spline_velocity_rmse",
        "spline_acceleration_rmse",
        "anchor_rmse",
        "knot_loss",
        "span_min",
        "span_max",
    )
    parts: list[str] = []
    for key in ordered_keys:
        if key in metrics:
            parts.append(f"{key}={float(metrics[key]):.4f}")
    for key, value in metrics.items():
        if key not in ordered_keys:
            parts.append(f"{key}={float(value):.4f}")
    return " ".join(parts)


def _paths(config: dict[str, Any]) -> dict[str, Path]:
    output_root = as_path(config["paths"]["output_root"])
    sim_embedding_root = as_path(config["paths"]["sim_embedding_root"])
    preprocessing = config["preprocessing"]
    predicted_u_root_raw = config["paths"].get("predicted_human_u_root")
    predicted_u_root = as_path(predicted_u_root_raw) if predicted_u_root_raw else None
    return {
        "sim_dataset_root": as_path(config["paths"]["sim_dataset_root"]),
        "sim_embedding_root": sim_embedding_root,
        "human_dataset_root": as_path(config["paths"]["human_dataset_root"]),
        "human_bspline_root": as_path(config["paths"]["human_bspline_root"]),
        "robot_embedding_root": sim_embedding_root / "frame_embeddings",
        "robot_spline_root": sim_embedding_root / "fitted_bspline_splines",
        "robot_local_window_root": sim_embedding_root / str(config["paths"]["robot_local_window_dirname"]),
        "robot_pairing_root": sim_embedding_root / str(config["paths"]["robot_pairing_dirname"]),
        "predicted_human_u_root": predicted_u_root,
        "output_root": output_root,
        "exact_human_interval_cache_root": output_root / str(preprocessing["exact_human_interval_cache_dirname"]),
        "checkpoints_root": output_root / "checkpoints",
        "logs_root": output_root / "logs",
        "splits_path": output_root / str(preprocessing["splits_filename"]),
        "state_norm_path": output_root / str(preprocessing["state_norm_filename"]),
        "category_metadata_path": output_root / str(preprocessing["category_metadata_filename"]),
        "resolved_config_path": output_root / "resolved_config.yaml",
        "preprocessing_summary_path": output_root / "preprocessing_summary.json",
    }


def _state_norm_payload(config: dict[str, Any], paths: dict[str, Path], train_episode_ids: list[int]) -> dict[str, Any]:
    if paths["state_norm_path"].exists():
        return json.loads(paths["state_norm_path"].read_text(encoding="utf-8"))
    info = read_dataset_info(paths["sim_dataset_root"])
    state_dims = resolve_state_dims(config, info)
    return compute_state_normalization(paths["sim_dataset_root"], train_episode_ids, state_dims, paths["state_norm_path"])


def _count_valid_samples_for_slot(robot_episode_ids: list[int], cache_root: Path, robot_local_window_root: Path, slot: int) -> int:
    total = 0
    for episode_index in robot_episode_ids:
        interval_cache = load_exact_human_interval_cache(exact_human_interval_npz_path(cache_root, episode_index))
        with np.load(robot_local_window_npz_path(robot_local_window_root, episode_index), allow_pickle=False) as archive:
            local_valid = np.asarray(archive["exact_local_spline_valid"], dtype=bool)
        total += int(np.count_nonzero(interval_cache.interval_valid_mask[:, int(slot)] & local_valid))
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
        project=str(wandb_cfg.get("project", "human-to-robot-local-spline-translator")),
        entity=wandb_cfg.get("entity"),
        name=wandb_cfg.get("run_name"),
        dir=str(output_root),
        config=config,
    )
    return run


def _save_checkpoint(
    path: Path,
    model: LocalHumanToRobotSplineModel,
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


def _load_checkpoint(path: Path, model: LocalHumanToRobotSplineModel, optimizer: AdamW, scheduler: LambdaLR, scaler: GradScaler) -> tuple[int, int, float]:
    payload = torch.load(path, map_location="cpu")
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    scaler.load_state_dict(payload["scaler"])
    return int(payload["epoch_index"]), int(payload["global_step"]), float(payload["best_metric"])


def _build_robot_targets(batch: dict[str, Tensor], model: LocalHumanToRobotSplineModel, config: dict[str, Any]) -> dict[str, Tensor]:
    dense_position, dense_velocity, dense_acceleration = evaluate_global_spline_interval_batch(
        global_coefficients=batch["robot_global_coefficients"],
        global_knots=batch["robot_global_knots"],
        coeff_counts=batch["robot_global_coeff_counts"],
        knot_counts=batch["robot_global_knot_counts"],
        interval_start_u=batch["robot_gt_start_u"],
        interval_end_u=batch["robot_gt_end_u"],
        phase_grid=model.phase_grid,
        degree=model.degree,
    )
    gt_span_widths, gt_span_valid_mask = derive_gt_span_widths_batch(
        batch["robot_exact_local_knot_local_u"],
        batch["robot_exact_local_num_knots"],
        expected_num_spans=int(config["model"]["span_decoder"]["num_output_spans"]),
    )
    return {
        "dense_position": dense_position,
        "dense_velocity": dense_velocity,
        "dense_acceleration": dense_acceleration,
        "gt_span_widths": gt_span_widths,
        "gt_span_valid_mask": gt_span_valid_mask,
    }


def _select_human_input_interval(
    batch: dict[str, Tensor],
    predicted_u_alpha: float,
    min_width: float,
) -> tuple[Tensor, Tensor]:
    gt_start = batch["human_gt_start_u"]
    gt_end = batch["human_gt_end_u"]
    gt_width = torch.clamp(gt_end - gt_start, min=float(min_width))
    predicted_start = batch["predicted_human_start_u"]
    has_prediction = torch.isfinite(predicted_start)
    blended = (1.0 - float(predicted_u_alpha)) * gt_start + float(predicted_u_alpha) * predicted_start
    start_u = torch.where(has_prediction, blended, gt_start)
    start_u = torch.clamp(start_u, min=0.0, max=1.0 - float(min_width))
    end_u = torch.minimum(start_u + gt_width, torch.ones_like(start_u))
    min_end = torch.clamp(start_u + float(min_width), max=torch.ones_like(start_u))
    end_u = torch.maximum(end_u, min_end)
    return start_u, end_u


def _evaluate(
    model: LocalHumanToRobotSplineModel,
    dataloader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    config: dict[str, Any],
    predicted_u_alpha: float,
    teacher_forcing_alpha: float,
    log_frequency_steps: int,
    effective_loss_weights: dict[str, float],
) -> dict[str, float]:
    model.eval()
    averages = MetricAverages()
    min_input_width = float(config["human_input_u"]["min_interval_width"])
    with torch.no_grad():
        iterator = tqdm(dataloader, desc="validation", unit="batch", leave=False)
        for batch_index, batch in enumerate(iterator, start=1):
            batch = _move_to_device(batch, device)
            targets = _build_robot_targets(batch, model, config)
            human_start_u, human_end_u = _select_human_input_interval(batch, predicted_u_alpha, min_input_width)
            with autocast(device_type=device.type, enabled=amp_enabled):
                outputs = model(
                    robot_history_embeddings=batch["robot_history_embeddings"],
                    robot_history_states=batch["robot_history_states"],
                    robot_history_mask=batch["robot_history_mask"],
                    human_global_coefficients=batch["human_global_coefficients"],
                    human_global_knots=batch["human_global_knots"],
                    human_global_coeff_counts=batch["human_global_coeff_counts"],
                    human_global_knot_counts=batch["human_global_knot_counts"],
                    human_input_start_u=human_start_u,
                    human_input_end_u=human_end_u,
                    dense_robot_teacher=targets["dense_position"],
                    teacher_forcing_alpha=teacher_forcing_alpha,
                    compressor_gradient_gamma=0.0,
                )
                _, metrics = compute_losses(
                    outputs,
                    targets,
                    batch,
                    config,
                    teacher_forcing_alpha=teacher_forcing_alpha,
                    compressor_gradient_gamma=0.0,
                    predicted_u_alpha=predicted_u_alpha,
                    effective_loss_weights=effective_loss_weights,
                )
            averages.update(metrics)
            if batch_index % max(1, log_frequency_steps) == 0:
                current = averages.result()
                iterator.set_postfix(val_total=f"{current.get('total', float('nan')):.4f}", val_spline=f"{current.get('spline_position_rmse', float('nan')):.4f}")
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
        paths["robot_spline_root"],
        paths["robot_local_window_root"],
        paths["robot_pairing_root"],
        categories,
        skip_missing=bool(config["data"]["skip_missing_robot_episodes"]),
    )
    split_map = split_robot_episodes_by_category(
        valid_robot_episodes,
        split_seed=int(config["validation"]["split_seed"]),
        per_category_ratio=float(config["validation"]["per_category_ratio"]),
        min_episodes_per_category=int(config["validation"]["min_episodes_per_category"]),
        max_episodes_per_category=(
            int(config["validation"]["max_episodes_per_category"])
            if config["validation"].get("max_episodes_per_category") is not None
            else None
        ),
    )
    atomic_json_dump(split_map, paths["splits_path"])

    train_episode_ids = [episode for category in categories for episode in split_map[category.category_id]["train"]]
    val_episode_ids = [episode for category in categories for episode in split_map[category.category_id]["val"]]
    human_episode_ids = collect_human_episode_ids_from_pairings(train_episode_ids + val_episode_ids, paths["robot_pairing_root"])
    preprocessing_summary = precompute_exact_human_interval_cache(
        train_episode_ids + val_episode_ids,
        paths["sim_dataset_root"],
        paths["human_dataset_root"],
        paths["robot_pairing_root"],
        paths["robot_local_window_root"],
        paths["human_bspline_root"],
        paths["exact_human_interval_cache_root"],
        overwrite=bool(config["preprocessing"]["overwrite"]),
    )
    state_norm = _state_norm_payload(config, paths, train_episode_ids)
    category_metadata = {
        "category_order": [category.category_id for category in categories],
        "train_episode_count": len(train_episode_ids),
        "val_episode_count": len(val_episode_ids),
        "human_episode_count": len(human_episode_ids),
    }
    atomic_json_dump(category_metadata, paths["category_metadata_path"])
    atomic_json_dump(preprocessing_summary, paths["preprocessing_summary_path"])

    model = LocalHumanToRobotSplineModel(config=config, state_dim=len(state_norm["state_dims"]))
    device = torch.device(str(config["training"]["device"]))
    model.to(device)

    pairing_slot_schedule = [int(value) for value in config["training"]["pairing_slot_schedule"]]
    if not pairing_slot_schedule:
        raise ValueError("training.pairing_slot_schedule must not be empty")
    batch_size = int(config["training"]["batch_size"])
    grad_accum_steps = max(1, int(config["training"]["gradient_accumulation_steps"]))
    total_optimizer_steps = 0
    for epoch_index in range(int(config["training"]["epochs"])):
        slot = pairing_slot_schedule[epoch_index % len(pairing_slot_schedule)]
        slot_samples = _count_valid_samples_for_slot(train_episode_ids, paths["exact_human_interval_cache_root"], paths["robot_local_window_root"], slot)
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

    teacher_forcing_schedule = PiecewiseLinearSchedule.from_config(config["training"]["teacher_forcing_alpha_schedule"])
    gamma_schedule = PiecewiseLinearSchedule.from_config(config["training"]["compressor_gradient_gamma_schedule"])
    predicted_u_schedule = PiecewiseLinearSchedule.from_config(config["training"]["predicted_u_alpha_schedule"])
    derivative_weight_schedules = _build_derivative_weight_schedules(config)

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
        "collate_fn": translator_collate,
    }

    dataset_paths = {
        "sim_dataset_root": paths["sim_dataset_root"],
        "human_bspline_root": paths["human_bspline_root"],
        "robot_embedding_root": paths["robot_embedding_root"],
        "robot_spline_root": paths["robot_spline_root"],
        "robot_local_window_root": paths["robot_local_window_root"],
        "robot_pairing_root": paths["robot_pairing_root"],
        "predicted_human_u_root": paths["predicted_human_u_root"],
        "exact_human_interval_cache_root": paths["exact_human_interval_cache_root"],
    }
    min_input_width = float(config["human_input_u"]["min_interval_width"])
    validation_predicted_u_alpha = float(config["human_input_u"]["validation_predicted_u_alpha"])
    validation_teacher_forcing_alpha = float(config["validation"]["teacher_forcing_alpha"])
    validation_loss_weights = _effective_derivative_weights(config, derivative_weight_schedules, 1.0)

    for epoch_index in range(start_epoch, int(config["training"]["epochs"])):
        pairing_slot = pairing_slot_schedule[epoch_index % len(pairing_slot_schedule)]
        train_dataset = TranslatorDataset(
            train_episode_ids,
            pairing_slot=pairing_slot,
            paths=dataset_paths,
            state_dims=state_norm["state_dims"],
            state_mean=np.asarray(state_norm["state_mean"], dtype=np.float32),
            state_std=np.asarray(state_norm["state_std"], dtype=np.float32),
            category_to_index=category_to_index,
            categories=categories,
            config=config,
        )
        val_dataset = TranslatorDataset(
            val_episode_ids,
            pairing_slot=pairing_slot,
            paths=dataset_paths,
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
            schedule_progress = global_step / float(max(1, total_optimizer_steps - 1))
            teacher_forcing_alpha = teacher_forcing_schedule.value(schedule_progress)
            compressor_gradient_gamma = gamma_schedule.value(schedule_progress)
            predicted_u_alpha = predicted_u_schedule.value(schedule_progress)
            effective_loss_weights = _effective_derivative_weights(config, derivative_weight_schedules, schedule_progress)
            with torch.no_grad():
                targets = _build_robot_targets(batch, model, config)
                human_input_start_u, human_input_end_u = _select_human_input_interval(batch, predicted_u_alpha, min_input_width)
            with autocast(device_type=device.type, enabled=amp_enabled):
                outputs = model(
                    robot_history_embeddings=batch["robot_history_embeddings"],
                    robot_history_states=batch["robot_history_states"],
                    robot_history_mask=batch["robot_history_mask"],
                    human_global_coefficients=batch["human_global_coefficients"],
                    human_global_knots=batch["human_global_knots"],
                    human_global_coeff_counts=batch["human_global_coeff_counts"],
                    human_global_knot_counts=batch["human_global_knot_counts"],
                    human_input_start_u=human_input_start_u,
                    human_input_end_u=human_input_end_u,
                    dense_robot_teacher=targets["dense_position"],
                    teacher_forcing_alpha=teacher_forcing_alpha,
                    compressor_gradient_gamma=compressor_gradient_gamma,
                )
                loss, metrics = compute_losses(
                    outputs,
                    targets,
                    batch,
                    config,
                    teacher_forcing_alpha=teacher_forcing_alpha,
                    compressor_gradient_gamma=compressor_gradient_gamma,
                    predicted_u_alpha=predicted_u_alpha,
                    effective_loss_weights=effective_loss_weights,
                )
                loss = loss / grad_accum_steps
            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                tqdm.write(
                    f"[warn] skipping non-finite batch epoch={epoch_index + 1}/{config['training']['epochs']} "
                    f"slot={pairing_slot} batch={batch_index} step={global_step} "
                    f"robot_episode={batch['robot_episode_index'][0].item()} frame_row={batch['robot_frame_row'][0].item()} "
                    f"human_episode={batch['human_episode_index'][0].item()}"
                )
                continue
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
                        spline=f"{current.get('spline_position_rmse', float('nan')):.4f}",
                        dense=f"{current.get('dense_position_rmse', float('nan')):.4f}",
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
                        config,
                        predicted_u_alpha=validation_predicted_u_alpha,
                        teacher_forcing_alpha=validation_teacher_forcing_alpha,
                        log_frequency_steps=int(config["training"]["validation_log_frequency_steps"]),
                        effective_loss_weights=validation_loss_weights,
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
                config,
                predicted_u_alpha=validation_predicted_u_alpha,
                teacher_forcing_alpha=validation_teacher_forcing_alpha,
                log_frequency_steps=int(config["training"]["validation_log_frequency_steps"]),
                effective_loss_weights=validation_loss_weights,
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
