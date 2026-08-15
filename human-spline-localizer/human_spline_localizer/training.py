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
    ExplicitUnseenValidationDataset,
    LocalizerDataset,
    as_path,
    collect_human_episode_ids_from_pairings,
    compute_state_normalization,
    discover_valid_robot_episodes,
    load_category_configs,
    localizer_collate,
    load_robot_target_cache,
    prepare_human_spline_cache,
    prepare_robot_alignment_target_cache,
    read_dataset_info,
    resolve_state_dims,
    robot_target_cache_npz_path,
    split_robot_episodes_by_category,
    load_explicit_unseen_validation_pairs,
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
)


def _interval_prediction_enabled(config: dict[str, Any]) -> bool:
    return bool(config["model"]["auxiliary"].get("interval_prediction", {}).get("enabled", False))


def _move_to_device(batch: dict[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}


def _format_metric_summary(metrics: dict[str, float]) -> str:
    ordered_keys = (
        "total",
        "loc_loss",
        "end_loss",
        "phase_mae",
        "end_u_mae",
        "delta_u_mae",
        "c_max",
        "entropy",
        "margin",
        "end_valid_fraction",
    )
    parts: list[str] = []
    for key in ordered_keys:
        if key in metrics:
            parts.append(f"{key}={float(metrics[key]):.4f}")
    for key, value in metrics.items():
        if key not in ordered_keys:
            parts.append(f"{key}={float(value):.4f}")
    return " ".join(parts)


def _unseen_validation_config(config: dict[str, Any]) -> dict[str, Any]:
    validation_cfg = config.get("validation", {})
    if not isinstance(validation_cfg, dict):
        return {}
    unseen_cfg = validation_cfg.get("unseen_validation", {})
    return unseen_cfg if isinstance(unseen_cfg, dict) else {}


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
        "robot_local_window_root": (
            as_path(config["paths"]["robot_local_window_root"])
            if config["paths"].get("robot_local_window_root") is not None
            else None
        ),
        "human_bspline_root": as_path(config["paths"]["human_bspline_root"]),
        "output_root": output_root,
        "human_cache_root": output_root / str(preprocessing["human_cache_dirname"]),
        "robot_target_cache_root": output_root / str(preprocessing["robot_target_cache_dirname"]),
        "checkpoints_root": output_root / "checkpoints",
        "logs_root": output_root / "logs",
        "splits_path": output_root / str(preprocessing["splits_filename"]),
        "state_norm_path": output_root / str(preprocessing["state_norm_filename"]),
        "resolved_config_path": output_root / "resolved_config.yaml",
    }


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


def _save_best_unseen_checkpoint_if_improved(
    checkpoints_root: Path,
    model: GlobalSplineLocalizer,
    optimizer: AdamW,
    scheduler: LambdaLR,
    scaler: GradScaler,
    epoch_index: int,
    global_step: int,
    current_best_unseen_metric: float,
    unseen_validation_metrics: dict[str, float] | None,
    config: dict[str, Any],
    save_reason: str,
) -> float:
    if unseen_validation_metrics is None:
        return current_best_unseen_metric
    monitored = float(unseen_validation_metrics.get("total", float("inf")))
    if not math.isfinite(monitored):
        return current_best_unseen_metric
    if monitored < current_best_unseen_metric:
        _save_checkpoint(
            checkpoints_root / "best_unseen_val.pt",
            model,
            optimizer,
            scheduler,
            scaler,
            epoch_index,
            global_step,
            monitored,
            config,
        )
        tqdm.write(
            f"[checkpoint] saved best unseen-val checkpoint at {save_reason} "
            f"step={global_step} metric=unseen_val_total value={monitored:.4f}"
        )
        return monitored
    return current_best_unseen_metric


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
    config: dict[str, Any],
) -> tuple[Tensor, dict[str, Tensor]]:
    loc_loss = -(batch["soft_target"] * torch.log_softmax(outputs["logits"], dim=-1)).sum(dim=-1).mean()
    interval_enabled = _interval_prediction_enabled(config)
    end_loss = outputs["logits"].new_zeros(())
    end_u_mae = outputs["logits"].new_zeros(())
    delta_u_mae = outputs["logits"].new_zeros(())
    end_valid_fraction = outputs["logits"].new_tensor(batch["end_target_valid"].to(torch.float32).mean().item())
    if interval_enabled:
        valid_mask = batch["end_target_valid"].to(torch.bool)
        if torch.any(valid_mask):
            end_predictions = outputs["u_end_hat"][valid_mask]
            end_targets = batch["target_end_u"][valid_mask]
            end_loss = F.huber_loss(
                end_predictions,
                end_targets,
                delta=float(config["loss"]["end_delta"]),
            )
            end_u_mae = torch.abs(end_predictions - end_targets).mean()
            delta_u_mae = torch.abs(outputs["delta_u_hat"][valid_mask] - batch["target_delta_u"][valid_mask]).mean()
    total = loc_loss + float(config["loss"].get("lambda_end", 0.5)) * end_loss
    phase_mae = torch.abs(outputs["u_hat"] - batch["target_u"]).mean()
    metrics = {
        "total": total.detach(),
        "loc_loss": loc_loss.detach(),
        "phase_mae": phase_mae.detach(),
        "c_max": outputs["c_max"].mean().detach(),
        "entropy": outputs["entropy"].mean().detach(),
        "margin": outputs["margin"].mean().detach(),
    }
    if interval_enabled:
        metrics["end_loss"] = end_loss.detach()
        metrics["end_u_mae"] = end_u_mae.detach()
        metrics["delta_u_mae"] = delta_u_mae.detach()
        metrics["end_valid_fraction"] = end_valid_fraction.detach()
    return total, metrics


def _evaluate(
    model: GlobalSplineLocalizer,
    dataloader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    config: dict[str, Any],
    log_frequency_steps: int,
    progress_desc: str = "validation",
) -> dict[str, float]:
    model.eval()
    averages = MetricAverages()
    with torch.no_grad():
        iterator = tqdm(dataloader, desc=progress_desc, unit="batch", leave=False)
        for batch_index, batch in enumerate(iterator, start=1):
            batch = _move_to_device(batch, device)
            with autocast(device_type=device.type, enabled=amp_enabled):
                outputs = model(**_model_inputs(batch))
                _, metrics = _compute_losses(outputs, batch, config)
            averages.update(metrics)
            if batch_index % max(1, log_frequency_steps) == 0:
                current = averages.result()
                iterator.set_postfix(total=f"{current.get('total', float('nan')):.4f}", mae=f"{current.get('phase_mae', float('nan')):.4f}")
    return averages.result()


def _resolve_checkpoint_metric(
    checkpoint_metric: str,
    validation_metrics: dict[str, float] | None,
    unseen_validation_metrics: dict[str, float] | None,
) -> tuple[str, float]:
    metric_name = str(checkpoint_metric)
    if metric_name.startswith("unseen_val_"):
        key = metric_name[len("unseen_val_") :]
        source = unseen_validation_metrics or {}
        return metric_name, float(source.get(key, float("inf")))
    if metric_name.startswith("val_"):
        key = metric_name[len("val_") :]
        source = validation_metrics or {}
        return metric_name, float(source.get(key, float("inf")))
    source = validation_metrics or {}
    return metric_name, float(source.get(metric_name, float("inf")))


def train(config: dict[str, Any], resume: str | None = None) -> Path:
    seed_everything(int(config["training"]["seed"]))
    paths = _paths(config)
    for path in (paths["output_root"], paths["checkpoints_root"], paths["logs_root"]):
        path.mkdir(parents=True, exist_ok=True)
    save_resolved_config(config, paths["resolved_config_path"])

    categories = load_category_configs(config)
    valid_robot_episodes = discover_valid_robot_episodes(
        paths["sim_dataset_root"],
        paths["robot_embedding_root"],
        paths["robot_pairing_root"],
        categories,
        skip_missing=bool(config["data"]["skip_missing_robot_episodes"]),
        robot_local_window_root=paths["robot_local_window_root"],
        require_local_windows=_interval_prediction_enabled(config),
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
        robot_local_window_root=paths["robot_local_window_root"],
        include_end_targets=_interval_prediction_enabled(config),
    )
    state_norm = _state_norm_payload(config, paths, train_episode_ids)

    unseen_val_loader: DataLoader | None = None
    unseen_cfg = _unseen_validation_config(config)
    if bool(unseen_cfg.get("enabled", False)):
        unseen_pairs = load_explicit_unseen_validation_pairs(
            as_path(unseen_cfg["prep_config_path"]),
            [str(value) for value in unseen_cfg.get("pair_ids", [])] if unseen_cfg.get("pair_ids") is not None else None,
        )
        unseen_dataset = ExplicitUnseenValidationDataset(
            pairs=unseen_pairs,
            exact_match_root=as_path(unseen_cfg["exact_match_root"]),
            state_dims=state_norm["state_dims"],
            state_mean=np.asarray(state_norm["state_mean"], dtype=np.float32),
            state_std=np.asarray(state_norm["state_std"], dtype=np.float32),
            config=config,
        )
        unseen_val_loader = DataLoader(
            unseen_dataset,
            shuffle=False,
            drop_last=False,
            batch_size=int(config["training"]["batch_size"]),
            num_workers=int(config["training"]["num_workers"]),
            pin_memory=bool(config["training"]["pin_memory"]) and str(config["training"]["device"]) == "cuda",
            persistent_workers=bool(config["training"]["persistent_workers"]) and int(config["training"]["num_workers"]) > 0,
            collate_fn=localizer_collate,
        )

    model = GlobalSplineLocalizer(
        config=config,
        state_dim=len(state_norm["state_dims"]),
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
    best_unseen_metric = float("inf")
    if resume is not None:
        start_epoch, global_step, best_metric = _load_checkpoint(Path(resume), model, optimizer, scheduler, scaler)
        start_epoch += 1
    best_unseen_checkpoint_path = paths["checkpoints_root"] / "best_unseen_val.pt"
    if best_unseen_checkpoint_path.exists():
        try:
            best_unseen_payload = torch.load(best_unseen_checkpoint_path, map_location="cpu")
            best_unseen_metric = float(best_unseen_payload.get("best_metric", float("inf")))
        except Exception:
            best_unseen_metric = float("inf")

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
            config=config,
        )
        val_dataset = LocalizerDataset(
            val_episode_ids,
            pairing_slot=pairing_slot,
            paths=paths,
            state_dims=state_norm["state_dims"],
            state_mean=np.asarray(state_norm["state_mean"], dtype=np.float32),
            state_std=np.asarray(state_norm["state_std"], dtype=np.float32),
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
                loss, metrics = _compute_losses(outputs, batch, config)
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
                        end_mae=f"{current.get('end_u_mae', float('nan')):.4f}",
                    )

                if (
                    (val_loader is not None or unseen_val_loader is not None)
                    and int(config["training"]["validation_frequency_steps"]) > 0
                    and global_step % int(config["training"]["validation_frequency_steps"]) == 0
                ):
                    validation_metrics: dict[str, float] | None = None
                    unseen_validation_metrics: dict[str, float] | None = None
                    if val_loader is not None:
                        validation_metrics = _evaluate(
                            model,
                            val_loader,
                            device,
                            amp_enabled,
                            config,
                            log_frequency_steps=int(config["training"]["validation_log_frequency_steps"]),
                            progress_desc="validation",
                        )
                        if wandb_run is not None:
                            wandb_run.log({f"val/{key}": value for key, value in validation_metrics.items()}, step=global_step)
                        tqdm.write(
                            f"[val] epoch={epoch_index + 1}/{config['training']['epochs']} "
                            f"slot={pairing_slot} step={global_step} "
                            f"{_format_metric_summary(validation_metrics)}"
                        )
                    if unseen_val_loader is not None:
                        unseen_validation_metrics = _evaluate(
                            model,
                            unseen_val_loader,
                            device,
                            amp_enabled,
                            config,
                            log_frequency_steps=int(config["training"]["validation_log_frequency_steps"]),
                            progress_desc="unseen-validation",
                        )
                        if wandb_run is not None:
                            wandb_run.log({f"unseen_val/{key}": value for key, value in unseen_validation_metrics.items()}, step=global_step)
                        tqdm.write(
                            f"[unseen-val] epoch={epoch_index + 1}/{config['training']['epochs']} "
                            f"slot={pairing_slot} step={global_step} "
                            f"{_format_metric_summary(unseen_validation_metrics)}"
                        )
                        best_unseen_metric = _save_best_unseen_checkpoint_if_improved(
                            paths["checkpoints_root"],
                            model,
                            optimizer,
                            scheduler,
                            scaler,
                            epoch_index,
                            global_step,
                            best_unseen_metric,
                            unseen_validation_metrics,
                            config,
                            save_reason="validation",
                        )
                    metric_name, monitored = _resolve_checkpoint_metric(
                        str(config["training"]["checkpoint_metric"]),
                        validation_metrics,
                        unseen_validation_metrics,
                    )
                    if monitored < best_metric:
                        best_metric = monitored
                        _save_checkpoint(paths["checkpoints_root"] / "best.pt", model, optimizer, scheduler, scaler, epoch_index, global_step, best_metric, config)
                        tqdm.write(
                            f"[checkpoint] saved best checkpoint at step={global_step} "
                            f"metric={metric_name} value={best_metric:.4f}"
                        )
                    model.train()

        validation_metrics = None
        unseen_validation_metrics = None
        if val_loader is not None:
            validation_metrics = _evaluate(
                model,
                val_loader,
                device,
                amp_enabled,
                config,
                log_frequency_steps=int(config["training"]["validation_log_frequency_steps"]),
                progress_desc="validation",
            )
            if wandb_run is not None:
                wandb_run.log({f"val_epoch/{key}": value for key, value in validation_metrics.items()}, step=global_step)
            tqdm.write(
                f"[val-epoch] epoch={epoch_index + 1}/{config['training']['epochs']} "
                f"slot={pairing_slot} step={global_step} "
                f"{_format_metric_summary(validation_metrics)}"
            )
        if unseen_val_loader is not None:
            unseen_validation_metrics = _evaluate(
                model,
                unseen_val_loader,
                device,
                amp_enabled,
                config,
                log_frequency_steps=int(config["training"]["validation_log_frequency_steps"]),
                progress_desc="unseen-validation",
            )
            if wandb_run is not None:
                wandb_run.log({f"unseen_val_epoch/{key}": value for key, value in unseen_validation_metrics.items()}, step=global_step)
            tqdm.write(
                f"[unseen-val-epoch] epoch={epoch_index + 1}/{config['training']['epochs']} "
                f"slot={pairing_slot} step={global_step} "
                f"{_format_metric_summary(unseen_validation_metrics)}"
            )
            best_unseen_metric = _save_best_unseen_checkpoint_if_improved(
                paths["checkpoints_root"],
                model,
                optimizer,
                scheduler,
                scaler,
                epoch_index,
                global_step,
                best_unseen_metric,
                unseen_validation_metrics,
                config,
                save_reason="epoch_end",
            )
        if validation_metrics is not None or unseen_validation_metrics is not None:
            metric_name, monitored = _resolve_checkpoint_metric(
                str(config["training"]["checkpoint_metric"]),
                validation_metrics,
                unseen_validation_metrics,
            )
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
