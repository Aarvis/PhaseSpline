from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor


def _rmse(prediction: Tensor, target: Tensor) -> Tensor:
    prediction32 = prediction.to(dtype=torch.float32)
    target32 = target.to(dtype=torch.float32)
    return torch.sqrt(torch.mean((prediction32 - target32) ** 2))


def _masked_huber(prediction: Tensor, target: Tensor, delta: float, mask: Tensor | None = None) -> Tensor:
    prediction32 = prediction.to(dtype=torch.float32)
    target32 = target.to(dtype=torch.float32)
    loss = F.huber_loss(prediction32, target32, delta=float(delta), reduction="none")
    if mask is None:
        return loss.mean()
    while mask.ndim < loss.ndim:
        mask = mask.unsqueeze(-1)
    mask = torch.broadcast_to(mask, loss.shape).to(loss.dtype)
    total_weight = torch.clamp(mask.sum(), min=1.0)
    return (loss * mask).sum() / total_weight


def compute_losses(
    outputs: dict[str, Tensor],
    targets: dict[str, Tensor],
    batch: dict[str, Tensor],
    config: dict[str, Any],
    teacher_forcing_alpha: float,
    compressor_gradient_gamma: float,
    predicted_u_alpha: float,
    effective_loss_weights: dict[str, float] | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    dense_cfg = config["loss"]["dense"]
    spline_cfg = config["loss"]["spline"]
    dense_velocity_weight = (
        float(effective_loss_weights["dense_velocity"])
        if effective_loss_weights is not None and "dense_velocity" in effective_loss_weights
        else float(dense_cfg["lambda_velocity"])
    )
    dense_acceleration_weight = (
        float(effective_loss_weights["dense_acceleration"])
        if effective_loss_weights is not None and "dense_acceleration" in effective_loss_weights
        else float(dense_cfg["lambda_acceleration"])
    )
    spline_velocity_weight = (
        float(effective_loss_weights["spline_velocity"])
        if effective_loss_weights is not None and "spline_velocity" in effective_loss_weights
        else float(spline_cfg["lambda_velocity"])
    )
    spline_acceleration_weight = (
        float(effective_loss_weights["spline_acceleration"])
        if effective_loss_weights is not None and "spline_acceleration" in effective_loss_weights
        else float(spline_cfg["lambda_acceleration"])
    )
    dense_position_loss = _masked_huber(outputs["dense_robot_pred"], targets["dense_position"], dense_cfg["position_delta"])
    dense_velocity_loss = _masked_huber(outputs["dense_robot_velocity"], targets["dense_velocity"], dense_cfg["velocity_delta"])
    dense_acceleration_loss = _masked_huber(outputs["dense_robot_acceleration"], targets["dense_acceleration"], dense_cfg["acceleration_delta"])
    dense_total = (
        dense_position_loss
        + dense_velocity_weight * dense_velocity_loss
        + dense_acceleration_weight * dense_acceleration_loss
    )

    spline_position_loss = _masked_huber(outputs["reconstructed_robot_curve"], targets["dense_position"], spline_cfg["position_delta"])
    spline_velocity_loss = _masked_huber(outputs["reconstructed_robot_velocity"], targets["dense_velocity"], spline_cfg["velocity_delta"])
    spline_acceleration_loss = _masked_huber(outputs["reconstructed_robot_acceleration"], targets["dense_acceleration"], spline_cfg["acceleration_delta"])
    spline_total = (
        spline_position_loss
        + spline_velocity_weight * spline_velocity_loss
        + spline_acceleration_weight * spline_acceleration_loss
    )

    knot_valid_mask = targets["gt_span_valid_mask"]
    if torch.any(knot_valid_mask):
        epsilon = float(config["loss"]["knot"]["log_epsilon"])
        predicted_log = torch.log(outputs["predicted_span_widths"] + epsilon)
        target_log = torch.log(targets["gt_span_widths"] + epsilon)
        knot_loss = _masked_huber(predicted_log, target_log, config["loss"]["knot"]["delta"], knot_valid_mask)
    else:
        knot_loss = outputs["dense_robot_pred"].new_zeros(())

    anchor_loss = _masked_huber(
        outputs["reconstructed_robot_curve"][:, 0, :],
        batch["robot_current_embedding"],
        config["loss"]["anchor"]["delta"],
    )
    total = (
        dense_total
        + float(config["loss"]["lambda_spline"]) * spline_total
        + float(config["loss"]["lambda_knot"]) * knot_loss
        + float(config["loss"]["lambda_anchor"]) * anchor_loss
    )

    metrics = {
        "total": total.detach(),
        "dense_total": dense_total.detach(),
        "dense_position_loss": dense_position_loss.detach(),
        "dense_velocity_loss": dense_velocity_loss.detach(),
        "dense_acceleration_loss": dense_acceleration_loss.detach(),
        "spline_total": spline_total.detach(),
        "spline_position_loss": spline_position_loss.detach(),
        "spline_velocity_loss": spline_velocity_loss.detach(),
        "spline_acceleration_loss": spline_acceleration_loss.detach(),
        "knot_loss": knot_loss.detach(),
        "anchor_loss": anchor_loss.detach(),
        "dense_position_rmse": _rmse(outputs["dense_robot_pred"], targets["dense_position"]).detach(),
        "dense_velocity_rmse": _rmse(outputs["dense_robot_velocity"], targets["dense_velocity"]).detach(),
        "dense_acceleration_rmse": _rmse(outputs["dense_robot_acceleration"], targets["dense_acceleration"]).detach(),
        "spline_position_rmse": _rmse(outputs["reconstructed_robot_curve"], targets["dense_position"]).detach(),
        "spline_velocity_rmse": _rmse(outputs["reconstructed_robot_velocity"], targets["dense_velocity"]).detach(),
        "spline_acceleration_rmse": _rmse(outputs["reconstructed_robot_acceleration"], targets["dense_acceleration"]).detach(),
        "anchor_rmse": _rmse(outputs["reconstructed_robot_curve"][:, 0, :], batch["robot_current_embedding"]).detach(),
        "span_min": outputs["predicted_span_widths"].min().detach(),
        "span_max": outputs["predicted_span_widths"].max().detach(),
        "span_entropy": outputs["span_entropy"].mean().detach(),
        "projection_condition_proxy": outputs["projection_condition_proxy"].mean().detach(),
        "gt_span_valid_fraction": knot_valid_mask.to(torch.float32).mean().detach(),
        "teacher_forcing_alpha": outputs["dense_robot_pred"].new_tensor(float(teacher_forcing_alpha)),
        "compressor_gradient_gamma": outputs["dense_robot_pred"].new_tensor(float(compressor_gradient_gamma)),
        "predicted_u_alpha": outputs["dense_robot_pred"].new_tensor(float(predicted_u_alpha)),
        "dense_velocity_weight": outputs["dense_robot_pred"].new_tensor(dense_velocity_weight),
        "dense_acceleration_weight": outputs["dense_robot_pred"].new_tensor(dense_acceleration_weight),
        "spline_velocity_weight": outputs["dense_robot_pred"].new_tensor(spline_velocity_weight),
        "spline_acceleration_weight": outputs["dense_robot_pred"].new_tensor(spline_acceleration_weight),
        "human_local_coeff_count": outputs["human_local_coefficient_count"].mean().detach(),
    }
    return total, metrics
