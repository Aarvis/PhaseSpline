from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass
class FeatureLoss:
    total: Tensor
    mse: Tensor
    cosine: Tensor


def feature_reconstruction_loss(prediction: Tensor, target: Tensor, cosine_weight: float) -> FeatureLoss:
    target = target.detach()
    prediction_norm = F.layer_norm(prediction, (prediction.shape[-1],))
    target_norm = F.layer_norm(target, (target.shape[-1],))
    mse = F.mse_loss(prediction_norm, target_norm)
    cosine = (1.0 - F.cosine_similarity(prediction, target, dim=-1)).mean()
    return FeatureLoss(total=mse + cosine_weight * cosine, mse=mse, cosine=cosine)


def standard_normal_kl(mean: Tensor, log_variance: Tensor) -> Tensor:
    """KL(q || N(0,I)), averaged over batch, slots, and coordinates."""
    return -0.5 * (1.0 + log_variance - mean.square() - log_variance.exp()).mean()


def diagonal_gaussian_kl(
    posterior_mean: Tensor,
    posterior_log_variance: Tensor,
    prior_mean: Tensor,
    prior_log_variance: Tensor,
) -> Tensor:
    """KL(q || p) for diagonal Gaussians, averaged over every dimension."""
    posterior_variance = posterior_log_variance.exp()
    prior_variance = prior_log_variance.exp()
    return 0.5 * (
        prior_log_variance
        - posterior_log_variance
        + (posterior_variance + (posterior_mean - prior_mean).square()) / prior_variance
        - 1.0
    ).mean()


def mean_alignment_loss(prediction: Tensor, target: Tensor, cosine_weight: float) -> FeatureLoss:
    return feature_reconstruction_loss(prediction, target.detach(), cosine_weight)


def variance_floor_loss(latent_mean: Tensor, floor: float) -> Tensor:
    flattened = latent_mean.flatten(1)
    standard_deviation = torch.sqrt(flattened.var(dim=0, unbiased=False) + 1e-4)
    return F.relu(float(floor) - standard_deviation).square().mean()


def normalized_horizon_weights(horizons: list[int] | tuple[int, ...], device: torch.device) -> Tensor:
    values = torch.tensor([float(value) ** -0.5 for value in horizons], device=device)
    return values / values.sum()

