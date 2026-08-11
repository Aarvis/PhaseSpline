from __future__ import annotations

import torch
from torch import nn

from lehome_spline.model import DinoFeatures, VisualSplineVAE
from lehome_spline.training import (
    Stage,
    allocate_stage_steps,
    calculate_batch_loss,
    episode_aware_stage_indices,
)


class FakeBackbone(nn.Module):
    hidden_size = 32

    def forward(self, images: torch.Tensor) -> DinoFeatures:
        batch = images.shape[0]
        base = images.mean(dim=(1, 2, 3), keepdim=False)[:, None, None]
        patches = base.expand(batch, 4, self.hidden_size).contiguous()
        global_token = base[:, 0].expand(batch, self.hidden_size).contiguous()
        return DinoFeatures(patches=patches, global_token=global_token)


def tiny_config() -> dict:
    return {
        "dataset": {"horizons": [1, 2]},
        "model": {
            "dino": {"model_name": "unused", "hidden_size": 32},
            "latent": {
                "num_slots": 2,
                "slot_dim": 16,
                "layers": 1,
                "heads": 4,
                "mlp_ratio": 2,
                "log_variance_min": -8.0,
                "log_variance_max": 4.0,
            },
            "decoder": {"num_patches": 4, "layers": 1, "heads": 4, "mlp_ratio": 2},
            "temporal": {"layers": 1, "heads": 4, "mlp_ratio": 2},
        },
        "loss": {
            "cosine_weight": 0.1,
            "current_patch": 1.0,
            "current_global": 0.25,
            "current_state": 0.1,
            "variance": 0.01,
            "variance_floor": 0.1,
            "future_mean": 1.0,
            "conditional_kl": 0.1,
            "future_patch": 1.0,
            "future_global": 0.25,
            "joint_spatial": 1.0,
            "joint_temporal": 1.0,
        },
    }


def test_variational_and_temporal_shapes() -> None:
    model = VisualSplineVAE(tiny_config(), backbone=FakeBackbone())
    images = torch.randn(3, 3, 16, 16)
    dino = model.encode_dino(images)
    posterior = model.posterior_from_patches(dino.patches)
    assert posterior.mean.shape == (3, 2, 16)
    assert posterior.log_variance.shape == (3, 2, 16)
    reconstruction = model.reconstruct(posterior.mean, patch_count=4)
    assert reconstruction.patches.shape == (3, 4, 32)
    assert reconstruction.global_token.shape == (3, 32)
    state = torch.randn(3, 4)
    actions = torch.randn(3, 2, 4)
    prior_mean, prior_log_variance = model.temporal_prior(posterior.mean, state, actions)
    assert prior_mean.shape == (3, 2, 2, 16)
    assert prior_log_variance.shape == (3, 2, 2, 16)


def test_spatial_and_temporal_losses_are_finite() -> None:
    config = tiny_config()
    model = VisualSplineVAE(config, backbone=FakeBackbone())
    spatial_stage = Stage("spatial", 1, 1e-3, True, False, 1e-3, 0.1)
    spatial_batch = {
        "images": torch.randn(2, 1, 3, 16, 16),
        "state": torch.randn(2, 4),
        "actions": torch.randn(2, 2, 4),
    }
    spatial_loss, _ = calculate_batch_loss(model, spatial_batch, spatial_stage, config, 1e-3)
    assert torch.isfinite(spatial_loss)

    temporal_stage = Stage("temporal", 1, 1e-3, True, True, 1e-3, 0.0)
    temporal_batch = {
        "images": torch.randn(2, 3, 3, 16, 16),
        "state": torch.randn(2, 4),
        "actions": torch.randn(2, 2, 4),
    }
    temporal_loss, metrics = calculate_batch_loss(model, temporal_batch, temporal_stage, config, 1e-3)
    assert torch.isfinite(temporal_loss)
    assert torch.isfinite(metrics["conditional_kl"])
    assert torch.allclose(
        temporal_loss,
        metrics["spatial_total"] + metrics["temporal_total"],
    )
    temporal_loss.backward()
    assert any(parameter.grad is not None for parameter in model.resampler.parameters())
    assert any(parameter.grad is not None for parameter in model.temporal_prior.parameters())


def test_stage_step_allocation_covers_one_epoch() -> None:
    assert allocate_stage_steps(10, [0.3, 0.3, 0.4]) == [3, 3, 4]
    assert allocate_stage_steps(11, [0.3, 0.3, 0.4]) == [3, 3, 5]
    assert sum(allocate_stage_steps(101, [0.3, 0.3, 0.4])) == 101


def test_episode_aware_partition_is_disjoint_and_exact() -> None:
    class TinyWindowDataset:
        episodes = [object(), object(), object()]
        windows = [(0, index) for index in range(6)]
        windows += [(1, index) for index in range(5)]
        windows += [(2, index) for index in range(4)]

        def __len__(self) -> int:
            return len(self.windows)

    dataset = TinyWindowDataset()
    partition = episode_aware_stage_indices(dataset, [4, 4, 4], seed=2027)
    assert [len(indices) for indices in partition] == [4, 4, 4]
    flattened = [index for stage_indices in partition for index in stage_indices]
    assert len(flattened) == len(set(flattened)) == 12

    for stage_indices in partition:
        episode_sequence = [dataset.windows[index][0] for index in stage_indices]
        completed: set[int] = set()
        previous = episode_sequence[0]
        for current in episode_sequence[1:]:
            if current != previous:
                completed.add(previous)
                assert current not in completed
                previous = current
