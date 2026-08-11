from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn


@dataclass
class Posterior:
    slots: Tensor
    mean: Tensor
    log_variance: Tensor

    def sample(self, stochastic: bool) -> Tensor:
        if not stochastic:
            return self.mean
        noise = torch.randn_like(self.mean)
        return self.mean + torch.exp(0.5 * self.log_variance) * noise


@dataclass
class DinoFeatures:
    patches: Tensor
    global_token: Tensor


class FrozenDinoV3(nn.Module):
    """Frozen Hugging Face DINOv3 wrapper returning global and spatial tokens."""

    def __init__(self, model_name: str) -> None:
        super().__init__()
        try:
            from transformers import AutoModel
        except ImportError as exc:  # pragma: no cover - dependency error is user-facing
            raise RuntimeError("transformers>=4.56 is required for DINOv3") from exc
        self.backbone = AutoModel.from_pretrained(model_name)
        self.backbone.requires_grad_(False)
        self.backbone.eval()
        model_config = self.backbone.config
        self.hidden_size = int(model_config.hidden_size)
        self.num_register_tokens = int(getattr(model_config, "num_register_tokens", 0))

    def train(self, mode: bool = True) -> "FrozenDinoV3":
        super().train(False)
        self.backbone.eval()
        return self

    @torch.no_grad()
    def forward(self, images: Tensor) -> DinoFeatures:
        outputs = self.backbone(pixel_values=images)
        tokens = outputs.last_hidden_state
        global_token = tokens[:, 0]
        patch_start = 1 + self.num_register_tokens
        return DinoFeatures(patches=tokens[:, patch_start:], global_token=global_token)


class LatentResampler(nn.Module):
    def __init__(
        self,
        input_dim: int,
        model_dim: int,
        num_latents: int,
        num_layers: int,
        num_heads: int,
        mlp_ratio: int,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_dim, model_dim)
        self.latent_queries = nn.Parameter(torch.empty(1, num_latents, model_dim))
        nn.init.trunc_normal_(self.latent_queries, std=0.02)
        layer = nn.TransformerDecoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * mlp_ratio,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(model_dim)

    def forward(self, patch_tokens: Tensor) -> Tensor:
        memory = self.input_projection(patch_tokens)
        queries = self.latent_queries.expand(patch_tokens.shape[0], -1, -1)
        return self.output_norm(self.decoder(queries, memory))


class VariationalPosteriorHead(nn.Module):
    def __init__(self, model_dim: int, log_variance_min: float, log_variance_max: float) -> None:
        super().__init__()
        self.mean = nn.Sequential(nn.LayerNorm(model_dim), nn.Linear(model_dim, model_dim))
        self.log_variance = nn.Sequential(nn.LayerNorm(model_dim), nn.Linear(model_dim, model_dim))
        self.log_variance_min = float(log_variance_min)
        self.log_variance_max = float(log_variance_max)

    def forward(self, slots: Tensor) -> Posterior:
        mean = self.mean(slots)
        log_variance = self.log_variance(slots).clamp(self.log_variance_min, self.log_variance_max)
        return Posterior(slots=slots, mean=mean, log_variance=log_variance)


class PatchFeatureDecoder(nn.Module):
    def __init__(
        self,
        model_dim: int,
        output_dim: int,
        num_patches: int,
        num_layers: int,
        num_heads: int,
        mlp_ratio: int,
    ) -> None:
        super().__init__()
        self.num_patches = int(num_patches)
        self.patch_queries = nn.Parameter(torch.empty(1, num_patches, model_dim))
        nn.init.trunc_normal_(self.patch_queries, std=0.02)
        layer = nn.TransformerDecoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * mlp_ratio,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.output = nn.Sequential(nn.LayerNorm(model_dim), nn.Linear(model_dim, output_dim))

    def forward(self, latent_slots: Tensor, patch_count: int | None = None) -> Tensor:
        patch_count = patch_count or self.num_patches
        if patch_count > self.num_patches:
            raise ValueError(f"Decoder supports at most {self.num_patches} patches, got {patch_count}")
        queries = self.patch_queries[:, :patch_count].expand(latent_slots.shape[0], -1, -1)
        return self.output(self.decoder(queries, latent_slots))


class GlobalFeatureHead(nn.Module):
    def __init__(self, latent_size: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(latent_size),
            nn.Linear(latent_size, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, latent_slots: Tensor) -> Tensor:
        return self.network(latent_slots.flatten(1))


class StateProbe(nn.Module):
    def __init__(self, latent_size: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(latent_size),
            nn.Linear(latent_size, 512),
            nn.SiLU(),
            nn.Linear(512, 128),
            nn.SiLU(),
            nn.Linear(128, 4),
        )

    def forward(self, mean_slots: Tensor) -> Tensor:
        return self.network(mean_slots.flatten(1))


class TemporalPrior(nn.Module):
    """Produces horizon-specific diagonal-Gaussian priors over future latent slots."""

    def __init__(
        self,
        horizons: list[int],
        model_dim: int,
        num_latents: int,
        num_layers: int,
        num_heads: int,
        mlp_ratio: int,
        log_variance_min: float,
        log_variance_max: float,
    ) -> None:
        super().__init__()
        self.horizons = tuple(sorted(int(value) for value in horizons))
        self.state_encoder = nn.Sequential(nn.Linear(4, model_dim), nn.SiLU(), nn.Linear(model_dim, model_dim))
        self.action_encoder = nn.Sequential(nn.Linear(8, model_dim), nn.SiLU(), nn.Linear(model_dim, model_dim))
        self.action_positions = nn.Parameter(torch.empty(1, max(self.horizons), model_dim))
        self.horizon_embeddings = nn.Parameter(torch.empty(len(self.horizons), 1, model_dim))
        nn.init.trunc_normal_(self.action_positions, std=0.02)
        nn.init.trunc_normal_(self.horizon_embeddings, std=0.02)
        layer = nn.TransformerDecoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * mlp_ratio,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.mean_residual = nn.Sequential(nn.LayerNorm(model_dim), nn.Linear(model_dim, model_dim))
        self.log_variance = nn.Sequential(nn.LayerNorm(model_dim), nn.Linear(model_dim, model_dim))
        self.log_variance_min = float(log_variance_min)
        self.log_variance_max = float(log_variance_max)
        self.num_latents = int(num_latents)

    @staticmethod
    def _action_features(current_state: Tensor, actions: Tensor) -> Tensor:
        previous = torch.cat([current_state[:, None], actions[:, :-1]], dim=1)
        delta = actions - previous
        return torch.cat([actions, delta], dim=-1)

    def forward(self, current_mean: Tensor, current_state: Tensor, actions: Tensor) -> tuple[Tensor, Tensor]:
        if actions.shape[1] < max(self.horizons):
            raise ValueError("Action path is shorter than the largest configured horizon")
        action_features = self._action_features(current_state, actions)
        encoded_actions = self.action_encoder(action_features)
        encoded_actions = encoded_actions + self.action_positions[:, : encoded_actions.shape[1]]
        state_token = self.state_encoder(current_state)[:, None]

        means: list[Tensor] = []
        log_variances: list[Tensor] = []
        for horizon_index, horizon in enumerate(self.horizons):
            memory = torch.cat([state_token, encoded_actions[:, :horizon]], dim=1)
            queries = current_mean + self.horizon_embeddings[horizon_index]
            decoded = self.decoder(queries, memory)
            means.append(current_mean + self.mean_residual(decoded))
            log_variances.append(
                self.log_variance(decoded).clamp(self.log_variance_min, self.log_variance_max)
            )
        return torch.stack(means, dim=1), torch.stack(log_variances, dim=1)


class VisualSplineVAE(nn.Module):
    def __init__(self, config: dict[str, Any], backbone: FrozenDinoV3 | None = None) -> None:
        super().__init__()
        model_config = config["model"]
        dino_config = model_config["dino"]
        self.backbone = backbone or FrozenDinoV3(dino_config["model_name"])
        dino_dim = int(dino_config.get("hidden_size", self.backbone.hidden_size))
        latent_config = model_config["latent"]
        decoder_config = model_config["decoder"]
        temporal_config = model_config["temporal"]
        self.model_dim = int(latent_config["slot_dim"])
        self.num_latents = int(latent_config["num_slots"])
        self.latent_size = self.model_dim * self.num_latents
        self.horizons = tuple(sorted(int(value) for value in config["dataset"]["horizons"]))

        self.resampler = LatentResampler(
            dino_dim,
            self.model_dim,
            self.num_latents,
            int(latent_config["layers"]),
            int(latent_config["heads"]),
            int(latent_config["mlp_ratio"]),
        )
        self.posterior_head = VariationalPosteriorHead(
            self.model_dim,
            float(latent_config["log_variance_min"]),
            float(latent_config["log_variance_max"]),
        )
        self.patch_decoder = PatchFeatureDecoder(
            self.model_dim,
            dino_dim,
            int(decoder_config["num_patches"]),
            int(decoder_config["layers"]),
            int(decoder_config["heads"]),
            int(decoder_config["mlp_ratio"]),
        )
        self.global_head = GlobalFeatureHead(self.latent_size, 1024, dino_dim)
        self.state_probe = StateProbe(self.latent_size)
        self.temporal_prior = TemporalPrior(
            list(self.horizons),
            self.model_dim,
            self.num_latents,
            int(temporal_config["layers"]),
            int(temporal_config["heads"]),
            int(temporal_config["mlp_ratio"]),
            float(latent_config["log_variance_min"]),
            float(latent_config["log_variance_max"]),
        )

    def encode_dino(self, images: Tensor) -> DinoFeatures:
        return self.backbone(images)

    def posterior_from_patches(self, patches: Tensor) -> Posterior:
        return self.posterior_head(self.resampler(patches))

    def reconstruct(self, latent_slots: Tensor, patch_count: int) -> DinoFeatures:
        return DinoFeatures(
            patches=self.patch_decoder(latent_slots, patch_count),
            global_token=self.global_head(latent_slots),
        )

    def freeze_spatial_vae(self) -> None:
        modules = [self.resampler, self.posterior_head, self.patch_decoder, self.global_head, self.state_probe]
        for module in modules:
            module.requires_grad_(False)
            module.eval()
        self.temporal_prior.requires_grad_(True)
        self.temporal_prior.train()

    def unfreeze_spatial_vae(self) -> None:
        for module in [self.resampler, self.posterior_head, self.patch_decoder, self.global_head, self.state_probe]:
            module.requires_grad_(True)

