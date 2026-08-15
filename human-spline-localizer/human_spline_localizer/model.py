from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class FourierFeatures(nn.Module):
    def __init__(self, input_dim: int, num_bands: int) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_bands = int(num_bands)
        frequencies = 2.0 ** torch.arange(self.num_bands, dtype=torch.float32)
        self.register_buffer("frequencies", frequencies, persistent=False)

    def output_dim(self) -> int:
        return self.input_dim * (1 + 2 * self.num_bands)

    def forward(self, values: Tensor) -> Tensor:
        if values.shape[-1] != self.input_dim:
            raise ValueError(f"Expected last dim {self.input_dim}, got {values.shape[-1]}")
        expanded = values.unsqueeze(-1) * self.frequencies.view(*((1,) * values.ndim), -1)
        sin = torch.sin(2.0 * torch.pi * expanded)
        cos = torch.cos(2.0 * torch.pi * expanded)
        return torch.cat([values, sin.flatten(-2), cos.flatten(-2)], dim=-1)


class ContinuousMultiDimRoPE(nn.Module):
    def __init__(
        self,
        head_dim: int,
        coordinate_dims: list[int],
        coordinate_scale: float,
        rope_base: float,
    ) -> None:
        super().__init__()
        if sum(int(value) for value in coordinate_dims) != int(head_dim):
            raise ValueError("coordinate_dims must sum to head_dim")
        if any(int(value) <= 0 or int(value) % 2 != 0 for value in coordinate_dims):
            raise ValueError("Every coordinate RoPE slice must be a positive even dimension")
        self.head_dim = int(head_dim)
        self.coordinate_dims = [int(value) for value in coordinate_dims]
        self.coordinate_scale = float(coordinate_scale)
        self.rope_base = float(rope_base)
        inv_freq = []
        for dim in self.coordinate_dims:
            half = dim // 2
            inv = 1.0 / (self.rope_base ** (torch.arange(half, dtype=torch.float32) / max(1, half)))
            inv_freq.append(inv)
        self.inv_freq = nn.ParameterList([nn.Parameter(item, requires_grad=False) for item in inv_freq])

    @staticmethod
    def _rotate_pairs(values: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        batch, heads, tokens, dim = values.shape
        pairs = values.reshape(batch, heads, tokens, dim // 2, 2)
        even = pairs[..., 0]
        odd = pairs[..., 1]
        rotated_even = even * cos - odd * sin
        rotated_odd = even * sin + odd * cos
        return torch.stack([rotated_even, rotated_odd], dim=-1).reshape(batch, heads, tokens, dim)

    def forward(self, query: Tensor, key: Tensor, coordinates: Tensor) -> tuple[Tensor, Tensor]:
        if coordinates.shape[-1] != len(self.coordinate_dims):
            raise ValueError(
                f"Expected {len(self.coordinate_dims)} coordinate channels, got {coordinates.shape[-1]}"
            )
        query_chunks = []
        key_chunks = []
        start = 0
        for coord_index, dim in enumerate(self.coordinate_dims):
            end = start + dim
            coord = coordinates[..., coord_index].unsqueeze(1).unsqueeze(-1) * self.coordinate_scale
            angles = coord * self.inv_freq[coord_index].view(1, 1, 1, -1)
            cos = torch.cos(angles)
            sin = torch.sin(angles)
            query_chunks.append(self._rotate_pairs(query[..., start:end], cos, sin))
            key_chunks.append(self._rotate_pairs(key[..., start:end], cos, sin))
            start = end
        return torch.cat(query_chunks, dim=-1), torch.cat(key_chunks, dim=-1)


class FeedForward(nn.Module):
    def __init__(self, width: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(width, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, width),
            nn.Dropout(dropout),
        )

    def forward(self, values: Tensor) -> Tensor:
        return self.network(values)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, width: int, heads: int, dropout: float, rope: ContinuousMultiDimRoPE | None = None) -> None:
        super().__init__()
        if width % heads != 0:
            raise ValueError(f"width {width} must be divisible by heads {heads}")
        self.width = int(width)
        self.heads = int(heads)
        self.head_dim = self.width // self.heads
        self.dropout = nn.Dropout(dropout)
        self.qkv = nn.Linear(self.width, self.width * 3)
        self.output = nn.Linear(self.width, self.width)
        self.rope = rope

    def forward(self, values: Tensor, valid_mask: Tensor | None = None, coordinates: Tensor | None = None) -> Tensor:
        batch, length, _ = values.shape
        qkv = self.qkv(values).reshape(batch, length, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        if self.rope is not None:
            if coordinates is None:
                raise ValueError("RoPE attention requires coordinates")
            query, key = self.rope(query, key, coordinates)
        logits = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if valid_mask is not None:
            key_invalid = ~valid_mask[:, None, None, :]
            logits = logits.masked_fill(key_invalid, torch.finfo(logits.dtype).min)
        attention = torch.softmax(logits, dim=-1)
        attention = self.dropout(attention)
        attended = torch.matmul(attention, value).transpose(1, 2).reshape(batch, length, self.width)
        output = self.output(attended)
        if valid_mask is not None:
            output = output * valid_mask.unsqueeze(-1).to(output.dtype)
        return output


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, width: int, heads: int, dropout: float) -> None:
        super().__init__()
        if width % heads != 0:
            raise ValueError(f"width {width} must be divisible by heads {heads}")
        self.width = int(width)
        self.heads = int(heads)
        self.head_dim = self.width // self.heads
        self.dropout = nn.Dropout(dropout)
        self.query = nn.Linear(self.width, self.width)
        self.key = nn.Linear(self.width, self.width)
        self.value = nn.Linear(self.width, self.width)
        self.output = nn.Linear(self.width, self.width)

    def forward(
        self,
        query_tokens: Tensor,
        context_tokens: Tensor,
        query_mask: Tensor | None = None,
        context_mask: Tensor | None = None,
    ) -> Tensor:
        batch, q_len, _ = query_tokens.shape
        k_len = context_tokens.shape[1]
        query = self.query(query_tokens).reshape(batch, q_len, self.heads, self.head_dim).permute(0, 2, 1, 3)
        key = self.key(context_tokens).reshape(batch, k_len, self.heads, self.head_dim).permute(0, 2, 1, 3)
        value = self.value(context_tokens).reshape(batch, k_len, self.heads, self.head_dim).permute(0, 2, 1, 3)
        logits = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if context_mask is not None:
            context_invalid = ~context_mask[:, None, None, :]
            logits = logits.masked_fill(context_invalid, torch.finfo(logits.dtype).min)
        attention = torch.softmax(logits, dim=-1)
        attention = self.dropout(attention)
        attended = torch.matmul(attention, value).transpose(1, 2).reshape(batch, q_len, self.width)
        output = self.output(attended)
        if query_mask is not None:
            output = output * query_mask.unsqueeze(-1).to(output.dtype)
        return output


class SelfAttentionBlock(nn.Module):
    def __init__(self, width: int, heads: int, ffn_dim: int, dropout: float, rope: ContinuousMultiDimRoPE | None = None) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(width)
        self.attn = MultiHeadSelfAttention(width, heads, dropout, rope=rope)
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = FeedForward(width, ffn_dim, dropout)

    def forward(self, values: Tensor, valid_mask: Tensor | None = None, coordinates: Tensor | None = None) -> Tensor:
        values = values + self.attn(self.attn_norm(values), valid_mask=valid_mask, coordinates=coordinates)
        values = values + self.ffn(self.ffn_norm(values))
        if valid_mask is not None:
            values = values * valid_mask.unsqueeze(-1).to(values.dtype)
        return values


class CrossAttentionBlock(nn.Module):
    def __init__(self, width: int, heads: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(width)
        self.context_norm = nn.LayerNorm(width)
        self.cross_attn = MultiHeadCrossAttention(width, heads, dropout)
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = FeedForward(width, ffn_dim, dropout)

    def forward(
        self,
        query_tokens: Tensor,
        context_tokens: Tensor,
        query_mask: Tensor | None = None,
        context_mask: Tensor | None = None,
    ) -> Tensor:
        query_tokens = query_tokens + self.cross_attn(
            self.query_norm(query_tokens),
            self.context_norm(context_tokens),
            query_mask=query_mask,
            context_mask=context_mask,
        )
        query_tokens = query_tokens + self.ffn(self.ffn_norm(query_tokens))
        if query_mask is not None:
            query_tokens = query_tokens * query_mask.unsqueeze(-1).to(query_tokens.dtype)
        return query_tokens


class HumanCoefficientEncoder(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        human_cfg = config["model"]["human"]
        width = int(config["model"]["width"])
        self.coefficient_projection = nn.Linear(
            int(human_cfg["coefficient_input_dim"]),
            int(human_cfg["coefficient_projection_dim"]),
        )
        width_mode = str(human_cfg.get("width_encoder", "fourier_mlp"))
        width_hidden = int(human_cfg["width_hidden_dim"])
        if width_mode == "fourier_mlp":
            self.width_features = FourierFeatures(1, int(human_cfg["width_fourier_bands"]))
            width_input_dim = self.width_features.output_dim()
        elif width_mode == "mlp":
            self.width_features = None
            width_input_dim = 1
        else:
            raise ValueError(f"Unsupported human width encoder: {width_mode}")
        self.width_projection = nn.Sequential(
            nn.Linear(width_input_dim, width_hidden),
            nn.GELU(),
            nn.Linear(width_hidden, width),
        )
        self.human_type = nn.Parameter(torch.zeros((1, 1, width)))
        self.output_norm = nn.LayerNorm(width)

    def forward(self, coefficients: Tensor, support_width: Tensor) -> Tensor:
        coeff = self.coefficient_projection(coefficients)
        width_values = support_width.unsqueeze(-1)
        if self.width_features is not None:
            width_values = self.width_features(width_values)
        width_embed = self.width_projection(width_values)
        return self.output_norm(coeff + width_embed + self.human_type)


class HumanSplineTransformer(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        model_cfg = config["model"]
        human_cfg = model_cfg["human"]
        width = int(model_cfg["width"])
        heads = int(human_cfg["heads"])
        head_dim = width // heads
        rope = ContinuousMultiDimRoPE(
            head_dim=head_dim,
            coordinate_dims=[head_dim // 4] * 4,
            coordinate_scale=float(human_cfg["rope_coordinate_scale"]),
            rope_base=float(human_cfg["rope_base"]),
        )
        self.encoder = HumanCoefficientEncoder(config)
        self.blocks = nn.ModuleList(
            [
                SelfAttentionBlock(
                    width=width,
                    heads=heads,
                    ffn_dim=int(human_cfg["ffn_dim"]),
                    dropout=float(model_cfg["dropout"]),
                    rope=rope,
                )
                for _ in range(int(human_cfg["transformer_layers"]))
            ]
        )

    def forward(
        self,
        coefficients: Tensor,
        support_width: Tensor,
        left_support: Tensor,
        right_support: Tensor,
        support_midpoint: Tensor,
        greville_phase: Tensor,
        valid_mask: Tensor,
    ) -> Tensor:
        values = self.encoder(coefficients, support_width)
        coordinates = torch.stack([left_support, right_support, support_midpoint, greville_phase], dim=-1)
        for block in self.blocks:
            values = block(values, valid_mask=valid_mask, coordinates=coordinates)
        return values


class RobotObservationEncoder(nn.Module):
    def __init__(self, config: dict[str, Any], state_dim: int) -> None:
        super().__init__()
        model_cfg = config["model"]
        robot_cfg = model_cfg["robot"]
        width = int(model_cfg["width"])
        self.visual_projection = nn.Linear(
            int(robot_cfg["visual_input_dim"]),
            int(robot_cfg["visual_projection_dim"]),
        )
        self.state_projection = nn.Sequential(
            nn.Linear(state_dim, int(robot_cfg["state_hidden_dim"])),
            nn.SiLU(),
            nn.Linear(int(robot_cfg["state_hidden_dim"]), int(robot_cfg["state_projection_dim"])),
        )
        fusion_hidden = int(robot_cfg["fusion_hidden_dim"])
        self.fusion = nn.Sequential(
            nn.Linear(width, fusion_hidden),
            nn.SiLU(),
            nn.Linear(fusion_hidden, width),
        )
        self.fusion_norm = nn.LayerNorm(width)
        self.robot_type = nn.Parameter(torch.zeros((1, 1, width)))
        self.output_norm = nn.LayerNorm(width)

    def forward(self, robot_visual_history: Tensor, robot_state_history: Tensor) -> Tensor:
        visual = self.visual_projection(robot_visual_history)
        state = self.state_projection(robot_state_history)
        fused = torch.cat([visual, state], dim=-1)
        fused = self.fusion_norm(fused + self.fusion(fused))
        return self.output_norm(fused + self.robot_type)


class RobotTemporalTransformer(nn.Module):
    def __init__(self, config: dict[str, Any], state_dim: int) -> None:
        super().__init__()
        model_cfg = config["model"]
        robot_cfg = model_cfg["robot"]
        width = int(model_cfg["width"])
        heads = int(robot_cfg["heads"])
        head_dim = width // heads
        rope = ContinuousMultiDimRoPE(
            head_dim=head_dim,
            coordinate_dims=[head_dim],
            coordinate_scale=float(robot_cfg["temporal_rope_coordinate_scale"]),
            rope_base=float(robot_cfg["rope_base"]),
        )
        self.observation_encoder = RobotObservationEncoder(config, state_dim)
        self.blocks = nn.ModuleList(
            [
                SelfAttentionBlock(
                    width=width,
                    heads=heads,
                    ffn_dim=int(robot_cfg["ffn_dim"]),
                    dropout=float(model_cfg["dropout"]),
                    rope=rope,
                )
                for _ in range(int(robot_cfg["transformer_layers"]))
            ]
        )
        history_length = int(config["data"]["history_length"])
        history_stride = int(config["data"]["history_stride"])
        if str(robot_cfg["temporal_rope_coordinate_mode"]) == "relative_steps":
            positions = torch.arange(-(history_length - 1), 1, dtype=torch.float32) * float(history_stride)
        else:
            positions = torch.arange(-(history_length - 1), 1, dtype=torch.float32)
        self.register_buffer("temporal_positions", positions.view(1, history_length, 1), persistent=False)

    def forward(self, robot_visual_history: Tensor, robot_state_history: Tensor, valid_mask: Tensor) -> Tensor:
        values = self.observation_encoder(robot_visual_history, robot_state_history)
        coords = self.temporal_positions.expand(values.shape[0], -1, -1)
        for block in self.blocks:
            values = block(values, valid_mask=valid_mask, coordinates=coords)
        return values


class RobotHumanCrossAttention(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        model_cfg = config["model"]
        cross_cfg = model_cfg["cross_attention"]
        self.blocks = nn.ModuleList(
            [
                CrossAttentionBlock(
                    width=int(model_cfg["width"]),
                    heads=int(cross_cfg["heads"]),
                    ffn_dim=int(cross_cfg["ffn_dim"]),
                    dropout=float(model_cfg["dropout"]),
                )
                for _ in range(int(cross_cfg["layers"]))
            ]
        )

    def forward(self, robot_tokens: Tensor, human_tokens: Tensor, robot_mask: Tensor, human_mask: Tensor) -> Tensor:
        values = robot_tokens
        for block in self.blocks:
            values = block(values, human_tokens, query_mask=robot_mask, context_mask=human_mask)
        return values


class SplineCandidateBuilder(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        width = int(config["model"]["width"])
        phase_bins = int(config["data"]["phase_bin_count"])
        candidates_cfg = config["model"]["candidates"]
        bin_centers = (torch.arange(phase_bins, dtype=torch.float32) + 0.5) / float(phase_bins)
        self.register_buffer("bin_centers", bin_centers, persistent=False)
        self.phase_features = FourierFeatures(1, int(candidates_cfg["phase_fourier_bands"]))
        self.phase_encoder = nn.Sequential(
            nn.Linear(self.phase_features.output_dim(), int(candidates_cfg["phase_hidden_dim"])),
            nn.GELU(),
            nn.Linear(int(candidates_cfg["phase_hidden_dim"]), width),
        )

    def forward(self, human_context: Tensor, basis_200: Tensor) -> Tensor:
        candidates = torch.bmm(basis_200, human_context)
        phase_values = self.bin_centers.view(1, -1, 1).expand(human_context.shape[0], -1, -1)
        phase_embedding = self.phase_encoder(self.phase_features(phase_values))
        return candidates + phase_embedding


class SharedPhaseMatcher(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        matcher_cfg = config["model"]["matcher"]
        width = int(config["model"]["width"])
        dropout = float(config["model"]["dropout"])
        input_dim = 4 * width
        self.network = nn.Sequential(
            nn.Linear(input_dim, int(matcher_cfg["hidden_dim_1"])),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(matcher_cfg["hidden_dim_1"]), int(matcher_cfg["hidden_dim_2"])),
            nn.GELU(),
            nn.Linear(int(matcher_cfg["hidden_dim_2"]), 1),
        )

    def forward(self, robot_token: Tensor, candidate_tokens: Tensor) -> Tensor:
        expanded_robot = robot_token.unsqueeze(1).expand(-1, candidate_tokens.shape[1], -1)
        features = torch.cat(
            [
                expanded_robot,
                candidate_tokens,
                expanded_robot * candidate_tokens,
                torch.abs(expanded_robot - candidate_tokens),
            ],
            dim=-1,
        )
        return self.network(features).squeeze(-1)


class LocalizationHeads(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        width = int(config["model"]["width"])
        interval_cfg = ((config["model"].get("auxiliary") or {}).get("interval_prediction") or {})
        self.interval_prediction_enabled = bool(interval_cfg.get("enabled", False))
        self.min_delta_u = float(interval_cfg.get("min_delta_u", 1.0e-4))
        if self.interval_prediction_enabled:
            hidden_dim = int(interval_cfg.get("hidden_dim", 256))
            self.end_head = nn.Sequential(
                nn.LayerNorm(width),
                nn.Linear(width, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.end_head = None

    def forward(self, robot_token: Tensor, start_u: Tensor) -> tuple[Tensor | None, Tensor | None]:
        if self.end_head is None:
            return None, None
        delta_u_hat = F.softplus(self.end_head(robot_token).squeeze(-1)) + float(self.min_delta_u)
        end_u_hat = torch.clamp(start_u + delta_u_hat, min=0.0, max=1.0)
        return delta_u_hat, end_u_hat


class GlobalSplineLocalizer(nn.Module):
    def __init__(self, config: dict[str, Any], state_dim: int) -> None:
        super().__init__()
        self.config = config
        self.human_transformer = HumanSplineTransformer(config)
        self.robot_transformer = RobotTemporalTransformer(config, state_dim)
        self.cross_attention = RobotHumanCrossAttention(config)
        self.candidate_builder = SplineCandidateBuilder(config)
        self.matcher = SharedPhaseMatcher(config)
        self.localization_heads = LocalizationHeads(config)
        self.expectation_window_radius = int(config["model"]["candidates"]["expectation_window_radius"])
        phase_bin_count = int(config["data"]["phase_bin_count"])
        self.register_buffer(
            "bin_centers",
            (torch.arange(phase_bin_count, dtype=torch.float32) + 0.5) / float(phase_bin_count),
            persistent=False,
        )

    def _estimate_u(self, probabilities: Tensor) -> Tensor:
        radius = int(self.expectation_window_radius)
        max_indices = probabilities.argmax(dim=-1)
        candidate_indices = torch.arange(probabilities.shape[1], device=probabilities.device).view(1, -1)
        window_mask = (candidate_indices >= (max_indices[:, None] - radius)) & (candidate_indices <= (max_indices[:, None] + radius))
        local = probabilities * window_mask.to(probabilities.dtype)
        local = local / torch.clamp(local.sum(dim=-1, keepdim=True), min=1e-12)
        return (local * self.bin_centers.view(1, -1)).sum(dim=-1)

    def _top_two_margin(self, probabilities: Tensor) -> Tensor:
        top2 = torch.topk(probabilities, k=2, dim=-1).values
        return top2[:, 0] - top2[:, 1]

    def forward(
        self,
        robot_history_embeddings: Tensor,
        robot_history_states: Tensor,
        robot_history_mask: Tensor,
        human_coefficients: Tensor,
        human_left_support: Tensor,
        human_right_support: Tensor,
        human_support_midpoint: Tensor,
        human_support_width: Tensor,
        human_greville_phase: Tensor,
        human_basis_200: Tensor,
        human_mask: Tensor,
    ) -> dict[str, Tensor]:
        human_context = self.human_transformer(
            human_coefficients,
            human_support_width,
            human_left_support,
            human_right_support,
            human_support_midpoint,
            human_greville_phase,
            human_mask,
        )
        robot_context = self.robot_transformer(
            robot_history_embeddings,
            robot_history_states,
            robot_history_mask,
        )
        fused_robot = self.cross_attention(robot_context, human_context, robot_history_mask, human_mask)
        current_robot_token = fused_robot[:, -1]
        candidate_tokens = self.candidate_builder(human_context, human_basis_200)
        logits = self.matcher(current_robot_token, candidate_tokens)
        probabilities = torch.softmax(logits, dim=-1)
        u_hat = self._estimate_u(probabilities)
        delta_u_hat, u_end_hat = self.localization_heads(current_robot_token, u_hat)
        entropy = -(probabilities * torch.log(torch.clamp(probabilities, min=1e-12))).sum(dim=-1)
        outputs = {
            "logits": logits,
            "probabilities": probabilities,
            "u_hat": u_hat,
            "c_max": probabilities.max(dim=-1).values,
            "entropy": entropy,
            "margin": self._top_two_margin(probabilities),
            "robot_token": current_robot_token,
        }
        if delta_u_hat is not None and u_end_hat is not None:
            outputs["delta_u_hat"] = delta_u_hat
            outputs["u_end_hat"] = u_end_hat
        return outputs
