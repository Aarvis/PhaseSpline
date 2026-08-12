from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from .blocks import ContinuousMultiDimRoPE, CrossAttentionBlock, FourierFeatures, SelfAttentionBlock
from .spline_math import (
    build_clamped_knot_vector_from_spans,
    build_local_human_spline_batch,
    finite_difference_first,
    finite_difference_second,
    make_phase_grid,
    solve_weighted_control_points,
)


@dataclass(frozen=True)
class PredictedRobotSpline:
    knots: Tensor
    coefficients: Tensor
    degree: int


class HumanCoefficientEncoder(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        model_cfg = config["model"]
        human_cfg = model_cfg["human"]
        width = int(model_cfg["width"])
        self.coefficient_projection = nn.Linear(
            int(human_cfg["coefficient_input_dim"]),
            int(human_cfg["coefficient_projection_dim"]),
        )
        self.width_features = FourierFeatures(1, int(human_cfg["width_fourier_bands"]))
        self.width_projection = nn.Sequential(
            nn.Linear(self.width_features.output_dim(), int(human_cfg["width_hidden_dim"])),
            nn.GELU(),
            nn.Linear(int(human_cfg["width_hidden_dim"]), width),
        )
        self.human_type = nn.Parameter(torch.zeros((1, 1, width)))
        self.output_norm = nn.LayerNorm(width)

    def forward(self, coefficients: Tensor, support_width: Tensor) -> Tensor:
        coeff = self.coefficient_projection(coefficients)
        width_values = self.width_features(support_width.unsqueeze(-1))
        width_embed = self.width_projection(width_values)
        return self.output_norm(coeff + width_embed + self.human_type)


class HumanCoefficientTransformer(nn.Module):
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
                    activation="gelu",
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


class HumanPhaseFusion(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        width = int(config["model"]["width"])
        self.norm_position = nn.LayerNorm(width)
        self.norm_velocity = nn.LayerNorm(width)
        self.norm_acceleration = nn.LayerNorm(width)
        self.fusion = nn.Sequential(
            nn.Linear(width * 3, int(config["model"]["human_phase"]["fusion_hidden_dim"])),
            nn.GELU(),
            nn.Linear(int(config["model"]["human_phase"]["fusion_hidden_dim"]), width),
        )
        self.output_norm = nn.LayerNorm(width)

    def forward(self, position: Tensor, velocity: Tensor, acceleration: Tensor) -> Tensor:
        fused = torch.cat(
            [
                self.norm_position(position),
                self.norm_velocity(velocity),
                self.norm_acceleration(acceleration),
            ],
            dim=-1,
        )
        return self.output_norm(position + self.fusion(fused))


class HumanPhaseTransformer(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        model_cfg = config["model"]
        phase_cfg = model_cfg["human_phase"]
        width = int(model_cfg["width"])
        heads = int(phase_cfg["heads"])
        head_dim = width // heads
        rope = ContinuousMultiDimRoPE(
            head_dim=head_dim,
            coordinate_dims=[head_dim],
            coordinate_scale=float(phase_cfg["rope_coordinate_scale"]),
            rope_base=float(phase_cfg["rope_base"]),
        )
        self.blocks = nn.ModuleList(
            [
                SelfAttentionBlock(
                    width=width,
                    heads=heads,
                    ffn_dim=int(phase_cfg["ffn_dim"]),
                    dropout=float(model_cfg["dropout"]),
                    rope=rope,
                    activation="gelu",
                )
                for _ in range(int(phase_cfg["transformer_layers"]))
            ]
        )

    def forward(self, values: Tensor, phase_grid: Tensor) -> Tensor:
        coordinates = phase_grid.view(1, -1, 1).expand(values.shape[0], -1, -1)
        valid_mask = torch.ones((values.shape[0], values.shape[1]), device=values.device, dtype=torch.bool)
        for block in self.blocks:
            values = block(values, valid_mask=valid_mask, coordinates=coordinates)
        return values


class RobotObservationEncoder(nn.Module):
    def __init__(self, config: dict[str, Any], state_dim: int) -> None:
        super().__init__()
        model_cfg = config["model"]
        robot_cfg = model_cfg["robot"]
        width = int(model_cfg["width"])
        self.visual_projection = nn.Linear(int(robot_cfg["visual_input_dim"]), int(robot_cfg["visual_projection_dim"]))
        self.state_projection = nn.Sequential(
            nn.Linear(state_dim, int(robot_cfg["state_hidden_dim"])),
            nn.SiLU(),
            nn.Linear(int(robot_cfg["state_hidden_dim"]), int(robot_cfg["state_projection_dim"])),
        )
        self.fusion = nn.Sequential(
            nn.Linear(width, int(robot_cfg["fusion_hidden_dim"])),
            nn.SiLU(),
            nn.Linear(int(robot_cfg["fusion_hidden_dim"]), width),
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
        self.encoder = RobotObservationEncoder(config, state_dim)
        self.blocks = nn.ModuleList(
            [
                SelfAttentionBlock(
                    width=width,
                    heads=heads,
                    ffn_dim=int(robot_cfg["ffn_dim"]),
                    dropout=float(model_cfg["dropout"]),
                    rope=rope,
                    activation="gelu",
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
        values = self.encoder(robot_visual_history, robot_state_history)
        coordinates = self.temporal_positions.expand(values.shape[0], -1, -1)
        for block in self.blocks:
            values = block(values, valid_mask=valid_mask, coordinates=coordinates)
        return values


class HumanRobotTranslator(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        model_cfg = config["model"]
        cross_cfg = model_cfg["translator"]
        self.blocks = nn.ModuleList(
            [
                CrossAttentionBlock(
                    width=int(model_cfg["width"]),
                    heads=int(cross_cfg["heads"]),
                    ffn_dim=int(cross_cfg["ffn_dim"]),
                    dropout=float(model_cfg["dropout"]),
                    activation="gelu",
                )
                for _ in range(int(cross_cfg["layers"]))
            ]
        )

    def forward(self, human_phase_tokens: Tensor, robot_history_tokens: Tensor, robot_history_mask: Tensor) -> Tensor:
        valid_mask = torch.ones((human_phase_tokens.shape[0], human_phase_tokens.shape[1]), device=human_phase_tokens.device, dtype=torch.bool)
        values = human_phase_tokens
        for block in self.blocks:
            values = block(values, robot_history_tokens, query_mask=valid_mask, context_mask=robot_history_mask)
        return values


class DenseRobotTrajectoryHead(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        width = int(config["model"]["width"])
        hidden_dim = int(config["model"]["dense_head"]["hidden_dim"])
        output_dim = int(config["model"]["dense_head"]["output_dim"])
        self.network = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, values: Tensor) -> Tensor:
        return self.network(values)


class CompressorTokenizer(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        tokenizer_cfg = config["model"]["compressor_tokenizer"]
        self.position_norm = nn.LayerNorm(int(tokenizer_cfg["input_dim"]))
        self.velocity_norm = nn.LayerNorm(int(tokenizer_cfg["input_dim"]))
        self.acceleration_norm = nn.LayerNorm(int(tokenizer_cfg["input_dim"]))
        self.position_proj = nn.Linear(int(tokenizer_cfg["input_dim"]), int(tokenizer_cfg["position_projection_dim"]))
        self.velocity_proj = nn.Linear(int(tokenizer_cfg["input_dim"]), int(tokenizer_cfg["velocity_projection_dim"]))
        self.acceleration_proj = nn.Linear(int(tokenizer_cfg["input_dim"]), int(tokenizer_cfg["acceleration_projection_dim"]))
        fused_dim = (
            int(tokenizer_cfg["position_projection_dim"])
            + int(tokenizer_cfg["velocity_projection_dim"])
            + int(tokenizer_cfg["acceleration_projection_dim"])
        )
        width = int(config["model"]["width"])
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, width),
            nn.GELU(),
            nn.Linear(width, width),
        )
        self.base = nn.Linear(int(tokenizer_cfg["input_dim"]), width)
        self.output_norm = nn.LayerNorm(width)

    def forward(self, position: Tensor, velocity: Tensor, acceleration: Tensor) -> Tensor:
        pos = self.position_proj(self.position_norm(position))
        vel = self.velocity_proj(self.velocity_norm(velocity))
        acc = self.acceleration_proj(self.acceleration_norm(acceleration))
        fused = self.fusion(torch.cat([pos, vel, acc], dim=-1))
        return self.output_norm(self.base(position) + fused)


class CompressorTransformer(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        model_cfg = config["model"]
        comp_cfg = model_cfg["compressor_transformer"]
        width = int(model_cfg["width"])
        heads = int(comp_cfg["heads"])
        head_dim = width // heads
        rope = ContinuousMultiDimRoPE(
            head_dim=head_dim,
            coordinate_dims=[head_dim],
            coordinate_scale=float(comp_cfg["rope_coordinate_scale"]),
            rope_base=float(comp_cfg["rope_base"]),
        )
        self.blocks = nn.ModuleList(
            [
                SelfAttentionBlock(
                    width=width,
                    heads=heads,
                    ffn_dim=int(comp_cfg["ffn_dim"]),
                    dropout=float(model_cfg["dropout"]),
                    rope=rope,
                    activation="gelu",
                )
                for _ in range(int(comp_cfg["transformer_layers"]))
            ]
        )

    def forward(self, values: Tensor, phase_grid: Tensor) -> Tensor:
        coordinates = phase_grid.view(1, -1, 1).expand(values.shape[0], -1, -1)
        valid_mask = torch.ones((values.shape[0], values.shape[1]), device=values.device, dtype=torch.bool)
        for block in self.blocks:
            values = block(values, valid_mask=valid_mask, coordinates=coordinates)
        return values


class SpanQueryDecoder(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        model_cfg = config["model"]
        decoder_cfg = model_cfg["span_decoder"]
        self.num_output_spans = int(decoder_cfg["num_output_spans"])
        self.min_span_width = float(decoder_cfg["min_span_width"])
        width = int(model_cfg["width"])
        heads = int(decoder_cfg["heads"])
        head_dim = width // heads
        rope = ContinuousMultiDimRoPE(
            head_dim=head_dim,
            coordinate_dims=[head_dim],
            coordinate_scale=float(decoder_cfg["rope_coordinate_scale"]),
            rope_base=float(decoder_cfg["rope_base"]),
        )
        self.learned_queries = nn.Parameter(torch.zeros((1, self.num_output_spans, width)))
        self.position_features = FourierFeatures(1, int(decoder_cfg["position_fourier_bands"]))
        self.position_projection = nn.Sequential(
            nn.Linear(self.position_features.output_dim(), int(decoder_cfg["position_hidden_dim"])),
            nn.GELU(),
            nn.Linear(int(decoder_cfg["position_hidden_dim"]), width),
        )
        self.self_blocks = nn.ModuleList(
            [
                SelfAttentionBlock(
                    width=width,
                    heads=heads,
                    ffn_dim=int(decoder_cfg["ffn_dim"]),
                    dropout=float(model_cfg["dropout"]),
                    rope=rope,
                    activation="gelu",
                )
                for _ in range(int(decoder_cfg["layers"]))
            ]
        )
        self.cross_blocks = nn.ModuleList(
            [
                CrossAttentionBlock(
                    width=width,
                    heads=heads,
                    ffn_dim=int(decoder_cfg["ffn_dim"]),
                    dropout=float(model_cfg["dropout"]),
                    activation="gelu",
                )
                for _ in range(int(decoder_cfg["layers"]))
            ]
        )
        self.width_head = nn.Linear(width, 1)
        nominal_positions = (torch.arange(self.num_output_spans, dtype=torch.float32) + 0.5) / float(self.num_output_spans)
        self.register_buffer("nominal_positions", nominal_positions.view(1, self.num_output_spans, 1), persistent=False)

    def forward(self, compressor_tokens: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        batch_size = compressor_tokens.shape[0]
        queries = self.learned_queries.expand(batch_size, -1, -1)
        queries = queries + self.position_projection(self.position_features(self.nominal_positions.expand(batch_size, -1, -1)))
        query_mask = torch.ones((batch_size, self.num_output_spans), device=compressor_tokens.device, dtype=torch.bool)
        context_mask = torch.ones((batch_size, compressor_tokens.shape[1]), device=compressor_tokens.device, dtype=torch.bool)
        for self_block, cross_block in zip(self.self_blocks, self.cross_blocks):
            queries = self_block(queries, valid_mask=query_mask, coordinates=self.nominal_positions.expand(batch_size, -1, -1))
            queries = cross_block(queries, compressor_tokens, query_mask=query_mask, context_mask=context_mask)
        logits = self.width_head(queries).squeeze(-1)
        probabilities = torch.softmax(logits, dim=-1)
        if self.min_span_width > 0.0:
            widths = self.min_span_width + (1.0 - self.num_output_spans * self.min_span_width) * probabilities
        else:
            widths = probabilities
        entropy = -(probabilities * torch.log(torch.clamp(probabilities, min=1e-12))).sum(dim=-1)
        return widths, logits, entropy


class LocalHumanToRobotSplineModel(nn.Module):
    def __init__(self, config: dict[str, Any], state_dim: int) -> None:
        super().__init__()
        self.config = config
        self.degree = int(config["model"]["degree"])
        self.phase_count = int(config["data"]["local_phase_grid_size"])
        self.phase_grid_endpoint_inclusive = bool(config["data"]["phase_grid_endpoint_inclusive"])
        phase_grid = make_phase_grid(self.phase_count, endpoint_inclusive=self.phase_grid_endpoint_inclusive)
        self.register_buffer("phase_grid", phase_grid, persistent=False)
        self.human_coefficient_transformer = HumanCoefficientTransformer(config)
        self.human_phase_fusion = HumanPhaseFusion(config)
        self.human_phase_transformer = HumanPhaseTransformer(config)
        self.robot_history_transformer = RobotTemporalTransformer(config, state_dim)
        self.human_robot_translator = HumanRobotTranslator(config)
        self.dense_head = DenseRobotTrajectoryHead(config)
        self.compressor_tokenizer = CompressorTokenizer(config)
        self.compressor_transformer = CompressorTransformer(config)
        self.span_decoder = SpanQueryDecoder(config)
        self.num_output_spans = int(config["model"]["span_decoder"]["num_output_spans"])
        self.projection_eta_velocity = float(config["model"]["projection"]["eta_velocity"])
        self.projection_eta_acceleration = float(config["model"]["projection"]["eta_acceleration"])
        self.projection_ridge = float(config["model"]["projection"]["ridge"])
        self.debug_return_basis = bool(config["model"].get("debug_return_basis", True))

    def forward(
        self,
        robot_history_embeddings: Tensor,
        robot_history_states: Tensor,
        robot_history_mask: Tensor,
        human_global_coefficients: Tensor,
        human_global_knots: Tensor,
        human_global_coeff_counts: Tensor,
        human_global_knot_counts: Tensor,
        human_input_start_u: Tensor,
        human_input_end_u: Tensor,
        dense_robot_teacher: Tensor | None = None,
        teacher_forcing_alpha: float = 0.0,
        compressor_gradient_gamma: float = 0.0,
    ) -> dict[str, Tensor]:
        local_human = build_local_human_spline_batch(
            human_global_coefficients=human_global_coefficients,
            human_global_knots=human_global_knots,
            human_global_coeff_counts=human_global_coeff_counts,
            human_global_knot_counts=human_global_knot_counts,
            interval_start_u=human_input_start_u,
            interval_end_u=human_input_end_u,
            phase_grid=self.phase_grid,
            degree=self.degree,
        )
        human_coeff_context = self.human_coefficient_transformer(
            coefficients=local_human.coefficients,
            support_width=local_human.support_width,
            left_support=local_human.left_support,
            right_support=local_human.right_support,
            support_midpoint=local_human.support_midpoint,
            greville_phase=local_human.greville_phase,
            valid_mask=local_human.mask,
        )
        # Adaptive spline intervals can make the derivative bases large enough
        # to overflow float16 during autocast, especially for the second
        # derivative. Keep these reductions in float32; downstream layer norms
        # make their scale safe for the remainder of the mixed-precision model.
        with torch.autocast(device_type=human_coeff_context.device.type, enabled=False):
            human_coeff_context32 = human_coeff_context.to(dtype=torch.float32)
            human_position = torch.bmm(local_human.basis.to(dtype=torch.float32), human_coeff_context32)
            human_velocity = torch.bmm(local_human.basis_d1.to(dtype=torch.float32), human_coeff_context32)
            human_acceleration = torch.bmm(local_human.basis_d2.to(dtype=torch.float32), human_coeff_context32)
        human_phase_tokens = self.human_phase_fusion(human_position, human_velocity, human_acceleration)
        human_phase_tokens = self.human_phase_transformer(human_phase_tokens, self.phase_grid)

        robot_history_tokens = self.robot_history_transformer(robot_history_embeddings, robot_history_states, robot_history_mask)
        translated_phase_tokens = self.human_robot_translator(human_phase_tokens, robot_history_tokens, robot_history_mask)
        dense_robot_pred = self.dense_head(translated_phase_tokens)
        step = float(self.phase_grid[1].item() - self.phase_grid[0].item()) if self.phase_grid.numel() > 1 else 1.0
        dense_robot_velocity = finite_difference_first(dense_robot_pred, step)
        dense_robot_acceleration = finite_difference_second(dense_robot_pred, step)

        translated_for_compressor = dense_robot_pred.detach() + float(compressor_gradient_gamma) * (dense_robot_pred - dense_robot_pred.detach())
        if dense_robot_teacher is not None:
            dense_compressor_input = (1.0 - float(teacher_forcing_alpha)) * dense_robot_teacher + float(teacher_forcing_alpha) * translated_for_compressor
        else:
            dense_compressor_input = translated_for_compressor
        dense_compressor_velocity = finite_difference_first(dense_compressor_input, step)
        dense_compressor_acceleration = finite_difference_second(dense_compressor_input, step)

        compressor_tokens = self.compressor_tokenizer(dense_compressor_input, dense_compressor_velocity, dense_compressor_acceleration)
        compressor_tokens = self.compressor_transformer(compressor_tokens, self.phase_grid)
        predicted_span_widths, predicted_span_logits, span_entropy = self.span_decoder(compressor_tokens)
        predicted_knots = build_clamped_knot_vector_from_spans(predicted_span_widths, self.degree)
        (
            predicted_coefficients,
            predicted_basis,
            predicted_basis_d1,
            predicted_basis_d2,
            reconstructed_robot_curve,
            reconstructed_robot_velocity,
            reconstructed_robot_acceleration,
            projection_condition_proxy,
        ) = solve_weighted_control_points(
            predicted_knots=predicted_knots,
            dense_position=dense_compressor_input,
            dense_velocity=dense_compressor_velocity,
            dense_acceleration=dense_compressor_acceleration,
            phase_grid=self.phase_grid,
            degree=self.degree,
            eta_velocity=self.projection_eta_velocity,
            eta_acceleration=self.projection_eta_acceleration,
            ridge=self.projection_ridge,
        )

        outputs = {
            "dense_robot_pred": dense_robot_pred,
            "dense_robot_velocity": dense_robot_velocity,
            "dense_robot_acceleration": dense_robot_acceleration,
            "dense_compressor_input": dense_compressor_input,
            "dense_compressor_velocity": dense_compressor_velocity,
            "dense_compressor_acceleration": dense_compressor_acceleration,
            "predicted_span_widths": predicted_span_widths,
            "predicted_span_logits": predicted_span_logits,
            "predicted_knots": predicted_knots,
            "predicted_coefficients": predicted_coefficients,
            "reconstructed_robot_curve": reconstructed_robot_curve,
            "reconstructed_robot_velocity": reconstructed_robot_velocity,
            "reconstructed_robot_acceleration": reconstructed_robot_acceleration,
            "span_entropy": span_entropy,
            "projection_condition_proxy": projection_condition_proxy,
            "predicted_robot_spline": PredictedRobotSpline(knots=predicted_knots, coefficients=predicted_coefficients, degree=self.degree),
            "human_local_coefficient_count": local_human.mask.sum(dim=-1).to(torch.float32),
        }
        if self.debug_return_basis:
            outputs["predicted_basis"] = predicted_basis
            outputs["predicted_basis_d1"] = predicted_basis_d1
            outputs["predicted_basis_d2"] = predicted_basis_d2
        return outputs
