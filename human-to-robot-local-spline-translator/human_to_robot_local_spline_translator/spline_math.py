from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


EPSILON = 1.0e-8


@dataclass(frozen=True)
class LocalHumanSplineBatch:
    coefficients: Tensor
    left_support: Tensor
    right_support: Tensor
    support_midpoint: Tensor
    support_width: Tensor
    greville_phase: Tensor
    basis: Tensor
    basis_d1: Tensor
    basis_d2: Tensor
    mask: Tensor


def make_phase_grid(count: int, endpoint_inclusive: bool = True, device: torch.device | None = None, dtype: torch.dtype = torch.float32) -> Tensor:
    count = int(count)
    if count <= 1:
        return torch.zeros((max(1, count),), device=device, dtype=dtype)
    if endpoint_inclusive:
        return torch.linspace(0.0, 1.0, count, device=device, dtype=dtype)
    return torch.arange(count, device=device, dtype=dtype) / float(count)


def _finite_difference_compute_dtype(values: Tensor) -> torch.dtype:
    if values.dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return values.dtype


def finite_difference_first(values: Tensor, step: float) -> Tensor:
    if values.ndim != 3:
        raise ValueError(f"Expected [B,M,D], got {values.shape}")
    compute_dtype = _finite_difference_compute_dtype(values)
    work = values.to(dtype=compute_dtype)
    if work.shape[1] < 2:
        return torch.zeros_like(work)
    step = float(step)
    output = torch.zeros_like(work)
    if work.shape[1] == 2:
        output[:, 0] = (work[:, 1] - work[:, 0]) / step
        output[:, 1] = output[:, 0]
        return output
    output[:, 0] = (-3.0 * work[:, 0] + 4.0 * work[:, 1] - work[:, 2]) / (2.0 * step)
    output[:, -1] = (3.0 * work[:, -1] - 4.0 * work[:, -2] + work[:, -3]) / (2.0 * step)
    output[:, 1:-1] = (work[:, 2:] - work[:, :-2]) / (2.0 * step)
    return output


def finite_difference_second(values: Tensor, step: float) -> Tensor:
    if values.ndim != 3:
        raise ValueError(f"Expected [B,M,D], got {values.shape}")
    compute_dtype = _finite_difference_compute_dtype(values)
    work = values.to(dtype=compute_dtype)
    if work.shape[1] < 3:
        return torch.zeros_like(work)
    step_sq = float(step) ** 2
    output = torch.zeros_like(work)
    if work.shape[1] == 3:
        second = (work[:, 2] - 2.0 * work[:, 1] + work[:, 0]) / step_sq
        output[:] = second.unsqueeze(1)
        return output
    output[:, 0] = (2.0 * work[:, 0] - 5.0 * work[:, 1] + 4.0 * work[:, 2] - work[:, 3]) / step_sq
    output[:, -1] = (2.0 * work[:, -1] - 5.0 * work[:, -2] + 4.0 * work[:, -3] - work[:, -4]) / step_sq
    output[:, 1:-1] = (work[:, 2:] - 2.0 * work[:, 1:-1] + work[:, :-2]) / step_sq
    return output


def coefficient_support_geometry(knots: Tensor, control_count: int, degree: int) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    if int(degree) != 3:
        raise ValueError(f"Expected cubic splines, got degree={degree}")
    control_count = int(control_count)
    left = knots[:control_count]
    right = knots[degree + 1 : degree + 1 + control_count]
    midpoint = 0.5 * (left + right)
    width = right - left
    greville = (knots[1 : control_count + 1] + knots[2 : control_count + 2] + knots[3 : control_count + 3]) / 3.0
    return left, right, midpoint, width, greville


def global_u_to_local_rho(global_u: Tensor, interval_start: Tensor, interval_end: Tensor) -> Tensor:
    width = torch.clamp(interval_end - interval_start, min=EPSILON)
    return (global_u - interval_start) / width


def local_rho_to_global_u(local_rho: Tensor, interval_start: Tensor, interval_end: Tensor) -> Tensor:
    return interval_start + local_rho * (interval_end - interval_start)


def _basis_degree_zero(knots: Tensor, u: Tensor) -> Tensor:
    left = knots[:-1].unsqueeze(1)
    right = knots[1:].unsqueeze(1)
    phase = u.unsqueeze(0)
    basis = ((left <= phase) & (phase < right)).to(u.dtype)
    endpoint_mask = torch.isclose(u, knots[-1], atol=1e-7, rtol=0.0)
    if torch.any(endpoint_mask):
        positive_spans = torch.nonzero((knots[1:] - knots[:-1]) > EPSILON, as_tuple=False).squeeze(-1)
        if positive_spans.numel() > 0:
            last_span = int(positive_spans[-1].item())
            basis[:, endpoint_mask] = 0.0
            basis[last_span, endpoint_mask] = 1.0
    return basis


def bspline_basis_and_derivatives_single(knots: Tensor, u: Tensor, degree: int) -> tuple[Tensor, Tensor, Tensor]:
    if knots.ndim != 1 or u.ndim != 1:
        raise ValueError("knots and u must be rank-1 tensors")
    degree = int(degree)
    base = _basis_degree_zero(knots, u)
    d1_prev = torch.zeros_like(base)
    d2_prev = torch.zeros_like(base)
    basis_prev = base
    for p in range(1, degree + 1):
        row_count = int(knots.shape[0] - p - 1)
        basis_curr = []
        d1_curr = []
        d2_curr = []
        for i in range(row_count):
            left_den = knots[i + p] - knots[i]
            right_den = knots[i + p + 1] - knots[i + 1]
            left_valid = torch.abs(left_den) > EPSILON
            right_valid = torch.abs(right_den) > EPSILON
            safe_left = torch.where(left_valid, left_den, torch.ones_like(left_den))
            safe_right = torch.where(right_valid, right_den, torch.ones_like(right_den))
            a = torch.where(left_valid, (u - knots[i]) / safe_left, torch.zeros_like(u))
            b = torch.where(right_valid, (knots[i + p + 1] - u) / safe_right, torch.zeros_like(u))
            inv_left = torch.where(left_valid, 1.0 / safe_left, torch.zeros_like(safe_left))
            inv_right = torch.where(right_valid, -1.0 / safe_right, torch.zeros_like(safe_right))
            current_basis = a * basis_prev[i] + b * basis_prev[i + 1]
            current_d1 = inv_left * basis_prev[i] + a * d1_prev[i] + inv_right * basis_prev[i + 1] + b * d1_prev[i + 1]
            current_d2 = 2.0 * inv_left * d1_prev[i] + a * d2_prev[i] + 2.0 * inv_right * d1_prev[i + 1] + b * d2_prev[i + 1]
            basis_curr.append(current_basis)
            d1_curr.append(current_d1)
            d2_curr.append(current_d2)
        basis_prev = torch.stack(basis_curr, dim=0)
        d1_prev = torch.stack(d1_curr, dim=0)
        d2_prev = torch.stack(d2_curr, dim=0)
    return basis_prev.transpose(0, 1), d1_prev.transpose(0, 1), d2_prev.transpose(0, 1)


def evaluate_spline_with_derivatives_single(knots: Tensor, coefficients: Tensor, u: Tensor, degree: int) -> tuple[Tensor, Tensor, Tensor]:
    basis, d1_basis, d2_basis = bspline_basis_and_derivatives_single(knots, u, degree)
    position = basis @ coefficients
    d1 = d1_basis @ coefficients
    d2 = d2_basis @ coefficients
    return position, d1, d2


def build_local_human_spline_batch(
    human_global_coefficients: Tensor,
    human_global_knots: Tensor,
    human_global_coeff_counts: Tensor,
    human_global_knot_counts: Tensor,
    interval_start_u: Tensor,
    interval_end_u: Tensor,
    phase_grid: Tensor,
    degree: int,
) -> LocalHumanSplineBatch:
    batch_size = int(human_global_coefficients.shape[0])
    coeff_lists: list[Tensor] = []
    left_lists: list[Tensor] = []
    right_lists: list[Tensor] = []
    mid_lists: list[Tensor] = []
    width_lists: list[Tensor] = []
    greville_lists: list[Tensor] = []
    basis_lists: list[Tensor] = []
    d1_lists: list[Tensor] = []
    d2_lists: list[Tensor] = []
    max_local_coeffs = 1

    for batch_index in range(batch_size):
        coeff_count = int(human_global_coeff_counts[batch_index].item())
        knot_count = int(human_global_knot_counts[batch_index].item())
        global_coeffs = human_global_coefficients[batch_index, :coeff_count]
        global_knots = human_global_knots[batch_index, :knot_count]
        start_u = torch.clamp(interval_start_u[batch_index], min=0.0, max=1.0 - 1e-6)
        end_u = torch.clamp(interval_end_u[batch_index], min=float(start_u.item()) + 1e-6, max=1.0)
        left, right, midpoint, support_width, greville = coefficient_support_geometry(global_knots, coeff_count, degree)
        overlap = (right > start_u) & (left < end_u)
        if not torch.any(overlap):
            overlap[0] = True
        retained_indices = torch.nonzero(overlap, as_tuple=False).squeeze(-1)
        width = torch.clamp(end_u - start_u, min=1e-6)
        local_left = (left[retained_indices] - start_u) / width
        local_right = (right[retained_indices] - start_u) / width
        local_mid = (midpoint[retained_indices] - start_u) / width
        local_width = support_width[retained_indices] / width
        local_greville = (greville[retained_indices] - start_u) / width
        global_u = local_rho_to_global_u(phase_grid, start_u, end_u)
        basis, d1_u, d2_u = bspline_basis_and_derivatives_single(global_knots, global_u, degree)
        basis = basis[:, retained_indices]
        d1_rho = d1_u[:, retained_indices] * width
        d2_rho = d2_u[:, retained_indices] * (width ** 2)
        local_coeffs = global_coeffs[retained_indices]
        coeff_lists.append(local_coeffs)
        left_lists.append(local_left)
        right_lists.append(local_right)
        mid_lists.append(local_mid)
        width_lists.append(local_width)
        greville_lists.append(local_greville)
        basis_lists.append(basis)
        d1_lists.append(d1_rho)
        d2_lists.append(d2_rho)
        max_local_coeffs = max(max_local_coeffs, int(local_coeffs.shape[0]))

    feature_dim = int(human_global_coefficients.shape[-1])
    phase_count = int(phase_grid.shape[0])
    coefficients = human_global_coefficients.new_zeros((batch_size, max_local_coeffs, feature_dim))
    left_support = human_global_coefficients.new_zeros((batch_size, max_local_coeffs))
    right_support = human_global_coefficients.new_zeros((batch_size, max_local_coeffs))
    support_midpoint = human_global_coefficients.new_zeros((batch_size, max_local_coeffs))
    support_width = human_global_coefficients.new_zeros((batch_size, max_local_coeffs))
    greville_phase = human_global_coefficients.new_zeros((batch_size, max_local_coeffs))
    basis = human_global_coefficients.new_zeros((batch_size, phase_count, max_local_coeffs))
    basis_d1 = human_global_coefficients.new_zeros((batch_size, phase_count, max_local_coeffs))
    basis_d2 = human_global_coefficients.new_zeros((batch_size, phase_count, max_local_coeffs))
    mask = torch.zeros((batch_size, max_local_coeffs), device=human_global_coefficients.device, dtype=torch.bool)

    for batch_index in range(batch_size):
        local_count = int(coeff_lists[batch_index].shape[0])
        coefficients[batch_index, :local_count] = coeff_lists[batch_index]
        left_support[batch_index, :local_count] = left_lists[batch_index]
        right_support[batch_index, :local_count] = right_lists[batch_index]
        support_midpoint[batch_index, :local_count] = mid_lists[batch_index]
        support_width[batch_index, :local_count] = width_lists[batch_index]
        greville_phase[batch_index, :local_count] = greville_lists[batch_index]
        basis[batch_index, :, :local_count] = basis_lists[batch_index]
        basis_d1[batch_index, :, :local_count] = d1_lists[batch_index]
        basis_d2[batch_index, :, :local_count] = d2_lists[batch_index]
        mask[batch_index, :local_count] = True

    return LocalHumanSplineBatch(
        coefficients=coefficients,
        left_support=left_support,
        right_support=right_support,
        support_midpoint=support_midpoint,
        support_width=support_width,
        greville_phase=greville_phase,
        basis=basis,
        basis_d1=basis_d1,
        basis_d2=basis_d2,
        mask=mask,
    )


def evaluate_global_spline_interval_batch(
    global_coefficients: Tensor,
    global_knots: Tensor,
    coeff_counts: Tensor,
    knot_counts: Tensor,
    interval_start_u: Tensor,
    interval_end_u: Tensor,
    phase_grid: Tensor,
    degree: int,
) -> tuple[Tensor, Tensor, Tensor]:
    batch_size = int(global_coefficients.shape[0])
    feature_dim = int(global_coefficients.shape[-1])
    phase_count = int(phase_grid.shape[0])
    positions = global_coefficients.new_zeros((batch_size, phase_count, feature_dim))
    velocities = global_coefficients.new_zeros((batch_size, phase_count, feature_dim))
    accelerations = global_coefficients.new_zeros((batch_size, phase_count, feature_dim))
    for batch_index in range(batch_size):
        coeff_count = int(coeff_counts[batch_index].item())
        knot_count = int(knot_counts[batch_index].item())
        coeffs = global_coefficients[batch_index, :coeff_count]
        knots = global_knots[batch_index, :knot_count]
        start_u = interval_start_u[batch_index]
        end_u = interval_end_u[batch_index]
        width = torch.clamp(end_u - start_u, min=1e-6)
        global_u = local_rho_to_global_u(phase_grid, start_u, end_u)
        position, d1_u, d2_u = evaluate_spline_with_derivatives_single(knots, coeffs, global_u, degree)
        positions[batch_index] = position
        velocities[batch_index] = d1_u * width
        accelerations[batch_index] = d2_u * (width ** 2)
    return positions, velocities, accelerations


def build_clamped_knot_vector_from_spans(span_widths: Tensor, degree: int) -> Tensor:
    batch_size, num_spans = span_widths.shape
    cumulative = torch.cumsum(span_widths, dim=-1)
    interior = cumulative[:, :-1]
    zeros = span_widths.new_zeros((batch_size, degree + 1))
    ones = span_widths.new_ones((batch_size, degree + 1))
    return torch.cat([zeros, interior, ones], dim=-1)


def derive_gt_span_widths_batch(
    padded_local_knots: Tensor,
    local_knot_counts: Tensor,
    expected_num_spans: int,
) -> tuple[Tensor, Tensor]:
    batch_size = int(padded_local_knots.shape[0])
    output = padded_local_knots.new_zeros((batch_size, expected_num_spans))
    valid = torch.zeros((batch_size,), device=padded_local_knots.device, dtype=torch.bool)
    for batch_index in range(batch_size):
        knot_count = int(local_knot_counts[batch_index].item())
        knots = padded_local_knots[batch_index, :knot_count]
        unique_knots = torch.unique_consecutive(knots)
        if unique_knots.numel() < 2:
            continue
        widths = unique_knots[1:] - unique_knots[:-1]
        widths = widths[widths > EPSILON]
        if int(widths.numel()) != int(expected_num_spans):
            continue
        total = torch.clamp(widths.sum(), min=EPSILON)
        output[batch_index] = widths / total
        valid[batch_index] = True
    return output, valid


def solve_weighted_control_points(
    predicted_knots: Tensor,
    dense_position: Tensor,
    dense_velocity: Tensor,
    dense_acceleration: Tensor,
    phase_grid: Tensor,
    degree: int,
    eta_velocity: float,
    eta_acceleration: float,
    ridge: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    batch_size = int(predicted_knots.shape[0])
    feature_dim = int(dense_position.shape[-1])
    phase_count = int(phase_grid.shape[0])
    control_count = int(predicted_knots.shape[1] - degree - 1)

    coefficients = dense_position.new_zeros((batch_size, control_count, feature_dim))
    basis_stack = dense_position.new_zeros((batch_size, phase_count, control_count))
    basis_d1_stack = dense_position.new_zeros((batch_size, phase_count, control_count))
    basis_d2_stack = dense_position.new_zeros((batch_size, phase_count, control_count))
    recon_position = dense_position.new_zeros((batch_size, phase_count, feature_dim))
    recon_velocity = dense_position.new_zeros((batch_size, phase_count, feature_dim))
    recon_acceleration = dense_position.new_zeros((batch_size, phase_count, feature_dim))
    condition_proxy = dense_position.new_zeros((batch_size,))

    sqrt_eta_v = float(eta_velocity) ** 0.5
    sqrt_eta_a = float(eta_acceleration) ** 0.5
    solve_dtype = torch.float32
    device_type = predicted_knots.device.type

    for batch_index in range(batch_size):
        with torch.autocast(device_type=device_type, enabled=False):
            knots32 = predicted_knots[batch_index].to(dtype=solve_dtype)
            phase32 = phase_grid.to(device=knots32.device, dtype=solve_dtype)
            basis32, basis_d1_32, basis_d2_32 = bspline_basis_and_derivatives_single(knots32, phase32, degree)
            B = basis32
            Bd1 = basis_d1_32
            Bd2 = basis_d2_32
            Y = dense_position[batch_index].to(dtype=solve_dtype)
            V = dense_velocity[batch_index].to(dtype=solve_dtype)
            A = dense_acceleration[batch_index].to(dtype=solve_dtype)
            calB = torch.cat([B, sqrt_eta_v * Bd1, sqrt_eta_a * Bd2], dim=0)
            calY = torch.cat([Y, sqrt_eta_v * V, sqrt_eta_a * A], dim=0)
            lhs = calB.transpose(0, 1) @ calB
            lhs = lhs + float(ridge) * torch.eye(lhs.shape[0], device=lhs.device, dtype=lhs.dtype)
            rhs = calB.transpose(0, 1) @ calY
            solved32 = torch.linalg.solve(lhs, rhs)
            recon_position32 = basis32 @ solved32
            recon_velocity32 = basis_d1_32 @ solved32
            recon_acceleration32 = basis_d2_32 @ solved32
            singular_values = torch.linalg.svdvals(lhs)
            condition_value = singular_values.max() / torch.clamp(singular_values.min(), min=1e-12)

        basis_stack[batch_index] = basis32.to(dtype=basis_stack.dtype)
        basis_d1_stack[batch_index] = basis_d1_32.to(dtype=basis_d1_stack.dtype)
        basis_d2_stack[batch_index] = basis_d2_32.to(dtype=basis_d2_stack.dtype)
        solved = solved32.to(dtype=dense_position.dtype)
        coefficients[batch_index] = solved
        recon_position[batch_index] = recon_position32.to(dtype=recon_position.dtype)
        recon_velocity[batch_index] = recon_velocity32.to(dtype=recon_velocity.dtype)
        recon_acceleration[batch_index] = recon_acceleration32.to(dtype=recon_acceleration.dtype)
        condition_proxy[batch_index] = condition_value.to(dtype=condition_proxy.dtype)
    return (
        coefficients,
        basis_stack,
        basis_d1_stack,
        basis_d2_stack,
        recon_position,
        recon_velocity,
        recon_acceleration,
        condition_proxy,
    )
