from __future__ import annotations

import numpy as np


def normalize_knots_to_unit_domain(global_knots: np.ndarray, degree: int) -> np.ndarray:
    knots = np.asarray(global_knots, dtype=np.float64)
    degree = int(degree)
    domain_start = float(knots[degree])
    domain_end = float(knots[-degree - 1])
    width = max(domain_end - domain_start, 1e-12)
    return (knots - domain_start) / width


def compute_coefficient_geometry(
    normalized_knots: np.ndarray,
    degree: int,
    coefficient_count: int,
) -> dict[str, np.ndarray]:
    knots = np.asarray(normalized_knots, dtype=np.float64)
    degree = int(degree)
    coefficient_count = int(coefficient_count)
    if degree != 3:
        raise ValueError(f"Localizer V1 expects cubic splines (degree=3), got {degree}")
    left = knots[:coefficient_count]
    right = knots[degree + 1 : degree + 1 + coefficient_count]
    midpoint = 0.5 * (left + right)
    width = right - left
    greville = (knots[1 : coefficient_count + 1] + knots[2 : coefficient_count + 2] + knots[3 : coefficient_count + 3]) / 3.0
    return {
        "left_support": left.astype(np.float32),
        "right_support": right.astype(np.float32),
        "support_midpoint": midpoint.astype(np.float32),
        "support_width": width.astype(np.float32),
        "greville_phase": greville.astype(np.float32),
    }


def evaluate_bspline_basis_matrix(
    normalized_knots: np.ndarray,
    degree: int,
    phase_values: np.ndarray,
    coefficient_count: int,
) -> np.ndarray:
    knots = np.asarray(normalized_knots, dtype=np.float64)
    phase_values = np.asarray(phase_values, dtype=np.float64)
    coefficient_count = int(coefficient_count)
    if degree < 0:
        raise ValueError("degree must be non-negative")
    base_rows = knots.shape[0] - 1
    values = phase_values[None, :]
    basis_prev = ((knots[:-1, None] <= values) & (values < knots[1:, None])).astype(np.float64)
    if phase_values.size and np.isclose(phase_values[-1], knots[-1]):
        basis_prev[-2, -1] = 1.0
    for p in range(1, degree + 1):
        row_count = knots.shape[0] - p - 1
        basis_curr = np.zeros((row_count, phase_values.shape[0]), dtype=np.float64)
        for i in range(row_count):
            left_den = knots[i + p] - knots[i]
            right_den = knots[i + p + 1] - knots[i + 1]
            if left_den > 0:
                basis_curr[i] += ((phase_values - knots[i]) / left_den) * basis_prev[i]
            if right_den > 0:
                basis_curr[i] += ((knots[i + p + 1] - phase_values) / right_den) * basis_prev[i + 1]
        basis_prev = basis_curr
    if basis_prev.shape[0] != coefficient_count:
        raise RuntimeError(
            f"Basis matrix row count mismatch: expected {coefficient_count}, got {basis_prev.shape[0]}"
        )
    return basis_prev.T.astype(np.float32)
