from __future__ import annotations

import numpy as np
from scipy.interpolate import BSpline

from lehome_spline.bspline import fit_episode_bspline


def test_adaptive_bspline_meets_tolerance() -> None:
    frames = 40
    timestamps = np.arange(frames, dtype=np.float64) / 23.0
    phase = np.linspace(0.1, 2.0, frames)
    mean = np.stack([np.sin(phase), np.cos(phase), phase, phase**2], axis=1).astype(np.float32)
    state = np.stack([phase, phase**2, np.sin(phase), np.cos(phase)], axis=1).astype(np.float32)
    payload, result = fit_episode_bspline(
        mean,
        state,
        timestamps,
        np.arange(frames, dtype=np.int64),
        mean.mean(axis=0),
        np.maximum(mean.std(axis=0), 1e-6),
        state.mean(axis=0),
        np.maximum(state.std(axis=0), 1e-6),
        {
            "degree": 3,
            "epsilon_latent_rmse": 0.05,
            "epsilon_cosine_distance": 0.005,
            "epsilon_state_rmse": 0.05,
            "initial_internal_knots": 2,
            "max_internal_knots": frames - 4,
            "min_knot_spacing_frames": 1,
            "endpoint_weight": 1000.0,
            "compressed": True,
        },
    )
    assert result.tolerance_satisfied
    assert result.num_control_points <= frames
    assert payload["frame_u"].shape == (frames,)
    assert payload["global_knots"].ndim == 1
    assert payload["global_coefficients"].shape[1] == 4
    spline = BSpline(payload["global_knots"], payload["global_coefficients"].astype(np.float64), int(payload["global_degree"][0]))
    reconstructed = spline(payload["frame_u"].astype(np.float64))
    assert reconstructed.shape == mean.shape

