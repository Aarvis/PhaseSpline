from __future__ import annotations

import numpy as np

from lehome_spline.spline import fit_episode_spline


def test_adaptive_spline_meets_tolerance() -> None:
    frames = 40
    timestamps = np.arange(frames, dtype=np.float64) / 23.0
    phase = np.linspace(0.1, 2.0, frames)
    mean = np.stack([np.sin(phase), np.cos(phase), phase, phase**2], axis=1).astype(np.float32)
    state = np.stack([phase, phase**2, np.sin(phase), np.cos(phase)], axis=1).astype(np.float32)
    payload, result = fit_episode_spline(
        mean,
        state,
        timestamps,
        np.arange(frames, dtype=np.int64),
        mean.mean(axis=0),
        mean.std(axis=0),
        state.mean(axis=0),
        state.std(axis=0),
        {
            "mandatory_transition_quantile": 0.95,
            "maximum_gap_frames": 8,
            "maximum_knots": frames,
            "knots_per_iteration": 4,
            "epsilon_latent_rmse": 0.01,
            "epsilon_cosine_distance": 0.001,
            "epsilon_state_rmse": 0.01,
        },
    )
    assert result.tolerance_satisfied
    assert result.knots <= frames
    assert payload["knot_embeddings"].shape[1] == 4
    assert payload["frame_u"].shape == (frames,)
    assert payload["knot_u"].shape == (result.knots,)
    assert payload["frame_u"][0] == 0.0
    assert payload["frame_u"][-1] == 1.0
