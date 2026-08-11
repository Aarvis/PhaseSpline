from __future__ import annotations

from pathlib import Path

from lehome_spline.data import EpisodeRecord, TemporalWindowDataset


def _dataset(length: int, stride: int = 1) -> TemporalWindowDataset:
    return TemporalWindowDataset(
        episodes=[EpisodeRecord(episode_index=0, length=length, path=Path("unused.parquet"))],
        horizons=[1, 2, 4, 8, 16, 32],
        stride=stride,
        image_size=16,
        state_mean=[0.0, 0.0, 0.0, 0.0],
        state_std=[1.0, 1.0, 1.0, 1.0],
        cache_episodes=1,
        include_future_images=True,
    )


def test_temporal_windows_never_request_past_episode_end() -> None:
    dataset = _dataset(length=262)
    assert dataset.windows[0] == (0, 0)
    assert dataset.windows[-1] == (0, 229)
    assert all(start + dataset.max_horizon < 262 for _, start in dataset.windows)


def test_temporal_window_minimum_episode_length() -> None:
    assert len(_dataset(length=32)) == 0
    assert _dataset(length=33).windows == [(0, 0)]
    assert _dataset(length=34).windows == [(0, 0), (0, 1)]


def test_strided_temporal_windows_obey_the_same_bound() -> None:
    dataset = _dataset(length=100, stride=8)
    assert dataset.windows[-1] == (0, 64)
    assert all(start + dataset.max_horizon < 100 for _, start in dataset.windows)
