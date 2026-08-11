from types import SimpleNamespace

import pytest

from lehome_spline.config import load_config
from lehome_spline.training import (
    _checkpoint_state_normalization,
    _episode_split_indices,
    _stages,
    _validate_initialization_split,
    allocate_stage_steps,
)


def test_joint_full_epoch_config_uses_every_training_step() -> None:
    config = load_config("configs/joint_full_epoch.yaml")
    stages = _stages(config)

    assert len(stages) == 1
    assert stages[0].name == "joint_cumulative_full_epoch"
    assert stages[0].fraction == 1.0
    assert stages[0].stochastic is True
    assert stages[0].temporal is True
    assert stages[0].kl_max == pytest.approx(1.0e-3)
    assert allocate_stage_steps(33_301, [stages[0].fraction]) == [33_301]


def test_initialization_checkpoint_requires_the_same_episode_split() -> None:
    splits = {
        "train": [SimpleNamespace(episode_index=3), SimpleNamespace(episode_index=7)],
        "val": [SimpleNamespace(episode_index=2)],
        "test": [],
    }
    current = _episode_split_indices(splits)
    checkpoint = {"split_indices": {"train": [3, 7], "val": [2], "test": []}}

    _validate_initialization_split(checkpoint, current)

    checkpoint["split_indices"]["train"] = [7, 3]
    with pytest.raises(ValueError, match="dataset split does not match"):
        _validate_initialization_split(checkpoint, current)


def test_initialization_checkpoint_supplies_state_normalization() -> None:
    normalization = {"mean": [0.0, 1.0], "std": [1.0, 2.0]}
    assert _checkpoint_state_normalization({"state_normalization": normalization}) is normalization

    with pytest.raises(ValueError, match="state normalization"):
        _checkpoint_state_normalization({})
