from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("PyYAML is required to load human-spline-localizer configs.") from exc


DEFAULT_OUTPUT_ROOT = "D:/LeHome-Challenge/Lehome-Spline-ICRA2027/human-spline-localizer/outputs/default_run"
DEFAULT_INTERVAL_OUTPUT_ROOT = "D:/LeHome-Challenge/Lehome-Spline-ICRA2027/human-spline-localizer/outputs/default_interval_prediction_run"
DEFAULT_INFERENCE_OUTPUT_ROOT = (
    "E:/Lehome-Dataset/lehome_round_2_dataset/sim_dataset/robot_sim_ft_lehome_all_garment_data_z180/"
    "embeddings/robot_sim_multiview_vae_joint_full_visual_epoch/human_localizer_predicted_u_default_run"
)
DEFAULT_INTERVAL_INFERENCE_OUTPUT_ROOT = (
    "E:/Lehome-Dataset/lehome_round_2_dataset/sim_dataset/robot_sim_ft_lehome_all_garment_data_z180/"
    "embeddings/robot_sim_multiview_vae_joint_full_visual_epoch/human_localizer_predicted_u_default_interval_prediction_run"
)
DEFAULT_CHECKPOINT_PATH = (
    "D:/LeHome-Challenge/Lehome-Spline-ICRA2027/human-spline-localizer/outputs/default_run/checkpoints/best.pt"
)
DEFAULT_INTERVAL_CHECKPOINT_PATH = (
    "D:/LeHome-Challenge/Lehome-Spline-ICRA2027/human-spline-localizer/outputs/default_interval_prediction_run/checkpoints/best.pt"
)
DEFAULT_INTERVAL_RUN_NAME = "default interval prediction run"


def _read_config_file(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping config at {path}")
    return data


def _set_nested(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    cursor: dict[str, Any] = config
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        next_value = cursor.get(part)
        if not isinstance(next_value, dict):
            next_value = {}
            cursor[part] = next_value
        cursor = next_value
    cursor[parts[-1]] = value


def _interval_prediction_enabled(config: dict[str, Any]) -> bool:
    model_cfg = config.get("model")
    if not isinstance(model_cfg, dict):
        return False
    auxiliary_cfg = model_cfg.get("auxiliary")
    if not isinstance(auxiliary_cfg, dict):
        return False
    interval_cfg = auxiliary_cfg.get("interval_prediction")
    if not isinstance(interval_cfg, dict):
        return False
    return bool(interval_cfg.get("enabled", False))


def _apply_interval_prediction_defaults(config: dict[str, Any]) -> None:
    if not _interval_prediction_enabled(config):
        return
    paths_cfg = config.get("paths")
    if isinstance(paths_cfg, dict):
        if str(paths_cfg.get("output_root", "")).replace("\\", "/") == DEFAULT_OUTPUT_ROOT:
            paths_cfg["output_root"] = DEFAULT_INTERVAL_OUTPUT_ROOT
    inference_cfg = config.get("inference")
    if isinstance(inference_cfg, dict):
        if str(inference_cfg.get("output_root", "")).replace("\\", "/") == DEFAULT_INFERENCE_OUTPUT_ROOT:
            inference_cfg["output_root"] = DEFAULT_INTERVAL_INFERENCE_OUTPUT_ROOT
        if str(inference_cfg.get("checkpoint_path", "")).replace("\\", "/") == DEFAULT_CHECKPOINT_PATH:
            inference_cfg["checkpoint_path"] = DEFAULT_INTERVAL_CHECKPOINT_PATH
    logging_cfg = config.get("logging")
    if isinstance(logging_cfg, dict):
        wandb_cfg = logging_cfg.get("wandb")
        if isinstance(wandb_cfg, dict) and wandb_cfg.get("run_name") in (None, ""):
            wandb_cfg["run_name"] = DEFAULT_INTERVAL_RUN_NAME


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    config = _read_config_file(path)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must have the form KEY=VALUE, got: {item!r}")
        key, raw_value = item.split("=", 1)
        _set_nested(config, key, yaml.safe_load(raw_value))
    _apply_interval_prediction_defaults(config)
    return config


def save_resolved_config(config: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
