from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


OUTPUT_PATH_KEYS = ("root", "embeddings_dir", "splines_dir", "bspline_dataset_dir", "bspline_external_dir")


def _parse_override(value: str) -> Any:
    return yaml.safe_load(value)


def _set_nested(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    cursor = config
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        child = cursor.get(part)
        if child is None:
            child = {}
            cursor[part] = child
        if not isinstance(child, dict):
            raise ValueError(f"Cannot set {dotted_key!r}: {part!r} is not a mapping")
        cursor = child
    cursor[parts[-1]] = value


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Configuration root must be a mapping: {path}")

    config = copy.deepcopy(loaded)
    for override in overrides or []:
        if "=" not in override:
            raise ValueError(f"Override must be KEY=VALUE, got {override!r}")
        key, value = override.split("=", 1)
        _set_nested(config, key.strip(), _parse_override(value.strip()))

    component_root = path.parent.parent
    output_config = config.get("output", {})
    for key in OUTPUT_PATH_KEYS:
        value = output_config.get(key)
        if value is not None:
            output_path = Path(value).expanduser()
            if not output_path.is_absolute():
                output_path = component_root / output_path
            output_config[key] = str(output_path.resolve())

    config["_config_path"] = str(path)
    config["_component_root"] = str(component_root)
    return config


def save_resolved_config(config: dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {key: value for key, value in config.items() if not key.startswith("_")}
    with destination.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
