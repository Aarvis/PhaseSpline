from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


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


def _resolve_path(value: str | None, config_dir: Path) -> str | None:
    if value is None or value == "":
        return value
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = (config_dir / path).resolve()
    return str(path)


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Configuration root must be a mapping: {config_path}")

    config = copy.deepcopy(loaded)
    for override in overrides or []:
        if "=" not in override:
            raise ValueError(f"Override must be KEY=VALUE, got {override!r}")
        key, value = override.split("=", 1)
        _set_nested(config, key.strip(), _parse_override(value.strip()))

    config_dir = config_path.parent
    for section_name in ("paths", "runtime", "prompt_selection", "end_mode", "server"):
        section = config.get(section_name)
        if not isinstance(section, dict):
            continue
        for key, value in list(section.items()):
            if key.endswith("_path") or key.endswith("_root") or key.endswith("_dir") or key.endswith("_repo"):
                section[key] = _resolve_path(value, config_dir)

    for section_name in ("robot_embedder", "localizer", "translator"):
        section = config.get(section_name)
        if isinstance(section, dict):
            for key, value in list(section.items()):
                if key.endswith("_path") or key.endswith("_root") or key.endswith("_dir"):
                    section[key] = _resolve_path(value, config_dir)

    config["_config_path"] = str(config_path)
    return config


def save_resolved_config(config: dict[str, Any], path: str | Path) -> None:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {key: value for key, value in config.items() if not key.startswith("_")}
    with destination.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)

