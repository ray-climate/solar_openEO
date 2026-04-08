"""Config loading helpers for training experiments."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def _deep_update(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    parent = config.pop("inherits", None)
    if parent:
        parent_path = (config_path.parent / parent).resolve()
        base = load_config(parent_path)
        config = _deep_update(base, config)

    return config
