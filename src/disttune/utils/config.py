"""YAML config loader with merge and override support."""

from __future__ import annotations
from pathlib import Path
from typing import Any

import yaml


def load_config(path: Path) -> dict[str, Any]:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if "base" in cfg:
        base_path = Path(path).parent / cfg.pop("base")
        base = load_config(base_path)
        cfg = _deep_merge(base, cfg)
    return cfg


def merge_configs(cfg: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    for override in overrides:
        key, val = override.split("=", 1)
        keys = key.split(".")
        d = cfg
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = _cast(val)
    return cfg


def _deep_merge(base: dict, override: dict) -> dict:
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _cast(val: str) -> int | float | bool | str:
    if val.lower() in ("true", "false"):
        return val.lower() == "true"
    try:
        return int(val)
    except ValueError:
        pass
    try:
        return float(val)
    except ValueError:
        pass
    return val
