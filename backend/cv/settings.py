from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from backend.config import load_config

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "config" / "thresholds.yaml"


def load_cv_settings(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"CV configuration not found: {config_path}")
    return load_config(path=config_path)["cv"]


def integer_setting(settings: dict[str, Any], section: str, name: str, *, minimum: int = 0) -> int:
    try:
        value = settings[section][name]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Missing CV setting {section}.{name}") from exc
    if type(value) is not int or value < minimum:
        raise ValueError(f"CV setting {section}.{name} must be an integer >= {minimum}")
    return value


def float_setting(settings: dict[str, Any], section: str, name: str) -> float:
    try:
        value = settings[section][name]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Missing CV setting {section}.{name}") from exc
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"CV setting {section}.{name} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"CV setting {section}.{name} must be finite")
    return parsed


def string_setting(settings: dict[str, Any], section: str, name: str) -> str:
    try:
        value = settings[section][name]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Missing CV setting {section}.{name}") from exc
    if not isinstance(value, str) or not value:
        raise ValueError(f"CV setting {section}.{name} must be a non-empty string")
    return value


def resolved_path(value: str, *, root: Path = ROOT) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path
