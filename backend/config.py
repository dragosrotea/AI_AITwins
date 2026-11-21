"""Load config/thresholds.yaml, applying the profile named by DRIVEMATE_PROFILE.

from backend.config import get_config
listen_s = get_config()["core"]["alerts"]["listen_s"]
"""

from __future__ import annotations

import copy
import os
from functools import cache
from pathlib import Path
from typing import Any

import yaml

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "thresholds.yaml"
PROFILE_ENV = "DRIVEMATE_PROFILE"
SECTIONS = ("core", "cv", "voice", "ui")


class ConfigError(ValueError):
    pass


def _overlay(base: dict[str, Any], override: dict[str, Any], where: str) -> None:
    """Merge override into base in place. Unknown keys are errors, so typos never pass."""
    for key, value in override.items():
        path = f"{where}.{key}"
        if key not in base:
            raise ConfigError(f"profile overrides unknown key {path}")
        if isinstance(base[key], dict):
            if not isinstance(value, dict):
                raise ConfigError(f"profile must override {path} with a mapping")
            _overlay(base[key], value, path)
        else:
            base[key] = value


def load_config(profile: str | None = None, path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Return the config with `profile` applied.

    profile=None reads DRIVEMATE_PROFILE; "" or "default" means no profile.
    """
    with path.open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    profiles = raw.pop("profiles", None) or {}
    missing = [name for name in SECTIONS if name not in raw]
    if missing:
        raise ConfigError(f"{path.name} is missing sections: {', '.join(missing)}")

    if profile is None:
        profile = os.environ.get(PROFILE_ENV, "").strip()
    if profile in ("", "default"):
        profile = None
    config = copy.deepcopy(raw)
    if profile is not None:
        if profile not in profiles:
            known = ", ".join(sorted(profiles)) or "none"
            raise ConfigError(f"unknown profile {profile!r} (known: {known})")
        _overlay(config, profiles[profile], profile)
    config["profile"] = profile or "default"
    return config


@cache
def get_config() -> dict[str, Any]:
    """Process-wide config, loaded once. Treat it as read-only."""
    return load_config()
