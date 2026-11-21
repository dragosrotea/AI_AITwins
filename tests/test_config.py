from __future__ import annotations

from pathlib import Path

import pytest

from backend.config import SECTIONS, ConfigError, load_config

BASE = """
core: {alerts: {listen_s: 8, t1_min_interval_s: 300}}
cv: {perclos_window_s: 60}
voice: {ring_buffer_s: 10}
ui: {}
profiles:
  demo:
    core: {alerts: {t1_min_interval_s: 60}}
"""


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "thresholds.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_default_profile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DRIVEMATE_PROFILE", raising=False)
    cfg = load_config(path=write(tmp_path, BASE))
    assert cfg["profile"] == "default"
    assert cfg["core"]["alerts"]["t1_min_interval_s"] == 300
    assert "profiles" not in cfg


def test_demo_overrides_only_given_keys(tmp_path: Path) -> None:
    cfg = load_config("demo", path=write(tmp_path, BASE))
    assert cfg["profile"] == "demo"
    assert cfg["core"]["alerts"] == {"listen_s": 8, "t1_min_interval_s": 60}
    assert cfg["cv"]["perclos_window_s"] == 60


def test_profile_from_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DRIVEMATE_PROFILE", "demo")
    assert load_config(path=write(tmp_path, BASE))["core"]["alerts"]["t1_min_interval_s"] == 60


def test_unknown_profile_fails(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="unknown profile"):
        load_config("demoo", path=write(tmp_path, BASE))


def test_profile_typo_key_fails(tmp_path: Path) -> None:
    text = BASE.replace("t1_min_interval_s: 60", "t1_min_interval: 60")
    with pytest.raises(ConfigError, match="core.alerts.t1_min_interval"):
        load_config("demo", path=write(tmp_path, text))


def test_missing_section_fails(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="ui"):
        load_config(path=write(tmp_path, BASE.replace("ui: {}\n", "")))


@pytest.mark.parametrize("profile", ["default", "demo"])
def test_real_config_loads(profile: str) -> None:
    cfg = load_config(profile)
    assert all(name in cfg for name in SECTIONS)


def test_real_demo_shortens_windows() -> None:
    base, demo = load_config("default"), load_config("demo")
    assert demo["core"]["zoning"]["monotony_window_s"] == 60
    assert demo["cv"]["zoning_stare_s"] == 30
    assert (
        demo["core"]["checkin"]["base_interval_min"] < base["core"]["checkin"]["base_interval_min"]
    )
