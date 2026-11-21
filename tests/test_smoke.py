from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from smoke_test import check, evaluate  # noqa: E402

AUDIO = {
    "loudness_db": -40.0,
    "baseline_db": -45.0,
    "shouting": False,
    "speaking": False,
    "swear_count_60s": 0,
}


def env(topic: str, data: dict, src: str) -> str:
    return json.dumps({"topic": topic, "ts": 1.0, "src": src, "data": data})


def status_of(lines: list[str], topic: str) -> str:
    return next(line for line in lines if line.startswith(topic + " "))


def test_rates_and_missing() -> None:
    reports: dict = {}
    for _ in range(50):
        check(env("audio.features", AUDIO, "voice"), reports)
    lines, passed = evaluate(reports, 10, ignore=set())
    assert "ok" in status_of(lines, "audio.features")
    assert "FAIL missing" in status_of(lines, "driver.state")
    assert "not seen" in status_of(lines, "voice.reply")
    assert not passed


def test_slow_topic_fails() -> None:
    reports: dict = {}
    for _ in range(20):
        check(env("audio.features", AUDIO, "voice"), reports)
    lines, _ = evaluate(reports, 10, ignore=set())
    assert "FAIL slow" in status_of(lines, "audio.features")


def test_contract_violations() -> None:
    reports: dict = {}
    check(env("audio.features", {**AUDIO, "loud": 1}, "voice"), reports)
    check(env("audio.features", AUDIO, "cv"), reports)
    check(env("made.up", {}, "core"), reports)
    lines, _ = evaluate(reports, 10, ignore=set())
    assert "2 invalid" in status_of(lines, "audio.features")
    assert "not in the contract" in status_of(lines, "made.up")


def test_ignore_passes() -> None:
    reports: dict = {}
    from backend.schemas import TOPICS

    periodic = {name for name, spec in TOPICS.items() if spec.rate_hz}
    _, passed = evaluate(reports, 10, ignore=periodic)
    assert passed
