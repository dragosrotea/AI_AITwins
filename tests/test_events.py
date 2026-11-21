"""LEAD-T11: a CV distraction event becomes a T2 alert with a fixed template."""

from __future__ import annotations

from typing import Any

import pytest

from backend.config import load_config
from backend.core.alerts import AlertManager
from backend.core.events import TEMPLATES, handle_detector_event
from backend.schemas import validate

CFG = load_config("default")["core"]["alerts"]


@pytest.fixture
def setup() -> tuple[AlertManager, list[tuple[str, dict[str, Any]]]]:
    sent: list[tuple[str, dict[str, Any]]] = []

    def publish(topic: str, data: dict[str, Any]) -> None:
        validate(topic, data)
        sent.append((topic, data))

    return AlertManager(CFG, publish, clock=lambda: 0.0), sent


def event(kind: str = "distraction", src: str = "cv", **data: Any) -> dict[str, Any]:
    envelope = {
        "topic": "detector.event",
        "ts": 1.0,
        "src": src,
        "data": {
            "event_id": "e-1",
            "type": kind,
            "tier": "T2",
            "score": 72.0,
            "signals": {"yaw": 41.0, "off_road_s": 2.4},
            **data,
        },
    }
    validate("detector.event", envelope["data"])
    return envelope


def says(sent: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    return [d for t, d in sent if t == "voice.say"]


def test_distraction_becomes_spoken_t2_alert(setup: Any) -> None:
    manager, sent = setup
    alert = handle_detector_event(event(), manager)
    assert alert is not None and alert.tier == "T2" and alert.type == "distraction"
    say = says(sent)[0]
    assert say["tier"] == "T2" and say["lang"] == "en"
    assert say["text"] == TEMPLATES["distraction"]["T2"]["en"][0]
    assert alert.signals == {"event_id": "e-1", "score": 72.0, "yaw": 41.0, "off_road_s": 2.4}


def test_driver_language_is_used(setup: Any) -> None:
    manager, sent = setup
    manager.lang = "ro"
    handle_detector_event(event(), manager)
    assert says(sent)[0]["text"] == TEMPLATES["distraction"]["T2"]["ro"][0]
    assert says(sent)[0]["lang"] == "ro"


def test_short_template_used_for_the_repeat(setup: Any) -> None:
    manager, sent = setup
    alert = handle_detector_event(event(), manager)
    assert alert is not None and alert.short_text == TEMPLATES["distraction"]["T2"]["en"][1]


def test_repeated_events_make_one_alert(setup: Any) -> None:
    manager, sent = setup
    handle_detector_event(event(), manager)
    assert handle_detector_event(event(event_id="e-2"), manager) is None
    assert len(says(sent)) == 1


def test_other_types_and_core_events_are_skipped(setup: Any) -> None:
    manager, sent = setup
    assert handle_detector_event(event("phone"), manager) is None  # LEAD-T15
    assert handle_detector_event(event(src="core"), manager) is None  # no loop on fusion events
    assert sent == []


@pytest.mark.parametrize("lang", ["en", "ro"])
def test_templates_follow_vr01(lang: str) -> None:
    """Every alert ends with a question (VR-01) and stays short (LLM rules: at most 25 words)."""
    for tiers in TEMPLATES.values():
        for langs in tiers.values():
            for text in langs[lang]:
                assert text.endswith("?") and len(text.split()) <= 25
