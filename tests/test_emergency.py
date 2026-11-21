"""VR-06 HELP confirmation and VR-05 critical events, through core's real wiring (run.py)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from backend.config import load_config
from backend.core.alerts import HELP_TEXT
from backend.core.run import attach_alerts
from backend.core.triplog import TripLog
from backend.schemas import validate

CFG = load_config("default")


class FakeBus:
    def __init__(self) -> None:
        self.sent: list[tuple[str, dict[str, Any]]] = []
        self.handlers: dict[str, list[Any]] = {}

    def subscribe(self, topics: list[str], handler: Any) -> None:
        for topic in topics:
            self.handlers.setdefault(topic, []).append(handler)

    def publish_nowait(self, topic: str, data: dict[str, Any]) -> None:
        validate(topic, data)
        self.sent.append((topic, data))

    def says(self) -> list[dict[str, Any]]:
        return [d for t, d in self.sent if t == "voice.say"]


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def setup(tmp_path: Path) -> tuple[FakeBus, Any, TripLog]:
    bus = FakeBus()
    triplog = TripLog(CFG["core"]["triplog"], root=tmp_path)
    manager, _writer = attach_alerts(bus, CFG, triplog)  # type: ignore[arg-type]
    manager.clock = Clock()
    return bus, manager, triplog


def wait(manager: Any, seconds: float = 1.0) -> None:
    """Let the voice gap pass, so the next voice.say goes out."""
    manager.clock.now += seconds
    manager.tick()


def reply(manager: Any, intent: str, transcript: str = "help") -> None:
    alert = manager.active
    manager.on_voice_reply(
        {
            "alert_id": alert.alert_id,
            "transcript": transcript,
            "intent": intent,
            "latency_ms": 1,
            "path": "keyword",
        }
    )


def critical_notes(triplog: TripLog) -> list[dict[str, Any]]:
    assert triplog.path is not None
    lines = [json.loads(x) for x in triplog.path.read_text(encoding="utf-8").splitlines()]
    return [x["data"] for x in lines if x["topic"] == "log.critical"]


def test_help_gets_a_spoken_t3_confirmation_and_a_critical_log(tmp_path: Path) -> None:
    """VR-06 acceptance: saying "help" during a check-in triggers it."""
    bus, manager, triplog = setup(tmp_path)
    manager.submit("checkin_due", "T1", "How are you feeling?")
    reply(manager, "HELP")
    say = bus.says()[-1]
    assert say["tier"] == "T3" and say["text"] == HELP_TEXT["en"][0]  # T3: no gap needed
    assert critical_notes(triplog)[0]["reason"] == "help"


def test_help_skips_the_rate_limits(tmp_path: Path) -> None:
    bus, manager, _ = setup(tmp_path)
    manager.submit("distraction", "T2", "Focus?")
    reply(manager, "OK", "yes")
    manager.submit("checkin_due", "T1", "How are you?", bypass_limits=True)
    wait(manager)
    reply(manager, "HELP")  # right after another prompt: still confirmed
    assert bus.says()[-1]["text"] == HELP_TEXT["en"][0]


def test_help_to_the_confirmation_does_not_loop(tmp_path: Path) -> None:
    bus, manager, triplog = setup(tmp_path)
    manager.submit("checkin_due", "T1", "How are you feeling?")
    reply(manager, "HELP")
    reply(manager, "HELP")  # "help!" again to "are you safe?"
    confirmations = [s for s in bus.says() if s["text"] == HELP_TEXT["en"][0]]
    assert len(confirmations) == 1
    assert len(critical_notes(triplog)) == 2  # both logged


def test_alarm_logs_a_critical_event(tmp_path: Path) -> None:
    """Demo step 5: two ignored prompts give a continuous alarm and a logged critical event."""
    bus, manager, triplog = setup(tmp_path)
    alert = manager.submit("microsleep", "T3", "Are you awake?")
    for _ in range(2):  # the alert, then its repeat: no answer
        reply(manager, "NONE", "")
        wait(manager)
    assert bus.says()[-1]["audio_key"] == CFG["core"]["alerts"]["alarm_audio_key"]
    assert critical_notes(triplog)[0] == {
        "reason": "no_response",
        "alert_id": alert.alert_id,
        "type": "microsleep",
        "reply": None,
    }
