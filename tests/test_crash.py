"""LEAD-T22: crash detection (CR-01) and the 20 s countdown (CR-02), with fake clocks."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock

import pytest

from backend.config import load_config
from backend.core.alerts import AlertManager
from backend.core.crash import CrashDetector
from backend.core.events import TEMPLATES, EventRouter
from backend.core.vehicle import Injector
from backend.core.wording import AlertWriter
from backend.schemas import validate

CORE = load_config("default")["core"]
CRASH, ALERTS = CORE["crash"], CORE["alerts"]


def sample(speed: float, mag: float = 0.1) -> dict[str, Any]:
    return {"accel_mag_g": mag, "speed_kmh": speed, "lat": 40.5, "lon": -3.4}


class Detector:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self.d = CrashDetector(CRASH, self.emit)
        self.t = 0.0

    def emit(self, data: dict[str, Any]) -> None:
        validate("detector.event", data)
        self.events.append(data)

    def feed(self, speed: float, mag: float = 0.1, dt: float = 0.1) -> None:
        self.t += dt
        self.d.on_vehicle(sample(speed, mag), self.t)


# CR-01


def test_spike_then_stop_is_a_crash_candidate() -> None:
    det = Detector()
    det.feed(100, mag=5.0)
    for speed in (80, 50, 20, 3):
        det.feed(speed, dt=0.5)
    assert len(det.events) == 1
    event = det.events[0]
    assert event["type"] == "crash_candidate" and event["tier"] == "T3"
    assert event["signals"]["spike_g"] == 5.0 and event["signals"]["lat"] == 40.5


def test_spike_while_moving_is_ignored() -> None:
    """CR-01 acceptance: a 4 g spike while the replay is still moving does not trigger."""
    det = Detector()
    det.feed(100, mag=4.5)
    for _ in range(int(CRASH["stop_within_s"] / 0.5) + 2):
        det.feed(100, dt=0.5)
    det.feed(0, dt=0.5)  # stopping later is not this crash
    assert det.events == []


def test_below_threshold_is_not_a_spike() -> None:
    det = Detector()
    det.feed(100, mag=CRASH["spike_g"] - 0.1)
    det.feed(0)
    assert det.events == []


def test_one_candidate_per_spike() -> None:
    det = Detector()
    det.feed(100, mag=5.0)
    for _ in range(20):
        det.feed(0)
    assert len(det.events) == 1


def test_the_injected_crash_is_detected() -> None:
    """Demo step 7: control.inject crash through the injector, at 10 Hz."""
    clock = [0.0]
    injector = Injector(CORE["inject"], clock=lambda: clock[0])
    det = Detector()
    base = {
        "accel_long_g": 0.0,
        "accel_lat_g": 0.0,
        "accel_mag_g": 0.0,
        "speed_kmh": 100.0,
        "lat": 40.5,
        "lon": -3.4,
        "road": "motorway",
        "source": "replay",
    }
    injector.inject("crash")
    for _ in range(60):
        out = injector.apply(base)
        det.d.on_vehicle(out, clock[0])
        clock[0] += 0.1
    assert len(det.events) == 1
    assert det.events[0]["signals"]["stop_after_s"] <= CORE["inject"]["crash_stop_s"] + 0.2


# CR-02: countdown


class Rig:
    def __init__(self) -> None:
        self.now = 1000.0
        self.sent: list[tuple[str, dict[str, Any]]] = []
        self.emergencies: list[tuple[str, str | None]] = []
        self.am = AlertManager(
            ALERTS,
            self.publish,
            clock=lambda: self.now,
            on_emergency=lambda a, r: self.emergencies.append((a.type, r and r["intent"])),
        )
        self.am.countdowns = {"crash_candidate": CRASH["countdown_s"]}

    def publish(self, topic: str, data: dict[str, Any]) -> None:
        validate(topic, data)
        self.sent.append((topic, data))

    def crash(self) -> Any:
        return self.am.submit("crash_candidate", "T3", "The car stopped suddenly. Are you okay?")

    def reply(self, intent: str, transcript: str = "x") -> None:
        alert = self.am.active
        self.am.on_voice_reply(
            {
                "alert_id": alert.alert_id,
                "transcript": transcript,
                "intent": intent,
                "latency_ms": 1,
                "path": "keyword",
            }
        )

    def advance(self, seconds: float) -> None:
        end = self.now + seconds
        while self.now < end:
            self.now = min(end, self.now + 0.1)
            self.am.tick()

    def statuses(self) -> list[str]:
        return [d["status"] for t, d in self.sent if t == "alert.state"]

    def says(self) -> list[dict[str, Any]]:
        return [d for t, d in self.sent if t == "voice.say"]


@pytest.fixture
def rig() -> Rig:
    return Rig()


def test_ok_cancels_the_countdown(rig: Rig) -> None:
    rig.crash()
    rig.advance(5)
    rig.reply("OK", "I'm fine")
    assert rig.statuses()[-1] == "closed" and rig.emergencies == []


def test_silence_sounds_the_alarm_then_escalates_at_20_s(rig: Rig) -> None:
    rig.crash()
    rig.advance(2)
    rig.reply("NONE", "")  # silence: alarm, not a repeat or a tier change
    gap = ALERTS["voice_gap_s"] + 0.5  # the pause before the next voice.say
    rig.advance(gap)
    assert rig.says()[-1]["audio_key"] == ALERTS["alarm_audio_key"]
    assert rig.says()[-1]["tier"] == "T3"
    rig.advance(CRASH["countdown_s"] - 2 - gap - 1)  # 1 s before the countdown ends
    assert rig.emergencies == []  # still counting down
    rig.advance(2)
    assert rig.statuses()[-1] == "escalated"  # the dashboard's emergency banner
    assert rig.emergencies == [("crash_candidate", None)]
    assert rig.am.active is None


def test_help_escalates_at_once(rig: Rig) -> None:
    rig.crash()
    rig.advance(3)
    rig.reply("HELP", "help")
    assert rig.statuses()[-1] == "escalated"
    assert rig.emergencies == [("crash_candidate", "HELP")]


def test_countdown_runs_even_without_any_voice_reply(rig: Rig) -> None:
    rig.crash()
    rig.advance(CRASH["countdown_s"] + 0.5)
    assert rig.statuses()[-1] == "escalated" and len(rig.emergencies) == 1


def test_other_alerts_have_no_countdown(rig: Rig) -> None:
    rig.am.submit("microsleep", "T3", "Awake?")
    rig.advance(CRASH["countdown_s"] + 1)
    assert rig.emergencies == [] and "escalated" not in rig.statuses()[:3]


def test_core_crash_event_routes_to_a_t3_alert_with_audio(monkeypatch: pytest.MonkeyPatch) -> None:
    from backend.llm import client

    monkeypatch.setattr(client, "request_text", AsyncMock(return_value=None))
    rig = Rig()
    calls: list = []

    async def llm(ctx: dict[str, Any], timeout_s: float) -> str | None:
        calls.append(ctx)
        return "never used"

    router = EventRouter(rig.am, CORE["routing"], AlertWriter(2.0, TEMPLATES, generate=llm))
    envelope = {
        "topic": "detector.event",
        "ts": 1.0,
        "src": "core",
        "data": {
            "event_id": "crash-1",
            "type": "crash_candidate",
            "tier": "T3",
            "score": 100.0,
            "signals": {},
        },
    }
    alert = asyncio.run(router.handle(envelope))
    assert alert is not None and alert.tier == "T3" and calls == []  # T3 never waits on the LLM
    assert rig.says()[0]["audio_key"].startswith("crash_candidate_en_")
    assert alert.countdown_until == pytest.approx(rig.now + CRASH["countdown_s"])
