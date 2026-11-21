"""LEAD-T20: check-in scheduler [CI-02, CI-03, CI-04, CI-05], with a fake wall clock."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from backend.config import load_config
from backend.core.alerts import AlertManager
from backend.core.checkin import CheckinScheduler, base_interval
from backend.core.events import TEMPLATES
from backend.core.wording import AlertWriter
from backend.schemas import validate

CORE = load_config("default")["core"]
CFG = CORE["checkin"]


def at(hour: int, minute: int = 0) -> float:
    """Unix time for today's local hour:minute, as the pre-trip departure time."""
    return time.mktime((2026, 11, 21, hour, minute, 0, 0, 0, -1))


def pretrip(sleep_h: float = 8, tiredness: int = 2) -> dict[str, Any]:
    return {
        "planned_min": 120,
        "sleep_h": sleep_h,
        "awake_h": 10,
        "tiredness": tiredness,
        "caffeine": False,
        "engagement": "trivia",
        "lang": "en",
        "emergency_contact": "+40700000000",
        "resting_hr": 65,
    }


async def no_llm(ctx: dict[str, Any], timeout_s: float) -> str | None:
    return None


class Rig:
    def __init__(self, start: float) -> None:
        self.now = start
        self.sent: list[tuple[str, dict[str, Any]]] = []
        self.am = AlertManager(CORE["alerts"], self.publish, clock=lambda: self.now)
        writer = AlertWriter(CORE["alerts"]["llm_timeout_s"], TEMPLATES, generate=no_llm)
        self.sched = CheckinScheduler(
            CFG, self.am, writer, TEMPLATES, self.publish, clock=lambda: self.now
        )
        self.am.on_submit = self.sched.on_alert
        self.am.on_follow_up = self.sched.on_not_ok

    def publish(self, topic: str, data: dict[str, Any]) -> None:
        validate(topic, data)
        self.sent.append((topic, data))

    def start(self, **kw: Any) -> None:
        self.sched.on_trip({"action": "start", "pretrip": pretrip(**kw)})

    def advance(self, seconds: float, step: float = 1.0) -> None:
        end = self.now + seconds
        while self.now < end:
            self.now = min(end, self.now + step)
            self.am.tick()
            asyncio.run(self.sched.tick())

    def schedules(self) -> list[dict[str, Any]]:
        return [d for t, d in self.sent if t == "checkin.schedule"]

    def spoken(self, kind: str) -> list[dict[str, Any]]:
        ids = {d["alert_id"] for t, d in self.sent if t == "alert.state" and d["type"] == kind}
        return [d for t, d in self.sent if t == "voice.say" and d["alert_id"] in ids]

    def answer(self, intent: str) -> None:
        alert = self.am.active
        assert alert is not None
        self.am.on_voice_reply(
            {
                "alert_id": alert.alert_id,
                "transcript": "x",
                "intent": intent,
                "latency_ms": 300,
                "path": "keyword",
            }
        )


# CI-02 and CI-03: the acceptance case and the rule


def test_acceptance_5h_sleep_departing_21_00() -> None:
    """5 h sleep, departure 21:00 -> check-ins every 15 min, a break at 1:45."""
    rig = Rig(at(21))
    rig.start(sleep_h=5)
    first = rig.schedules()[0]
    assert first["interval_min"] == 15
    assert first["next_checkin_ts"] == pytest.approx(at(21, 15))
    assert first["next_break_ts"] == pytest.approx(at(22, 45))  # 1 h 45 min


@pytest.mark.parametrize(
    ("sleep_h", "tiredness", "hour", "minutes", "risk"),
    [
        (8, 2, 10, 30, False),  # rested, mid-morning
        (5, 2, 10, 15, True),  # short sleep halves
        (8, 4, 10, 15, True),  # tiredness 4+ halves
        (5, 5, 10, 10, True),  # both: 7.5, clamped to the 10 min minimum
        (8, 2, 15, 19.8, True),  # 14:00-16:00 dip: x0.66
        (8, 2, 2, 19.8, True),  # 00:00-06:00 dip
        (8, 2, 6, 30, False),  # 06:00 is outside the night dip
    ],
)
def test_ci02_rule(sleep_h: float, tiredness: int, hour: int, minutes: float, risk: bool) -> None:
    got, got_risk = base_interval(pretrip(sleep_h, tiredness), at(hour), CFG)
    assert got == pytest.approx(minutes) and got_risk is risk


def test_break_every_2_h_without_risk() -> None:
    rig = Rig(at(10))
    rig.start()
    assert rig.schedules()[0]["next_break_ts"] == pytest.approx(at(12))


# Check-ins and breaks are spoken


def test_due_checkin_is_spoken_as_t1_and_reschedules() -> None:
    rig = Rig(at(21))
    rig.start(sleep_h=5)
    rig.advance(15 * 60)
    say = rig.spoken("checkin_due")
    assert len(say) == 1 and say[0]["tier"] == "T1" and say[0]["text"].endswith("?")
    rig.answer("OK")
    assert rig.schedules()[-1]["next_checkin_ts"] == pytest.approx(at(21, 30))


def test_break_reminder_at_1_45() -> None:
    rig = Rig(at(21))
    rig.start(sleep_h=5)
    for _ in range(8):  # answer every check-in so the queue stays clear
        rig.advance(15 * 60)
        if rig.am.active is not None:
            rig.answer("OK")
            rig.advance(1)
    rig.advance(10 * 60)
    assert len(rig.spoken("break_due")) >= 1
    first_break = next(d for t, d in rig.sent if t == "alert.state" and d["type"] == "break_due")
    assert first_break["status"] == "queued"


def test_checkin_held_by_rate_limits_waits_and_is_not_lost() -> None:
    rig = Rig(at(10))
    rig.start()
    rig.advance(30 * 60 - 30)
    rig.am.submit("distraction", "T2", "Focus?")  # 30 s before the check-in
    rig.answer("OK")
    rig.advance(31)
    assert rig.spoken("checkin_due") == []  # held: one non-critical prompt per 2 min
    rig.advance(CORE["alerts"]["noncritical_min_interval_s"])
    checkins = {d["alert_id"] for d in rig.spoken("checkin_due")}  # one alert (plus its repeat)
    assert len(checkins) == 1  # it came after the hold, not dropped


# CI-04: live adaptation


def test_warning_shortens_the_next_checkin_by_30_percent() -> None:
    rig = Rig(at(10))
    rig.start()
    rig.advance(5 * 60)
    rig.am.submit("phone", "T2", "Phone?")
    assert rig.schedules()[-1]["interval_min"] == pytest.approx(21)  # 30 * 0.7
    rig.answer("OK")


def test_quiet_half_hour_lengthens_by_20_percent() -> None:
    rig = Rig(at(10))
    rig.start()
    rig.advance(30 * 60)
    if rig.am.active is not None:
        rig.answer("OK")
    rig.advance(5)
    assert rig.schedules()[-1]["interval_min"] == pytest.approx(36)  # 30 * 1.2


def test_interval_never_leaves_the_bounds() -> None:
    rig = Rig(at(10))
    rig.start(sleep_h=5, tiredness=5)
    for kind in ("phone", "distraction", "no_face"):
        rig.sched.on_alert(type("A", (), {"type": kind, "tier": "T2"})())
    assert rig.schedules()[-1]["interval_min"] == CFG["min_interval_min"]


# CI-05: a tired reply offers a break and shortens the interval


def test_not_ok_offers_a_break_at_once_and_shortens() -> None:
    rig = Rig(at(10))
    rig.start()
    rig.advance(30 * 60)
    rig.answer("NOT_OK")
    assert rig.schedules()[-1]["interval_min"] < 30
    rig.advance(1)
    break_say = rig.spoken("break_due")
    assert len(break_say) == 1  # despite the 2 min non-critical limit: it answers the driver


def test_not_ok_to_a_break_offer_does_not_loop() -> None:
    rig = Rig(at(10))
    rig.start()
    rig.advance(30 * 60)
    rig.answer("NOT_OK")
    rig.advance(1)
    rig.answer("NOT_OK")  # "no" to the break offer
    rig.advance(1)
    assert len(rig.spoken("break_due")) == 1


# Lifecycle


def test_nothing_before_a_trip_and_after_stop() -> None:
    rig = Rig(at(10))
    rig.advance(60 * 60)
    assert rig.sent == []
    rig.start()
    rig.sched.on_trip({"action": "stop"})
    rig.advance(60 * 60)
    assert rig.spoken("checkin_due") == []


def test_demo_profile_shows_the_script_values() -> None:
    """Demo step 1: 5 h sleep shows check-ins every 15 min and a break at 1:45 (LEAD-T21)."""
    demo = load_config("demo")["core"]["checkin"]
    minutes, risk = base_interval(pretrip(sleep_h=5), at(21), demo)
    assert minutes == 15 and risk and demo["break_risk_min"] == 105
