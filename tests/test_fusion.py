"""LEAD-T17: road-rage fusion [RR-04, RR-05, RR-07, SW-03], driven by explicit times."""

from __future__ import annotations

from typing import Any

import pytest

from backend.config import load_config
from backend.core.fusion import RageFusion
from backend.schemas import validate

CORE = load_config("default")["core"]
RAGE, HR = CORE["rage"], CORE["hr"]


class Rig:
    def __init__(self, rage: dict[str, Any] = RAGE) -> None:
        self.events: list[dict[str, Any]] = []
        self.suppressed: list[tuple[dict[str, Any], str]] = []
        self.fusion = RageFusion(
            rage, HR, self.emit, lambda data, why: self.suppressed.append((data, why))
        )
        self.t = 0.0

    def emit(self, data: dict[str, Any]) -> None:
        validate("detector.event", data)  # what core publishes must match the contract
        self.events.append(data)

    def face(self, anger: float, present: bool = True) -> None:
        self.fusion.on_driver_state({"anger_p": anger, "face_present": present}, self.t)

    def audio(self, swears: int = 0, shouting: bool = False) -> None:
        self.fusion.on_audio({"swear_count_60s": swears, "shouting": shouting}, self.t)

    def drive(self, long_g: float = 0.0, lat_g: float = 0.0) -> None:
        self.fusion.on_vehicle({"accel_long_g": long_g, "accel_lat_g": lat_g}, self.t)

    def hr(self, bpm: float) -> None:
        self.fusion.on_hr({"bpm": bpm}, self.t)

    def wait(self, seconds: float, step: float = 0.1, **kw: Any) -> None:
        """Let time pass, feeding a calm frame and a calm audio frame each step."""
        end = self.t + seconds
        while self.t < end - 1e-9:
            self.t = round(self.t + step, 6)
            self.face(kw.get("anger", 0.02), kw.get("present", True))
            self.audio(kw.get("swears", 0), kw.get("shouting", False))
            self.drive()


@pytest.fixture
def rig() -> Rig:
    return Rig()


# Acceptance: swearing alone never triggers


def test_swearing_alone_never_triggers(rig: Rig) -> None:
    for swears in range(1, 30):  # however much the driver swears, for 5 minutes
        rig.audio(swears=swears)
        rig.wait(10, swears=swears)
    assert rig.events == [] and rig.suppressed == []
    assert rig.fusion.reading(rig.t).score < RAGE["trigger_score"]


def test_shouting_alone_never_triggers(rig: Rig) -> None:
    rig.wait(60, shouting=True)
    assert rig.events == []


def test_harsh_driving_alone_never_triggers(rig: Rig) -> None:
    for _ in range(10):
        rig.drive(long_g=-0.7)
        rig.wait(5)
    assert rig.events == []


# RR-05: two of three, or the face alone for 5 s


def test_frown_swear_and_shout_trigger_within_3_s(rig: Rig) -> None:
    """RR-05 acceptance: frown + swear + shout triggers within 3 s."""
    rig.audio(swears=1, shouting=True)
    rig.wait(3.1, anger=0.9, swears=1, shouting=True)  # frown from the first frame (0.1 s)
    assert len(rig.events) == 1  # once the frown has lasted 3 s (RR-01)
    event = rig.events[0]
    assert event["type"] == "road_rage" and event["tier"] == "T2" and event["score"] >= 60
    assert set(event["signals"]["agreeing"]) == {"face", "voice"}


def test_voice_and_harsh_driving_trigger(rig: Rig) -> None:
    """The live demo path today: shouting plus a harsh brake."""
    rig.drive(long_g=-0.6)
    rig.audio(shouting=True)
    assert len(rig.events) == 1
    assert rig.events[0]["score"] == pytest.approx(60.0)
    assert rig.events[0]["signals"]["agreeing"] == ["voice", "driving"]


def test_face_alone_needs_5_s(rig: Rig) -> None:
    rig.wait(4.8, anger=0.9)
    assert rig.events == []
    rig.wait(0.3, anger=0.9)
    assert len(rig.events) == 1 and rig.events[0]["signals"]["face_alone"]


def test_brief_frown_resets(rig: Rig) -> None:
    rig.wait(4, anger=0.9)
    rig.wait(0.2, anger=0.1)
    rig.wait(4, anger=0.9)
    assert rig.events == []


def test_no_face_means_no_face_anger(rig: Rig) -> None:
    rig.wait(10, anger=0.9, present=False)
    assert rig.events == []


def test_old_harsh_events_leave_the_window(rig: Rig) -> None:
    rig.drive(long_g=-0.6)
    rig.wait(RAGE["harsh_window_s"] + 1)
    rig.audio(shouting=True)
    assert rig.events == []


def test_firm_message_at_85(rig: Rig) -> None:
    """Inputs arrive one at a time, so an episode fires when the second modality agrees; it
    is firm when the stress bonuses already lift that to 85 (face + voice 75 + heart rate 10)."""
    rig.hr(RAGE["resting_hr"] + HR["stress_delta_bpm"] + 5)
    rig.wait(HR["stress_hold_s"] + 0.1)  # heart-rate stress established
    rig.wait(3.1, anger=0.95)  # a held frown: the face agrees, but alone for under 5 s
    assert rig.events == []
    rig.audio(shouting=True)
    assert rig.events[0]["score"] >= RAGE["firm_score"] and rig.events[0]["signals"]["firm"]
    assert rig.events[0]["signals"]["hr_stress"] is True


def test_two_modalities_at_60_are_not_firm(rig: Rig) -> None:
    rig.drive(long_g=-0.6)
    rig.audio(shouting=True)
    assert rig.events[0]["signals"]["firm"] is False


# RR-07: cooldown and at most 3 in 30 min


def episode(rig: Rig) -> None:
    rig.drive(long_g=-0.6)
    rig.audio(shouting=True)
    rig.wait(1, shouting=True)
    rig.wait(RAGE["harsh_window_s"] + 1)  # calm again: the episode ends


def test_cooldown_logs_but_does_not_emit(rig: Rig) -> None:
    episode(rig)  # over after ~62 s, still inside the 3 min cooldown
    rig.drive(long_g=-0.6)
    rig.audio(shouting=True)
    assert len(rig.events) == 1
    assert len(rig.suppressed) == 1 and "cooldown" in rig.suppressed[0][1]


def test_at_most_3_in_30_minutes(rig: Rig) -> None:
    for _ in range(4):
        episode(rig)
        rig.wait(RAGE["cooldown_s"] - RAGE["harsh_window_s"])  # past the cooldown
    assert len(rig.events) == 3
    assert "at most 3" in rig.suppressed[-1][1]


def test_one_event_per_episode(rig: Rig) -> None:
    rig.drive(long_g=-0.6)
    rig.wait(30, shouting=True)  # angry for 30 s straight
    assert len(rig.events) == 1 and rig.suppressed == []


# Bonuses


def test_heart_rate_stress_adds_to_the_score(rig: Rig) -> None:
    rig.hr(RAGE["resting_hr"] + HR["stress_delta_bpm"] + 5)
    rig.wait(HR["stress_hold_s"] + 1)
    reading = rig.fusion.reading(rig.t)
    assert reading.hr_stress and reading.score >= RAGE["hr_bonus"]
    assert rig.events == []  # a bonus is not a modality


def test_resting_hr_comes_from_the_pre_trip_form(rig: Rig) -> None:
    rig.fusion.on_trip({"action": "start", "pretrip": {"resting_hr": 50}})
    rig.hr(85)  # 35 above 50, but only 15 above the default 70
    rig.wait(HR["stress_hold_s"] + 1)
    assert rig.fusion.reading(rig.t).hr_stress


def test_music_turned_up_adds_to_the_score(rig: Rig) -> None:
    rig.fusion.on_music({"volume": 30}, rig.t)
    rig.wait(5)
    rig.fusion.on_music({"volume": 30 + RAGE["music_rise"]}, rig.t)
    assert rig.fusion.reading(rig.t).music_rise


# rage.score: the live reading the dashboard's rage meter shows


def test_snapshot_shows_a_harsh_brake_as_driving_and_matches_the_contract(rig: Rig) -> None:
    rig.wait(1)
    calm = rig.fusion.snapshot(rig.t)
    validate("rage.score", calm)
    assert calm["driving"] == 0.0 and calm["agreeing"] == []
    rig.drive(long_g=-0.6)  # the dashboard's Harsh brake injection
    braked = rig.fusion.snapshot(rig.t)
    validate("rage.score", braked)
    assert braked["agreeing"] == ["driving"]
    w = RAGE["weights"]
    assert braked["score"] == pytest.approx(100 * (w["face"] * braked["face"] + w["driving"]))
    assert braked["trigger_score"] == RAGE["trigger_score"]
    assert not rig.events  # driving alone never fires road_rage
