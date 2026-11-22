"""LEAD-T19: zoning-out fusion (ZO-01, ZO-06), core zoning_out to trivia, ZO-05 and ZO-04."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock

import pytest

from backend.config import load_config
from backend.core.alerts import AlertManager
from backend.core.events import TEMPLATES, EventRouter
from backend.core.wording import AlertWriter
from backend.core.zoning import ZoningFusion
from backend.schemas import validate

CORE = load_config("default")["core"]


@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never call the real LLM from tests: trivia and grading use their offline fallbacks."""
    from backend.llm import client

    monkeypatch.setattr(client, "request_text", AsyncMock(return_value=None))


DEMO = load_config("demo")["core"]
ZONING = CORE["zoning"]


class Fusion:
    def __init__(self, cfg: dict[str, Any] = ZONING) -> None:
        self.events: list[dict[str, Any]] = []
        self.z = ZoningFusion(cfg, self.emit)
        self.t = 0.0

    def emit(self, data: dict[str, Any]) -> None:
        validate("detector.event", data)
        self.events.append(data)

    def drive(self, seconds: float, speed: float, visual: float = 0.0, face: bool = True) -> None:
        end = self.t + seconds
        while self.t < end - 1e-9:
            self.t = round(self.t + 0.1, 6)
            self.z.on_vehicle({"speed_kmh": speed}, self.t)
            self.z.on_driver_state({"zoning_visual_p": visual, "face_present": face}, self.t)


# ZO-01: visual score plus monotony


def test_monotony_plus_a_staring_driver_raises_zoning_out() -> None:
    f = Fusion()
    f.drive(ZONING["monotony_window_s"], 100)
    assert f.z.monotonous(f.t)
    f.drive(1, 100, visual=0.5)  # 0.7 * 0.5 + 0.3 = 0.65
    assert len(f.events) == 1
    event = f.events[0]
    assert event["type"] == "zoning_out" and event["tier"] == "T1" and event["score"] == 65.0
    assert event["signals"] == {"visual": 0.5, "monotony": True}


def test_default_stare_without_monotony_is_not_enough() -> None:
    f = Fusion()
    f.drive(60, 50, visual=0.82)  # city speed: 0.7 * 0.82 = 0.57
    assert f.events == []


def test_demo_profile_stare_alone_crosses_the_threshold() -> None:
    """ZO-01 acceptance: a fixed stare in demo mode for 30 s raises the score above threshold."""
    f = Fusion(DEMO["zoning"])
    f.drive(1, 50, visual=0.82)  # CV's zoning_visual_p already covers the 30 s stare
    assert len(f.events) == 1


@pytest.mark.parametrize(
    ("speeds", "monotonous"),
    [
        ([100] * 2, True),
        ([100, 103], True),  # 3 km/h spread
        ([100, 106], False),  # 6 km/h spread
        ([65, 66], False),  # below 70 km/h
    ],
)
def test_monotony_rule(speeds: list[float], monotonous: bool) -> None:
    f = Fusion()
    half = ZONING["monotony_window_s"] / 2
    for speed in speeds:
        f.drive(half, speed)
    assert f.z.monotonous(f.t) is monotonous


def test_monotony_needs_the_whole_window() -> None:
    f = Fusion()
    f.drive(ZONING["monotony_window_s"] / 2, 100)
    assert not f.z.monotonous(f.t)


def test_no_face_means_no_visual_score() -> None:
    f = Fusion()
    f.drive(ZONING["monotony_window_s"], 100, visual=0.9, face=False)
    assert f.events == []  # monotony alone is 0.3


def test_one_event_per_episode_and_rearm() -> None:
    f = Fusion(DEMO["zoning"])
    f.drive(30, 50, visual=0.9)
    assert len(f.events) == 1
    f.drive(1, 50, visual=0.1)
    f.drive(1, 50, visual=0.9)
    assert len(f.events) == 2


# ZO-06: never in high-load zones or stop-and-go


@pytest.mark.parametrize("road", [{"load": "HIGH"}, {"load": "LOW", "traffic": "stop-and-go"}])
def test_busy_road_holds_zoning_out(road: dict[str, Any]) -> None:
    f = Fusion(DEMO["zoning"])
    f.z.on_road(road)
    f.drive(10, 50, visual=0.9)
    assert f.events == []
    f.z.on_road({"load": "LOW"})
    f.drive(1, 50, visual=0.9)
    assert len(f.events) == 1  # once the road is calm again


# Routing: core's zoning_out becomes trivia; ZO-05; ZO-04 feedback


async def no_llm(ctx: dict[str, Any], timeout_s: float) -> str | None:
    return None


class Router:
    def __init__(self) -> None:
        self.now = 1000.0
        self.sent: list[tuple[str, dict[str, Any]]] = []
        alerts = {**CORE["alerts"], "t1_min_interval_s": 0, "noncritical_min_interval_s": 0}
        alerts["cooldown_s"] = {}
        self.am = AlertManager(alerts, self.publish, clock=lambda: self.now)
        writer = AlertWriter(2.0, TEMPLATES, generate=no_llm)
        self.router = EventRouter(self.am, CORE["routing"], writer, ZONING, clock=lambda: self.now)
        self.router.on_trip({"action": "start", "pretrip": {"engagement": "trivia"}})

    def publish(self, topic: str, data: dict[str, Any]) -> None:
        validate(topic, data)
        self.sent.append((topic, data))

    def zoning_out(self) -> Any:
        envelope = {
            "topic": "detector.event",
            "ts": 1.0,
            "src": "core",
            "data": {
                "event_id": "z",
                "type": "zoning_out",
                "tier": "T1",
                "score": 70.0,
                "signals": {},
            },
        }
        return asyncio.run(self.router.handle(envelope))

    def close_active(self) -> None:
        alert = self.am.active
        assert alert is not None
        self.am.on_voice_reply(
            {
                "alert_id": alert.alert_id,
                "transcript": "x",
                "intent": "OK",
                "latency_ms": 1,
                "path": "keyword",
            }
        )
        self.now += CORE["alerts"]["voice_gap_s"] + 0.5  # past the pause before the next say
        self.am.tick()


@pytest.fixture
def router() -> Router:
    return Router()


def test_core_zoning_out_becomes_trivia(router: Router) -> None:
    alert = router.zoning_out()
    assert alert is not None and alert.type == "zoning_out" and alert.tier == "T1"
    assert alert.engagement_answer  # a trivia question with an expected answer


def test_quiet_drivers_get_no_trivia(router: Router) -> None:
    router.router.on_trip({"action": "start", "pretrip": {"engagement": "quiet"}})
    assert router.zoning_out() is None


def test_fourth_zoning_out_recommends_a_break(router: Router) -> None:
    """ZO-05 acceptance: the fourth trigger produces a break recommendation."""
    for _ in range(ZONING["max_engagements"]):
        assert router.zoning_out().type == "zoning_out"
        router.close_active()
    fourth = router.zoning_out()
    assert fourth is not None and fourth.type == "break_due"


def test_engagements_age_out_of_the_window(router: Router) -> None:
    for _ in range(ZONING["max_engagements"]):
        router.zoning_out()
        router.close_active()
    router.now += ZONING["engagement_window_s"] + 1
    assert router.zoning_out().type == "zoning_out"


@pytest.mark.parametrize(("reply", "correct"), [("it's Paris", True), ("London", False)])
def test_graded_answer_is_spoken_with_a_question(router: Router, reply: str, correct: bool) -> None:
    """ZO-04: graded leniently, one short sentence of feedback, ending in a question (VR-01)."""
    alert = router.am.submit(
        "zoning_out", "T1", "What is the capital of France?", engagement_answer="Paris"
    )
    router.am.on_voice_reply(
        {
            "alert_id": alert.alert_id,
            "transcript": reply,
            "intent": "ANSWER",
            "latency_ms": 1,
            "path": "keyword",
        }
    )
    feedback = asyncio.run(router.router.answer_feedback(alert, {"transcript": reply}))
    assert feedback is not None and feedback.signals["correct"] is correct
    assert feedback.text.endswith(ZONING["feedback_question"])
    if not correct:
        assert "Paris" in feedback.text  # tells the right answer
    assert feedback.engagement_answer is None  # the reply to it is not graded again


def test_feedback_is_not_held_by_rate_limits() -> None:
    rig = Router()
    rig.am.cfg = CORE["alerts"]  # the real AL-02 limits
    alert = rig.am.submit("zoning_out", "T1", "Capital of France?", engagement_answer="Paris")
    rig.am.on_voice_reply(
        {
            "alert_id": alert.alert_id,
            "transcript": "Paris",
            "intent": "ANSWER",
            "latency_ms": 1,
            "path": "keyword",
        }
    )
    assert asyncio.run(rig.router.answer_feedback(alert, {"transcript": "Paris"})) is not None


# Demo trigger: control.inject zoning_out (dashboard "Zone out (trivia)" button)


def test_injected_zoning_out_needs_no_signals_and_becomes_trivia(router: Router) -> None:
    f = Fusion()
    f.z.inject()  # no driving, no stare: the button alone raises the event
    [event] = f.events
    assert event["type"] == "zoning_out" and event["signals"] == {"injected": True}
    envelope = {"topic": "detector.event", "ts": 1.0, "src": "core", "data": event}
    alert = asyncio.run(router.router.handle(envelope))
    assert alert is not None and alert.engagement_answer


def test_injected_zoning_out_asks_even_for_a_quiet_driver(router: Router) -> None:
    router.router.on_trip({"action": "start", "pretrip": {"engagement": "quiet"}})
    f = Fusion()
    f.z.inject()
    envelope = {"topic": "detector.event", "ts": 1.0, "src": "core", "data": f.events[0]}
    alert = asyncio.run(router.router.handle(envelope))
    assert alert is not None and alert.engagement_answer  # the button forces trivia


def test_vehicle_injector_ignores_zoning_out() -> None:
    from backend.core.vehicle import Injector

    injector = Injector(CORE["inject"], clock=lambda: 0.0)
    injector.inject("zoning_out")
    sample = {"accel_long_g": 0.1, "accel_lat_g": 0.0, "speed_kmh": 90.0}
    assert injector.apply(sample)["accel_long_g"] == 0.1 and not injector.crashed
    validate("control.inject", {"event": "zoning_out"})


def _limited_router() -> Router:
    """A router with the real (default-profile) AL-02 limits and cooldowns."""
    r = Router()
    r.am = AlertManager(CORE["alerts"], r.publish, clock=lambda: r.now)
    writer = AlertWriter(2.0, TEMPLATES, generate=no_llm)
    r.router = EventRouter(r.am, CORE["routing"], writer, ZONING, clock=lambda: r.now)
    r.router.on_trip({"action": "start", "pretrip": {"engagement": "trivia"}})
    return r


def _envelope(signals: dict[str, Any]) -> dict[str, Any]:
    data = {"event_id": "z", "type": "zoning_out", "tier": "T1", "score": 100.0}
    return {
        "topic": "detector.event",
        "ts": 1.0,
        "src": "core",
        "data": {**data, "signals": signals},
    }


def test_injected_zoning_out_skips_rate_limits_but_fused_one_does_not() -> None:
    r = _limited_router()
    assert asyncio.run(r.router.handle(_envelope({}))) is not None
    r.close_active()
    assert asyncio.run(r.router.handle(_envelope({}))) is None  # AL-02 / cooldown hold it
    assert asyncio.run(r.router.handle(_envelope({"injected": True}))) is not None


def test_injected_zoning_out_is_asked_on_a_busy_road() -> None:
    r = _limited_router()
    r.router.on_road({"load": "HIGH"})
    r.am.on_road({"load": "HIGH"})
    alert = asyncio.run(r.router.handle(_envelope({"injected": True})))
    assert alert is not None and r.am.active is alert  # spoken now, not held


def test_injected_zoning_out_keeps_asking_past_zo05_and_after_trip_stop() -> None:
    r = _limited_router()
    for _ in range(ZONING["max_engagements"] + 2):
        alert = asyncio.run(r.router.handle(_envelope({"injected": True})))
        assert alert is not None and alert.type == "zoning_out" and alert.engagement_answer
        r.close_active()
    r.router.on_trip({"action": "stop"})
    assert asyncio.run(r.router.handle(_envelope({"injected": True}))) is not None


def test_injected_zoning_out_queues_behind_another_alert() -> None:
    r = _limited_router()
    other = r.am.submit("distraction", "T2", "Eyes on the road.")
    assert other is not None and r.am.active is other
    alert = asyncio.run(r.router.handle(_envelope({"injected": True})))
    assert alert is not None and alert in r.am.queue  # waits, then plays
    r.close_active()
    assert r.am.active is alert


def test_second_press_while_asking_does_not_stack() -> None:
    r = _limited_router()
    first = asyncio.run(r.router.handle(_envelope({"injected": True})))
    assert first is not None and r.am.active is first
    assert asyncio.run(r.router.handle(_envelope({"injected": True}))) is None


def test_button_trivia_does_not_repeat_when_the_llm_always_says_france(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import json

    from backend.llm import client

    france = json.dumps(
        {"kind": "trivia", "text": "What is the capital of France?", "answer": "Paris"}
    )
    llm = AsyncMock(return_value=france)
    monkeypatch.setattr(client, "request_text", llm)
    r = _limited_router()
    asked = []
    for _ in range(5):
        alert = asyncio.run(r.router.handle(_envelope({"injected": True})))
        assert alert is not None
        asked.append(alert.text)
        r.close_active()
    assert len(set(asked)) == 5, asked  # France once, then the bank takes over
    payload = json.loads(llm.call_args.args[0])
    assert "What is the capital of France?" in payload["avoid"] and payload["topic"]
