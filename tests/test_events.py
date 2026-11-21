"""CV detector events to alerts: LEAD-T11 (first alert), LEAD-T15 (routing, DD-04), LEAD-T13."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from backend.config import load_config
from backend.core.alerts import AlertManager
from backend.core.events import TEMPLATES, EventRouter
from backend.core.wording import AlertWriter
from backend.llm.templates import english_fallback
from backend.schemas import validate

_core = load_config("default")["core"]
# Routing tests submit alerts seconds apart: run them without the AL-02 rate limits.
CFG = {
    **_core,
    "alerts": {
        **_core["alerts"],
        "t1_min_interval_s": 0,
        "noncritical_min_interval_s": 0,
        "cooldown_s": {},
    },
}
ROUTING = CFG["routing"]
VISUAL_TYPES = ["distraction", "microsleep", "drowsiness", "phone", "no_face"]


async def no_llm(ctx: dict[str, Any], timeout_s: float) -> str | None:
    return None


class Rig:
    def __init__(self, generate: Any = no_llm) -> None:
        self.sent: list[tuple[str, dict[str, Any]]] = []
        self.am = AlertManager(CFG["alerts"], self.publish, clock=lambda: 0.0)
        writer = AlertWriter(CFG["alerts"]["llm_timeout_s"], TEMPLATES, generate=generate)
        self.router = EventRouter(self.am, ROUTING, writer)

    def publish(self, topic: str, data: dict[str, Any]) -> None:
        validate(topic, data)
        self.sent.append((topic, data))

    def handle(self, kind: str = "distraction", src: str = "cv", **data: Any) -> Any:
        return asyncio.run(self.router.handle(event(kind, src, **data)))

    def says(self) -> list[dict[str, Any]]:
        return [d for t, d in self.sent if t == "voice.say"]


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
            "signals": {"yaw": 41.0},
            **data,
        },
    }
    validate("detector.event", envelope["data"])
    return envelope


@pytest.fixture
def rig() -> Rig:
    return Rig()


# LEAD-T11


def test_distraction_becomes_spoken_t2_alert(rig: Rig) -> None:
    alert = rig.handle()
    assert alert is not None and alert.tier == "T2" and alert.type == "distraction"
    say = rig.says()[0]
    first = english_fallback("distraction", 1)
    assert say["tier"] == "T2" and say["lang"] == "en"
    assert say["text"] == first.text and say["audio_key"] == first.audio_key
    assert alert.short_text == TEMPLATES["distraction"]["en"][1]
    assert alert.signals == {
        "event_id": "e-1",
        "score": 72.0,
        "yaw": 41.0,
        "text_path": "template",  # AL-05: the path is logged with the alert
        "llm_ms": alert.signals["llm_ms"],
    }


def test_english_text_is_spoken_as_english_for_any_driver(rig: Rig) -> None:
    """The catalog and the LLM prompt are English only, so voice must not read it as Romanian."""
    rig.am.lang = "ro"
    rig.handle()
    assert rig.says()[0]["lang"] == "en"


def test_llm_text_is_spoken_without_audio_key() -> None:
    async def llm(ctx: dict[str, Any], timeout_s: float) -> str | None:
        return "Eyes ahead, can you refocus?"

    rig = Rig(generate=llm)
    alert = rig.handle()
    assert alert is not None and alert.signals["text_path"] == "llm"
    say = rig.says()[0]
    assert say["text"] == "Eyes ahead, can you refocus?" and "audio_key" not in say


def test_events_during_a_slow_llm_make_one_alert() -> None:
    async def slow(ctx: dict[str, Any], timeout_s: float) -> str | None:
        await asyncio.sleep(0.05)
        return "Are you with me?"

    rig = Rig(generate=slow)

    async def two_events() -> list[Any]:
        return await asyncio.gather(
            rig.router.handle(event(event_id="e-1")), rig.router.handle(event(event_id="e-2"))
        )

    results = asyncio.run(two_events())
    assert sum(r is not None for r in results) == 1 and len(rig.says()) == 1


def test_repeated_events_make_one_alert(rig: Rig) -> None:
    rig.handle()
    assert rig.handle(event_id="e-2") is None
    assert len(rig.says()) == 1


def test_core_events_and_unknown_types_are_skipped(rig: Rig) -> None:
    assert rig.handle(src="core") is None  # fusion events never loop back
    assert rig.handle("road_rage") is None  # a core fusion type, no visual route
    assert rig.sent == []


# LEAD-T15: every visual type routed at its configured tier


@pytest.mark.parametrize("kind", VISUAL_TYPES)
def test_each_visual_type_is_routed_at_its_tier(rig: Rig, kind: str) -> None:
    alert = rig.handle(kind)
    assert alert is not None and alert.tier == ROUTING["tiers"][kind]
    assert rig.says()[0]["text"] == english_fallback(kind, 1).text


def test_tiers_follow_the_requirements() -> None:
    assert ROUTING["tiers"] == {
        "distraction": "T2",  # DD-01
        "microsleep": "T3",  # DD-02
        "drowsiness": "T2",  # DD-03
        "phone": "T2",  # PH-02
        "no_face": "T2",  # CV-06
    }


def test_core_decides_the_tier_not_the_sender(rig: Rig) -> None:
    alert = rig.handle("microsleep", tier="T1")
    assert alert is not None and alert.tier == "T3"


def test_microsleep_preempts_a_playing_distraction(rig: Rig) -> None:
    rig.handle("distraction")
    micro = rig.handle("microsleep")
    assert rig.am.active is micro and rig.says()[-1]["tier"] == "T3"
    assert [a.type for a in rig.am.queue] == ["distraction"]


# DD-04: drowsiness and microsleep override trivia


def queue_trivia(rig: Rig) -> Any:
    rig.handle("phone")  # something active, so the trivia waits in the queue
    trivia = rig.am.submit("zoning_out", "T1", "Quick one: capital of Portugal?")
    assert trivia is not None and trivia in rig.am.queue
    return trivia


@pytest.mark.parametrize("kind", ["drowsiness", "microsleep"])
def test_tired_driver_cancels_queued_trivia(rig: Rig, kind: str) -> None:
    trivia = queue_trivia(rig)
    rig.handle(kind)
    assert trivia not in rig.am.queue
    closed = [d for t, d in rig.sent if t == "alert.state" and d["alert_id"] == trivia.alert_id]
    assert closed[-1]["status"] == "closed"
    assert all(s["alert_id"] != trivia.alert_id for s in rig.says())  # never spoken


def test_other_events_keep_queued_trivia(rig: Rig) -> None:
    trivia = queue_trivia(rig)
    rig.handle("no_face")
    assert trivia in rig.am.queue


def test_cancel_leaves_the_active_alert(rig: Rig) -> None:
    trivia = rig.am.submit("zoning_out", "T1", "Quick one?")
    assert rig.am.active is trivia
    rig.handle("drowsiness")
    assert rig.am.active is trivia  # voice would ignore a non-T3 interruption anyway


# Templates


@pytest.mark.parametrize("lang", ["en", "ro"])
def test_every_routed_type_has_templates_in_both_languages(lang: str) -> None:
    for kind in ROUTING["tiers"]:
        text, short = TEMPLATES[kind][lang]
        for line in (text, short):
            assert line.endswith("?"), line  # VR-01: needs a spoken reply
            assert len(line.split()) <= 25, line  # LLM rule: at most 25 words


def test_templates_never_ask_to_look_or_touch() -> None:
    banned = ("look at", "screen", "tap", "touch", "press", "uită-te", "atinge", "apasă")
    for langs in TEMPLATES.values():
        for text, short in langs.values():
            assert not any(word in (text + short).lower() for word in banned)
