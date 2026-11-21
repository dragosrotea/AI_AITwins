import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter
from unittest.mock import AsyncMock, Mock

import miniaudio
import pytest

from backend.config import get_config
from backend.core.alerts import AlertManager
from backend.core.events import TEMPLATES, EventRouter
from backend.core.wording import AlertWriter
from backend.llm import client, generate_engagement, grade_answer
from backend.llm.engagement import engagement_audio_key, engagement_clips, trivia_bank
from backend.llm.replies import validate_alert
from backend.schemas import VoiceSay
from backend.voice import live
from backend.voice.stt import Transcription
from backend.voice.tts import PlaybackTiming


@pytest.mark.parametrize(
    ("text", "correct"),
    [
        ("Paris", True),
        ("It's Paris.", True),
        ("I think it's Paris.", True),
        ("London", False),
        ("Not Paris", False),
        ("I think it's not Paris", False),
        ("Paris or Rome", False),
    ],
)
async def test_bank_answer_variants_are_graded_offline(monkeypatch, text, correct):
    provider = AsyncMock()
    monkeypatch.setattr(client, "request_text", provider)
    result = await grade_answer("What is the capital of France?", "Paris", text)
    assert result == {"correct": correct, "feedback": "That's right." if correct else "Not quite."}
    provider.assert_not_awaited()


def test_bank_has_50_distinct_items_and_all_wavs_resolve():
    bank = trivia_bank()
    assert len(bank) == len({entry["question"] for entry in bank}) == 50
    assert len({entry["fact"] for entry in bank}) == 50
    for entry in bank:
        assert entry["accepted_answers"]
        assert all(1 <= len(answer.split()) <= 3 for answer in entry["accepted_answers"])
    assert len(engagement_clips()) == 100
    for key, text in engagement_clips().items():
        assert validate_alert(text) == text
        raw = live.read_clip(key)
        assert raw is not None and miniaudio.decode(raw).num_frames > 0


@pytest.mark.parametrize(
    "output", [None, "not JSON", '{"kind":"trivia","text":"Broken","answer":4}']
)
async def test_offline_fallback_for_unavailable_or_unusable_provider(monkeypatch, output):
    monkeypatch.setattr(client, "request_text", AsyncMock(return_value=output))
    result = await generate_engagement({"region": "Japan"})
    assert result == {"kind": "trivia", "text": "What is the capital of Japan?", "answer": "Tokyo"}
    assert engagement_audio_key(result["text"]) == "engagement_trivia_en_7"


async def test_generation_honors_timeout_and_fact_preference(monkeypatch):
    async def slow(*_args, **_kwargs):
        await asyncio.sleep(3)

    monkeypatch.setattr(client, "request_text", slow)
    result = await generate_engagement({"engagement": "fact", "region": "France"}, timeout_s=0.1)
    assert result == {
        "kind": "fact",
        "text": "The Seine runs through Paris. Did you know that?",
        "answer": None,
    }
    assert engagement_audio_key(result["text"]) == "engagement_fact_en_1"


async def test_online_engagement_receives_region_and_validates_answer(monkeypatch):
    item = {
        "kind": "trivia",
        "text": "Which river runs through the city of Paris?",
        "answer": "the Seine",
    }
    provider = AsyncMock(return_value=json.dumps(item))
    monkeypatch.setattr(client, "request_text", provider)
    assert await generate_engagement({"region": "Paris"}) == item
    assert json.loads(provider.await_args.args[0]) == {"kind": "trivia", "region": "Paris"}


def event(kind, tier, **signals):
    return {
        "src": "cv",
        "data": {"event_id": kind, "type": kind, "tier": tier, "score": 90, "signals": signals},
    }


@pytest.fixture
def routing(monkeypatch):
    cfg = get_config()["core"]
    monkeypatch.setattr(client, "request_text", AsyncMock(return_value=None))
    messages = []
    answers = []
    manager = AlertManager(
        cfg["alerts"],
        lambda topic, data: messages.append((topic, data)),
        on_answer=lambda alert, reply: answers.append((alert, reply)),
    )
    return EventRouter(manager, cfg["routing"], AlertWriter(2, TEMPLATES)), messages, answers


async def test_offline_event_routes_to_voice_and_grades_local_answer(routing, monkeypatch):
    router, messages, answers = routing
    event_data = event("zoning_out", "T1", region="France")
    assert await router.handle(event_data) is None  # Pre-trip has not opted in.
    router.on_trip({"action": "start", "pretrip": {"engagement": "trivia"}})
    router.on_road({"load": "HIGH"})
    assert await router.handle(event_data) is None
    router.on_road({"load": "LOW", "traffic": "stop_and_go"})
    assert await router.handle(event_data) is None
    router.on_road({"load": "LOW"})
    assert await router.handle({**event_data, "src": "core"}) is None
    alert = await router.handle(event_data)
    [say] = [data for topic, data in messages if topic == "voice.say"]
    assert say["audio_key"] == "engagement_trivia_en_1"
    assert alert.engagement_answer == "Paris"
    provider = AsyncMock(side_effect=AssertionError("Offline answer reached provider"))
    monkeypatch.setattr(client, "request_text", provider)
    monkeypatch.setattr(
        live,
        "play_interruptible",
        AsyncMock(return_value=PlaybackTiming(perf_counter(), perf_counter())),
    )
    monkeypatch.setattr(live, "capture_reply", AsyncMock(return_value=(b"pcm", perf_counter())))
    bus = Mock(publish=AsyncMock())
    mic = Mock(cursor=Mock(return_value=0))
    stt = Mock(transcribe=Mock(return_value=Transcription("I think it's Paris.", 1)))
    with ThreadPoolExecutor(max_workers=1) as executor:
        await live.interact(bus, mic, stt, executor, VoiceSay.model_validate(say), engagement=True)
    [reply] = [
        call.args[1] for call in bus.publish.await_args_list if call.args[0] == "voice.reply"
    ]
    assert reply["intent"] == "ANSWER"
    router.manager.on_voice_reply(reply)
    assert answers == [(alert, reply)]
    assert await grade_answer(alert.text, alert.engagement_answer, reply["transcript"]) == {
        "correct": True,
        "feedback": "That's right.",
    }
    provider.assert_not_awaited()


async def test_microsleep_cuts_off_trivia_and_it_does_not_resume(routing):
    router, messages, _answers = routing
    router.on_trip({"action": "start", "pretrip": {"engagement": "trivia"}})
    trivia = await router.handle(event("zoning_out", "T1"))
    critical = await router.handle(event("microsleep", "T3"))
    says = [data for topic, data in messages if topic == "voice.say"]
    assert [say["alert_id"] for say in says] == [trivia.alert_id, critical.alert_id]
    assert router.manager.active is critical
    assert trivia.status == "closed" and trivia not in router.manager.pending()
    assert await router.handle(event("zoning_out", "T1")) is None
