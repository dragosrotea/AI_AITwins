import asyncio
import csv
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter
from unittest.mock import AsyncMock, Mock

import pytest

from backend.config import load_config
from backend.llm import client
from backend.schemas import VoiceSay
from backend.voice import intents, live
from backend.voice.stt import Transcription
from backend.voice.tts import PlaybackTiming

with (
    Path(__file__)
    .with_name("confirmation_replies.csv")
    .open(encoding="utf-8", newline="") as catalog
):
    REPLIES = [row for row in csv.DictReader(catalog) if row["lang"] == "en"]


def test_reply_set_coverage():
    assert len({intents.normalize(row["text"]) for row in REPLIES}) == len(REPLIES) == 32
    for category in ("yes", "no", "tired", "help"):
        rows = [row for row in REPLIES if row["category"] == category]
        assert len(rows) == 8
        assert {row["route"] for row in rows} == {"keyword", "llm"}


@pytest.mark.parametrize("row", REPLIES, ids=[row["phrase_id"] for row in REPLIES])
async def test_labeled_confirmation_reply(monkeypatch, row):
    # Simulate provider output after the network boundary. T09 still validates it.
    provider = AsyncMock(return_value=row["intent"])
    monkeypatch.setattr(client, "request_text", provider)
    result = await intents.classify_confirmation(row["text"])
    assert (result.intent, result.path, result.fallback_failed) == (
        row["intent"],
        row["route"],
        False,
    )
    if row["route"] == "keyword":
        provider.assert_not_awaited()
    else:
        provider.assert_awaited_once()
        assert json.loads(provider.await_args.args[0]) == {
            "transcript": row["text"],
            "expecting": "confirmation",
        }


async def test_failed_provider_is_visible_to_evaluator(monkeypatch):
    monkeypatch.setattr(client, "request_text", AsyncMock(return_value=None))
    result = await intents.classify_confirmation("I feel all right.")
    assert result == intents.ReplyClassification("UNCLEAR", "llm", fallback_failed=True)


@pytest.mark.parametrize(
    "transcript",
    [
        "Turn and keep going.",
        "I'll see you in the next video.",
        "Okay, turn left.",
        "Yes, turn left.",
    ],
)
async def test_unrelated_reply_cannot_be_confirmed_by_provider(monkeypatch, transcript):
    provider = AsyncMock(return_value="OK")
    monkeypatch.setattr(client, "request_text", provider)

    assert await intents.classify_confirmation(transcript) == intents.ReplyClassification(
        "UNCLEAR", "llm"
    )
    provider.assert_awaited_once()


@pytest.mark.parametrize(
    ("transcript", "expected", "path"),
    [
        ("Yes.", "OK", "keyword"),
        ("No.", "NOT_OK", "keyword"),
        ("I feel all right.", "OK", "llm"),
        ("I feel alright.", "OK", "llm"),
    ],
)
async def test_genuine_short_and_llm_confirmations(monkeypatch, transcript, expected, path):
    provider = AsyncMock(return_value="OK")
    monkeypatch.setattr(client, "request_text", provider)

    assert await intents.classify_confirmation(transcript) == intents.ReplyClassification(
        expected, path
    )
    assert provider.await_count == (path == "llm")


@pytest.mark.parametrize(
    ("transcript", "intent"),
    [
        ("Yes, I can", "OK"),
        ("I'm safe", "OK"),
        ("I'm feeling okay", "OK"),
        ("I'm feeling alright", "OK"),
        ("I'm feeling good", "OK"),
        ("Yes, but I'm tired", "NOT_OK"),
        ("I'm not feeling okay", "NOT_OK"),
        ("No, I need help", "HELP"),
    ],
)
async def test_common_confirmations_work_offline(monkeypatch, transcript, intent):
    provider = AsyncMock(return_value=None)
    monkeypatch.setattr(client, "request_text", provider)
    assert await intents.classify_confirmation(transcript) == intents.ReplyClassification(
        intent, "keyword"
    )
    provider.assert_not_awaited()


@pytest.mark.parametrize(
    ("transcript", "intent", "path"),
    [
        ("There's a rest area coming up. I'll stop there", "OK", "llm"),
        ("Turn and keep going", "UNCLEAR", "llm"),
        ("Maybe I'll stop at the next exit", "UNCLEAR", "llm"),
        ("I'm feeling good", "UNCLEAR", "llm"),
        ("I won't stop", "NOT_OK", "keyword"),
        ("Yes, I can", "OK", "keyword"),
    ],
)
async def test_pull_over_confirmation_checks_context_even_if_provider_says_ok(
    monkeypatch, transcript, intent, path
):
    question = "Can you pull over somewhere safe?"
    provider = AsyncMock(return_value="OK")
    monkeypatch.setattr(client, "request_text", provider)
    result = await intents.classify_confirmation(transcript, question=question, timeout_s=2.5)
    assert result == intents.ReplyClassification(intent, path)
    if path == "llm":
        args = provider.await_args.args
        assert args[1] == 2.5
        assert json.loads(args[0]) == {
            "expecting": "confirmation",
            "question": question,
            "transcript": transcript,
        }
    else:
        provider.assert_not_awaited()


async def test_contextual_deadline_is_distinct_from_provider_failure(monkeypatch):
    async def slow(*_args, **_kwargs):
        await asyncio.sleep(5)

    assert load_config("default")["voice"]["reply_timeout_s"] == 1.5
    assert load_config("demo")["voice"]["reply_timeout_s"] == 2.5
    transcript = "There's a rest area coming up. I'll stop there"
    question = "Can you pull over somewhere safe?"
    monkeypatch.setattr(client, "request_text", slow)
    assert await intents.classify_confirmation(
        transcript, question=question, timeout_s=0.05
    ) == intents.ReplyClassification("UNCLEAR", "llm", fallback_failed=True, timed_out=True)

    monkeypatch.setattr(client, "request_text", AsyncMock(return_value=None))
    assert await intents.classify_confirmation(
        transcript, question=question
    ) == intents.ReplyClassification("UNCLEAR", "llm", fallback_failed=True)


async def test_driving_direction_is_not_a_refusal(monkeypatch):
    monkeypatch.setattr(client, "request_text", AsyncMock(return_value="NOT_OK"))
    assert await intents.classify_confirmation(
        "Turn and keep going", question="Can you pull over somewhere safe?"
    ) == intents.ReplyClassification("UNCLEAR", "llm")


@pytest.mark.parametrize(
    ("transcript", "intent"),
    [
        ("I'm not okay.", "NOT_OK"),
        ("I'm tired.", "NOT_OK"),
        ("Yes, I'm okay.", "OK"),
        ("Yes, but I'm tired. Help!", "HELP"),
    ],
)
async def test_legacy_alert_reply_uses_english_keyword_path(monkeypatch, transcript, intent):
    bus = Mock()
    bus.publish = AsyncMock()
    mic = Mock()
    stt = Mock()
    stt.transcribe.return_value = Transcription(transcript, 0)
    fallback = AsyncMock(return_value="UNCLEAR")
    monkeypatch.setattr(intents, "classify_reply_with_status", fallback)
    monkeypatch.setattr(live, "read_clip", lambda _key: b"clip")
    now = perf_counter()
    monkeypatch.setattr(
        live, "play_interruptible", AsyncMock(return_value=PlaybackTiming(now, now))
    )
    monkeypatch.setattr(live, "capture_reply", AsyncMock(return_value=(b"pcm", perf_counter())))
    alert = VoiceSay(
        alert_id="t07-ro", text="Ești bine?", tier="T2", lang="ro", audio_key="demo_ro"
    )

    with ThreadPoolExecutor(max_workers=1) as executor:
        await live.interact(bus, mic, stt, executor, alert)

    [reply] = [
        data
        for topic, data in (call.args for call in bus.publish.await_args_list)
        if topic == "voice.reply"
    ]
    assert reply["transcript"] == transcript
    assert reply["intent"] == intent
    assert reply["path"] == "keyword"
    stt.transcribe.assert_called_once_with(b"pcm", "en")
    fallback.assert_not_awaited()
