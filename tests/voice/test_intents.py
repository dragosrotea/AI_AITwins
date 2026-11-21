import csv
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter
from unittest.mock import AsyncMock, Mock

import pytest

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
    monkeypatch.setattr(intents, "classify_reply", fallback)
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
