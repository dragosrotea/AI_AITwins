import asyncio
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter
from unittest.mock import AsyncMock, Mock

import miniaudio
import pytest

from backend.llm import client, generate_alert, generate_engagement, summarize_trip
from backend.llm.templates import english_fallback, english_templates
from backend.schemas import VoiceSay
from backend.voice import live
from backend.voice.prerender import CATALOG, speech_entries
from backend.voice.stt import Transcription
from backend.voice.tts import PlaybackTiming


@pytest.mark.parametrize(
    ("lang", "key", "text", "expected_key"),
    [
        ("ro", "drowsiness_ro_2", "Pari obosit. Poți trage pe dreapta?", "drowsiness_en_2"),
        ("en", "drowsiness_ro_2.wav", "Pari obosit. Poți trage pe dreapta?", "drowsiness_en_2"),
        ("en", "drowsiness_en_2", "Ești bine?", "drowsiness_en_2"),
        ("ro", None, "Ești bine?", "checkin_due_en_1"),
        ("en", None, "Opreste masina acum?", "checkin_due_en_1"),
        ("en", None, "Are you okay?", None),
    ],
)
def test_legacy_and_current_text_key_agreement(lang, key, text, expected_key):
    alert = VoiceSay(alert_id="test", tier="T2", lang=lang, text=text, audio_key=key)
    normalized = live.english_alert(alert)
    assert normalized.lang == "en"
    assert normalized.audio_key == expected_key
    assert normalized.text == (live.english_audio()[expected_key] if expected_key else text)


def test_english_runtime_wav_coverage():
    entries = speech_entries(CATALOG)
    assert len(entries) == 30
    for key, lang, text in entries:
        assert lang == "en"
        assert live.english_audio()[key] == text
        clip = live.read_clip(key)
        assert clip is not None and miniaudio.decode(clip).num_frames > 0
    assert live.read_clip("drowsiness_ro_1") is None
    assert live.read_clip("demo_ro") is None
    assert all(len(variants) == 3 for variants in english_templates().values())


def test_legacy_romanian_key_loads_matching_english_wav():
    alert = VoiceSay(
        alert_id="archived-key",
        tier="T2",
        lang="ro",
        text="Ești bine?",
        audio_key="drowsiness_ro_2",
    )
    normalized = live.english_alert(alert)
    assert normalized.audio_key == "drowsiness_en_2"
    assert normalized.text == english_fallback("drowsiness", 2).text
    clip = live.read_clip(normalized.audio_key)
    assert clip is not None and clip == live.read_clip("drowsiness_en_2")
    assert miniaudio.decode(clip).num_frames > 0


@pytest.mark.parametrize(
    ("transcript", "path"), [("Yes, I'm okay.", "keyword"), ("I feel all right.", "llm")]
)
async def test_provider_timeout_plays_keyed_wav_and_routes_english_reply(
    monkeypatch, transcript, path
):
    async def slow_provider(*_args, **_kwargs):
        await asyncio.sleep(5)

    monkeypatch.setattr(client, "request_text", slow_provider)
    assert await generate_alert({"lang": "ro", "type": "drowsiness"}, timeout_s=0.1) is None
    text, key = english_fallback("drowsiness")
    alert = VoiceSay(alert_id="timeout", tier="T2", lang="en", text=text, audio_key=key)

    provider = AsyncMock(return_value="OK")
    now = perf_counter()
    playback = AsyncMock(return_value=PlaybackTiming(now, now))
    synthesize = AsyncMock(side_effect=AssertionError("Offline fallback must use its local WAV"))
    monkeypatch.setattr(client, "request_text", provider)
    monkeypatch.setattr(live, "play_interruptible", playback)
    monkeypatch.setattr(live, "synthesize", synthesize)
    monkeypatch.setattr(live, "capture_reply", AsyncMock(return_value=(b"pcm", perf_counter())))
    bus = Mock(publish=AsyncMock())
    mic = Mock()
    mic.cursor.return_value = 1
    stt = Mock()
    stt.transcribe.return_value = Transcription(transcript, 1)
    with ThreadPoolExecutor(max_workers=1) as inference:
        await live.interact(bus, mic, stt, inference, alert)

    playback.assert_awaited_once_with(live.read_clip(key), chime=live.read_clip("chime"))
    synthesize.assert_not_awaited()
    stt.transcribe.assert_called_once_with(b"pcm", "en")
    reply = next(
        data
        for topic, data in (call.args for call in bus.publish.await_args_list)
        if topic == "voice.reply"
    )
    assert (reply["intent"], reply["path"]) == ("OK", path)
    assert provider.await_count == (path == "llm")


async def test_legacy_trivia_context_and_summary_stay_english(monkeypatch):
    monkeypatch.setattr(client, "request_text", AsyncMock(return_value=None))
    assert (await generate_engagement({"lang": "ro"}))["text"] == "What is the capital of France?"
    assert await summarize_trip([]) == "Thank you for driving with DriveMate."
