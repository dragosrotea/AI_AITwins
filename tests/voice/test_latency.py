import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from io import StringIO
from unittest.mock import AsyncMock, Mock

import pytest

from backend.schemas import VoiceSay
from backend.voice import live
from backend.voice.intents import ReplyClassification
from backend.voice.latency import LatencySample, report
from backend.voice.stt import Transcription
from backend.voice.tts import PlaybackTiming


@pytest.mark.parametrize(
    ("path", "provider_failed", "outcome"),
    [("keyword", False, "reply"), ("llm", False, "reply"), ("llm", True, "provider_failure")],
)
async def test_timing_starts_at_last_speech_frame(monkeypatch, path, provider_failed, outcome):
    now = 10.0

    async def synthesize(_text, _lang):
        nonlocal now
        now = 10.65
        return b"alert"

    async def playback(clip, *, chime):
        nonlocal now
        assert (clip, chime) == (b"alert", b"chime")
        now = 11.8  # Device cleanup returns after the chime's last sample.
        return PlaybackTiming(10.72, 11.6)

    async def capture(_mic, _cursor, deadline, *, started_at, alert_id):
        nonlocal now
        assert alert_id == "private-id"
        assert started_at == 11.6
        assert deadline == pytest.approx(19.6)
        now = 12.86  # 810 ms after the last detected speech frame.
        return b"raw audio", 12.05

    def transcribe(_pcm, lang):
        nonlocal now
        assert lang == "en"
        now = 13.01
        return Transcription("private transcript", 150)

    async def classify(_text):
        nonlocal now
        now = 13.13
        return ReplyClassification("UNCLEAR" if provider_failed else "OK", path, provider_failed)

    monkeypatch.setattr(live, "perf_counter", lambda: now)
    monkeypatch.setattr(live, "english_alert", lambda alert: alert)
    monkeypatch.setattr(live, "read_clip", lambda _key: b"chime")
    monkeypatch.setattr(live, "synthesize", synthesize)
    monkeypatch.setattr(live, "play_interruptible", playback)
    monkeypatch.setattr(live, "capture_reply", capture)
    monkeypatch.setattr(live, "classify_confirmation", classify)
    bus = Mock(publish=AsyncMock())
    log = StringIO()
    alert = VoiceSay(alert_id="private-id", text="Are you okay?", tier="T2", lang="en")
    with ThreadPoolExecutor(max_workers=1) as executor:
        await live.interact(
            bus, Mock(), Mock(transcribe=transcribe), executor, alert, latency_log=log
        )

    reply = next(
        data
        for topic, data in (c.args for c in bus.publish.await_args_list)
        if topic == "voice.reply"
    )
    sample = LatencySample(**json.loads(log.getvalue()))
    assert reply["latency_ms"] == 1080
    assert (sample.outcome, sample.path, sample.playback_source) == (outcome, path, "tts")
    assert sample.playback_start_ms == pytest.approx(720)
    assert sample.listening_ms == pytest.approx(1260)
    assert sample.whisper_ms == 150
    assert sample.classification_ms == pytest.approx(120)
    assert sample.speech_to_intent_ms == pytest.approx(1080)
    for private in ("private", "raw audio", alert.text):
        assert private not in log.getvalue()


@pytest.mark.parametrize("interrupted", [False, True])
async def test_silence_and_cancelled_capture_are_recorded_separately(monkeypatch, interrupted):
    monkeypatch.setattr(live, "english_alert", lambda alert: alert)
    monkeypatch.setattr(live, "read_clip", lambda _key: b"wav")
    monkeypatch.setattr(live, "play_interruptible", AsyncMock(return_value=PlaybackTiming(0, 1)))
    capture = AsyncMock(
        return_value=None, side_effect=asyncio.CancelledError if interrupted else None
    )
    monkeypatch.setattr(live, "capture_reply", capture)
    bus = Mock(publish=AsyncMock())
    alert = VoiceSay(
        alert_id="test", text="Are you okay?", tier="T2", lang="en", audio_key="demo_en"
    )
    log = StringIO()
    with ThreadPoolExecutor(max_workers=1) as executor:
        interaction = live.interact(bus, Mock(), Mock(), executor, alert, latency_log=log)
        if interrupted:
            with pytest.raises(asyncio.CancelledError):
                await interaction
        else:
            await interaction
    sample = LatencySample(**json.loads(log.getvalue()))
    assert sample.outcome == ("interrupted" if interrupted else "silence")
    assert sample.speech_to_intent_ms is None
    messages = [
        data
        for topic, data in (c.args for c in bus.publish.await_args_list)
        if topic == "voice.reply"
    ]
    assert messages == (
        []
        if interrupted
        else [
            {
                "alert_id": "test",
                "transcript": "",
                "intent": "NONE",
                "latency_ms": 0,
                "path": "keyword",
            }
        ]
    )


def test_report_excludes_unsuccessful_replies_and_splits_paths():
    text = report(
        [
            LatencySample(
                "reply", "keyword", "tts", playback_start_ms=500, speech_to_intent_ms=900
            ),
            LatencySample("reply", "llm", "wav", playback_start_ms=100, speech_to_intent_ms=1500),
            LatencySample(
                "reply", "keyword", "wav", playback_start_ms=140, speech_to_intent_ms=1000
            ),
            LatencySample(
                "provider_failure", "llm", "tts", playback_start_ms=1, speech_to_intent_ms=9000
            ),
            LatencySample("silence", listening_ms=8000),
            LatencySample("interrupted"),
        ]
    )
    assert "reply=3, silence=1, interrupted=1, provider_failure=1" in text
    assert "TTS request to playback start: n=1, median=500.0" in text
    assert "Local WAV request to playback start: n=2, median=120.0" in text
    assert "End of speech to intent (overall): n=3, median=1000.0" in text
    assert "End of speech to intent (keyword): n=2, median=950.0" in text
    assert "End of speech to intent (llm): n=1, median=1500.0" in text
