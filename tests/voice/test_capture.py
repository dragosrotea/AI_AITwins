from concurrent.futures import ThreadPoolExecutor
from struct import pack
from time import perf_counter
from unittest.mock import AsyncMock, Mock

import pytest

from backend.audio.devices import FRAME_MS, FRAME_SAMPLES, AudioFrame
from backend.schemas import VoiceSay
from backend.voice import live
from backend.voice.stt import Transcription
from backend.voice.tts import PlaybackTiming


async def test_capture_keeps_leading_and_trailing_frames_at_vad_cutoff(monkeypatch, caplog):
    started_at = perf_counter() - 2
    frames = [
        AudioFrame(i, bytes([i]) * 960, started_at + i * FRAME_MS / 1000) for i in range(1, 35)
    ]
    vad = Mock()
    vad.is_speech.side_effect = lambda pcm, _rate: 4 <= pcm[0] <= 6
    monkeypatch.setattr(live, "Vad", lambda _mode: vad)
    mic = Mock()
    mic.read_since.return_value = frames
    caplog.set_level("INFO", logger="backend.voice.capture")

    pcm, speech_end = await live.capture_reply(
        mic, 0, started_at + 1.5, started_at=started_at, alert_id="test-capture"
    )

    assert pcm == b"".join(frame.pcm for frame in frames[:11])
    assert speech_end == frames[5].ended_at
    assert vad.is_speech.call_count == 6 + 27
    assert "first_ms=120 last_ms=180 positive_frames=3 retained_ms=330" in caplog.text
    assert "first_frame_end_ms=30 masked_ms=0.0 signal_first_frame_end_ms=60" in caplog.text
    assert "signal_last_frame_end_ms=990 whisper_input_end_ms=330" in caplog.text
    assert "onset_min_peak_dbfs=-30.1 onset_max_peak_dbfs=-26.5" in caplog.text
    assert caplog.text.count("alert_id='test-capture'") == 2


async def test_silence_retains_no_audio(monkeypatch, caplog):
    started_at = perf_counter() - 1
    vad = Mock()
    vad.is_speech.return_value = False
    monkeypatch.setattr(live, "Vad", lambda _mode: vad)
    mic = Mock()
    mic.read_since.return_value = [AudioFrame(1, bytes(960), started_at + 0.03)]
    caplog.set_level("INFO", logger="backend.voice.capture")

    assert await live.capture_reply(mic, 0, started_at + 0.1, started_at=started_at) is None
    vad.is_speech.assert_called_once()
    assert "first_ms=none last_ms=none positive_frames=0" in caplog.text


@pytest.mark.parametrize(("wait", "expected_frames"), [(0.8, 41), (0.45, 29)])
async def test_endpoint_wait_uses_last_speech_timestamp_and_keeps_brief_pause(
    monkeypatch, wait, expected_frames
):
    # The second speech section starts after a brief gap and resets the timer.
    speech_indices = {1, 2, 3, 12, 13, 14}
    frames = [
        AudioFrame(i, pack("<h", 1500 if i in speech_indices else 0) * FRAME_SAMPLES, i * 0.03)
        for i in range(1, 44)
    ]
    mic = Mock()
    mic.read_since.return_value = frames
    vad = Mock()
    vad.is_speech.side_effect = lambda pcm, _rate: pcm[:2] != b"\x00\x00"
    monkeypatch.setattr(live, "Vad", lambda _mode: vad)
    monkeypatch.setattr(live, "perf_counter", lambda: 2.0)

    pcm, speech_end = await live.capture_reply(mic, 0, 1.5, started_at=0, end_silence_s=wait)

    assert speech_end == pytest.approx(0.42)
    assert vad.is_speech.call_count == expected_frames
    assert pcm == b"".join(frame.pcm for frame in frames[:19])


@pytest.mark.parametrize(
    ("transcript", "expected", "after_noise"),
    [
        (None, "NONE", True),
        ("Yes.", "OK", False),
        ("No.", "NOT_OK", False),
        ("No.", "NOT_OK", True),
        ("", "UNCLEAR", False),
    ],
)
async def test_onset_rejects_noise_but_retains_short_immediate_reply(
    monkeypatch, caplog, transcript, expected, after_noise
):
    started_at = perf_counter() - 10
    # Model an isolated hit, intermittent pairs, and WebRTC hangover below the peak floor.
    noise = [1200, 0, 1200, 1200, 100, 100, 100] + [0] * 32
    reply = [200, 1500, 1500, 1500] if transcript is not None else []
    amplitudes = (noise if after_noise else []) + reply + [0] * 29
    frames = [
        AudioFrame(i, pack("<h", amplitude) * FRAME_SAMPLES, started_at + i * FRAME_MS / 1000)
        for i, amplitude in enumerate(amplitudes, 1)
    ]
    mic = Mock()
    mic.cursor.return_value = 0
    mic.read_since.return_value = frames
    vad = Mock()
    vad.is_speech.side_effect = lambda pcm, _rate: pcm[:2] != b"\x00\x00"
    monkeypatch.setattr(live, "Vad", lambda _mode: vad)
    monkeypatch.setattr(live, "read_clip", lambda _key: b"audio")
    playback = PlaybackTiming(started_at - 1, started_at)
    monkeypatch.setattr(live, "play_interruptible", AsyncMock(return_value=playback))
    bus = Mock(publish=AsyncMock())
    stt = Mock()
    stt.transcribe.return_value = Transcription(transcript or "", 1)
    caplog.set_level("INFO", logger="backend.voice.capture")
    alert = VoiceSay(
        alert_id="onset-test", text="Are you okay?", tier="T2", lang="en", audio_key="demo_en"
    )

    with ThreadPoolExecutor(max_workers=1) as inference:
        await live.interact(bus, mic, stt, inference, alert)

    [message] = [
        data
        for topic, data in (call.args for call in bus.publish.await_args_list)
        if topic == "voice.reply"
    ]
    assert message["intent"] == expected
    assert message["transcript"] == (transcript or "")
    if transcript is None:
        stt.transcribe.assert_not_called()
        assert message["latency_ms"] == 0
        assert message["path"] == "keyword"
        assert "accepted_first_ms=none whisper_ms=skipped" in caplog.text
    else:
        # Include the quieter first frame as well as the three frames that confirmed onset.
        retained = frames[: len(amplitudes) - 29 + 5]
        stt.transcribe.assert_called_once_with(b"".join(frame.pcm for frame in retained), "en")
        assert message["path"] == ("keyword" if transcript else "llm")
