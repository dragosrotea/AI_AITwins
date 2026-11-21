import asyncio
from array import array
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from backend.audio import devices
from backend.voice import live, tts


def test_alert_and_chime_share_output_stream_and_trim_only_quiet_tail(monkeypatch):
    alert = array("h", [0, 1000, -1000] + [0] * 100)
    chime = array("h", [2000] * 4)
    decode = Mock(side_effect=[SimpleNamespace(samples=alert), SimpleNamespace(samples=chime)])
    monkeypatch.setattr(tts.miniaudio, "decode", decode)
    monkeypatch.setattr(
        tts.sd,
        "query_devices",
        lambda *_args: {"default_samplerate": 1000, "max_output_channels": 1, "name": "Headphones"},
    )
    monkeypatch.setattr(tts, "perf_counter", lambda: 10.0)
    output = array("h")

    def open_stream(**kwargs):
        stream = Mock(device=3)

        def start():
            buffer = bytearray(128)
            with pytest.raises(tts.sd.CallbackStop):
                kwargs["callback"](
                    buffer, 64, SimpleNamespace(outputBufferDacTime=2.1, currentTime=2), None
                )
            output.frombytes(buffer)
            kwargs["finished_callback"]()
            return stream

        stream.__enter__ = Mock(side_effect=start)
        stream.__exit__ = Mock()
        return stream

    stream = Mock(side_effect=open_stream)
    monkeypatch.setattr(tts.sd, "RawOutputStream", stream)
    result = tts.play(b"speech", chime=b"chime")

    stream.assert_called_once()
    assert list(output[:47]) == [0, 1000, -1000] + [0] * 40 + [2000] * 4
    assert not any(output[47:])
    assert result.started_at == pytest.approx(10.1)
    assert result.ended_at == pytest.approx(10.147)
    assert (result.output_device, result.output_name) == (3, "Headphones")


async def test_capture_preserves_immediate_reply_after_chime(monkeypatch):
    mic = devices.Microphone(devices.AudioSettings(None, None, 10))
    samples = devices.FRAME_SAMPLES
    reply_sample = b"\x00\x04"
    monkeypatch.setattr(devices, "perf_counter", lambda: 20.0)
    for ended_at, pcm in (
        (9.985, b"\x01\x00" * samples),
        (10.015, b"\x01\x00" * (samples // 2) + reply_sample * (samples // 2)),
        (10.045, reply_sample * samples),
        (10.075, reply_sample * samples),
    ):
        clock = SimpleNamespace(inputBufferAdcTime=ended_at - 0.03, currentTime=20.0)
        mic._capture(pcm, samples, clock, None)
    assert mic.cursor(after=10) == 1
    vad = Mock(is_speech=Mock(return_value=True))
    monkeypatch.setattr(live, "Vad", lambda _mode: vad)
    monkeypatch.setattr(live, "perf_counter", lambda: 11)

    pcm, speech_end = await live.capture_reply(mic, mic.cursor(after=10), 10.1, started_at=10)
    assert pcm == bytes(samples) + reply_sample * (2 * samples + samples // 2)
    assert speech_end == pytest.approx(10.075)
    assert vad.is_speech.call_count == 3


async def test_interrupted_playback_waits_for_combined_stream_cleanup(monkeypatch):
    opened = Event()
    cleaned = Event()

    def play(audio, *, chime, stop):
        assert (audio, chime) == (b"alert", b"chime")
        opened.set()
        assert stop.wait(2)
        cleaned.set()

    monkeypatch.setattr(tts, "play", play)
    task = asyncio.create_task(tts.play_interruptible(b"alert", chime=b"chime"))
    assert await asyncio.to_thread(opened.wait, 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cleaned.is_set()
