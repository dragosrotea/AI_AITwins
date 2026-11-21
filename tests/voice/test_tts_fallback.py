import asyncio
from io import BytesIO
from threading import Event, get_ident
from time import perf_counter
from unittest.mock import AsyncMock, Mock
from wave import open as open_wave

import pytest

from backend.llm.templates import english_fallback
from backend.schemas import VoiceSay
from backend.voice import live, tts

UNCACHED_LINE = "This is the offline speech check. Are you comfortable behind the wheel?"


def wav(frame_pair: bytes = b"\x00\x02\x00\xfe") -> bytes:
    target = BytesIO()
    with open_wave(target, "wb") as audio:
        audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        audio.writeframes(frame_pair * 80)
    return target.getvalue()


@pytest.mark.parametrize(
    "edge_outcome", ["success", "error", "timeout", "empty", "corrupt", "silent"]
)
async def test_live_tts_prefers_usable_edge_and_runs_system_speech_off_loop(
    monkeypatch, edge_outcome
):
    audio = wav()
    main_thread = get_ident()
    timed_out = asyncio.Event()

    async def edge(_text, _lang):
        if edge_outcome == "error":
            raise OSError("offline")
        if edge_outcome == "timeout":
            try:
                await asyncio.Future()
            finally:
                timed_out.set()
        return {"success": audio, "empty": b"", "corrupt": b"garbage", "silent": wav(bytes(4))}[
            edge_outcome
        ]

    def system(text, cancelled):
        assert text == UNCACHED_LINE and not cancelled.is_set()
        assert get_ident() != main_thread
        return audio

    fallback = Mock(side_effect=system)
    monkeypatch.setattr(tts, "LIVE_EDGE_TIMEOUT_S", 0.05 if edge_outcome == "timeout" else 2)
    monkeypatch.setattr(tts, "synthesize_edge", edge)
    monkeypatch.setattr(tts, "_synthesize_system", fallback)
    assert await asyncio.wait_for(tts.synthesize(UNCACHED_LINE, "en"), 2) == audio
    assert fallback.call_count == (edge_outcome != "success")
    assert timed_out.is_set() == (edge_outcome == "timeout")


@pytest.mark.parametrize("stage", ["edge", "system"])
async def test_cancelled_synthesis_cannot_play_before_or_after_cached_replacement(
    monkeypatch, stage
):
    loop = asyncio.get_running_loop()
    started = asyncio.Event()
    rendered = asyncio.Event()
    release = Event()

    async def edge(_text, _lang):
        if stage == "edge":
            started.set()
            await asyncio.Future()
        raise OSError("offline")

    def system(_text, cancelled):
        loop.call_soon_threadsafe(started.set)
        assert release.wait(5)
        audio = wav()
        loop.call_soon_threadsafe(rendered.set)
        return audio

    offline = Mock(side_effect=system)
    monkeypatch.setattr(tts, "synthesize_edge", edge)
    monkeypatch.setattr(tts, "_synthesize_system", offline)
    now = perf_counter()
    playback = AsyncMock(return_value=tts.PlaybackTiming(now, now))
    monkeypatch.setattr(live, "play_interruptible", playback)
    monkeypatch.setattr(live, "capture_reply", AsyncMock(return_value=None))
    bus = Mock(publish=AsyncMock())
    mic = Mock(cursor=Mock(return_value=1))
    alert = VoiceSay(alert_id="uncached", tier="T2", lang="en", text=UNCACHED_LINE)
    task = asyncio.create_task(live.interact(bus, mic, Mock(), None, alert))
    try:
        await asyncio.wait_for(started.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 1)
        playback.assert_not_awaited()

        text, key = english_fallback("microsleep")
        replacement = VoiceSay(alert_id="cached", tier="T3", lang="en", text=text, audio_key=key)
        await live.interact(bus, mic, Mock(), None, replacement)
        playback.assert_awaited_once_with(live.read_clip(key), chime=live.read_clip("chime"))
        assert offline.call_count == (stage == "system")
        if stage == "system":
            assert offline.call_args.args[1].is_set()
            release.set()
            await asyncio.wait_for(rendered.wait(), 1)
            assert playback.await_count == 1
        replies = [
            data
            for topic, data in (call.args for call in bus.publish.await_args_list)
            if topic == "voice.reply"
        ]
        assert [reply["alert_id"] for reply in replies] == ["cached"]
    finally:
        release.set()


@pytest.mark.parametrize("error", [ModuleNotFoundError(), RuntimeError("No English voice")])
async def test_live_interaction_returns_idle_when_system_speech_is_unavailable(monkeypatch, error):
    monkeypatch.setattr(tts, "synthesize_edge", AsyncMock(side_effect=OSError("offline")))
    monkeypatch.setattr(tts, "_synthesize_system", Mock(side_effect=error))
    playback = AsyncMock()
    monkeypatch.setattr(live, "play_interruptible", playback)
    bus = Mock(publish=AsyncMock())
    alert = VoiceSay(alert_id="uncached", tier="T2", lang="en", text=UNCACHED_LINE)
    await live.interact(bus, Mock(), Mock(), None, alert)
    playback.assert_not_awaited()
    assert [call.args[1]["state"] for call in bus.publish.await_args_list] == ["speaking", "idle"]
