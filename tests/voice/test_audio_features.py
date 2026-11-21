import asyncio
from statistics import mean
from struct import pack
from time import perf_counter
from unittest.mock import Mock

import pytest

from backend.audio import features
from backend.audio.devices import FRAME_SAMPLES, AudioFrame, AudioSettings
from backend.schemas import AudioFeatures


class SimulatedMic:
    def __init__(self):
        self.frames = []
        self.now = 100.0

    def cursor(self):
        return len(self.frames)

    def read_since(self, sequence):
        return self.frames[sequence:]

    def feed(self, amplitude, count, spacing=features.FRAME_S):
        pcm = pack("<h", amplitude) * FRAME_SAMPLES
        for _ in range(count):
            self.now += spacing
            self.frames.append(AudioFrame(self.cursor() + 1, pcm, self.now))


def test_shouting_requires_level_and_one_continuous_second():
    mic = SimulatedMic()
    monitor = features.CabinMonitor(mic, AudioSettings(None, None, 10))
    mic.feed(1000, 100)
    assert monitor.measure(mic.now).baseline_db == -30.3

    mic.feed(5500, 40)  # 14.8 dB above the original baseline for 1.2 s.
    assert not monitor.measure(mic.now).shouting

    mic.now += 60
    mic.feed(1000, 100)
    assert monitor.measure(mic.now).baseline_db == -30.3
    mic.feed(5630, 33)  # Just over +15 dB for 990 ms.
    assert not monitor.measure(mic.now).shouting
    mic.feed(5630, 1)
    assert monitor.measure(mic.now).shouting
    mic.feed(1000, 1)
    assert not monitor.measure(mic.now).shouting


@pytest.mark.parametrize(("duration", "after_510_ms"), [(0.5, True), (1.0, False)])
def test_configured_shout_duration(duration, after_510_ms):
    mic = SimulatedMic()
    monitor = features.CabinMonitor(mic, AudioSettings(None, None, 10, cabin_shout_s=duration))
    mic.feed(1000, 100)
    monitor.measure(mic.now)
    mic.feed(5630, 10)
    assert not monitor.measure(mic.now).shouting  # 300 ms.
    mic.feed(5630, 6)
    assert not monitor.measure(mic.now).shouting  # 480 ms.
    assert not monitor.measure(mic.now).shouting  # A poll without a new frame.
    mic.feed(5630, 1)
    assert monitor.measure(mic.now).shouting is after_510_ms
    mic.feed(5630, 6)
    assert monitor.measure(mic.now).shouting is after_510_ms  # 690 ms.


def test_shouting_uses_frame_timestamps_instead_of_frame_count():
    mic = SimulatedMic()
    monitor = features.CabinMonitor(mic, AudioSettings(None, None, 10, cabin_shout_s=0.5))
    mic.feed(1000, 100)
    monitor.measure(mic.now)
    mic.feed(5630, 17, spacing=0.025)
    assert not monitor.measure(mic.now).shouting  # 430 ms by timestamps.
    mic.feed(5630, 3, spacing=0.025)
    assert monitor.measure(mic.now).shouting  # 505 ms by timestamps.


@pytest.mark.parametrize("duration", [0.5, 1.0])
def test_playback_is_muted_and_excluded_from_baseline(monkeypatch, duration):
    mic = SimulatedMic()
    monkeypatch.setattr(features, "perf_counter", lambda: mic.now)
    vad = Mock()
    vad.is_speech.return_value = True
    monkeypatch.setattr(features, "Vad", lambda _mode: vad)
    monitor = features.CabinMonitor(mic, AudioSettings(None, None, 10, cabin_shout_s=duration))
    mic.feed(1000, 10)
    before = monitor.measure(mic.now)
    assert before.speaking
    reply_cursor = mic.cursor()

    with monitor.suppress_playback():
        mic.feed(25_000, 39)
        muted = monitor.measure(mic.now)
        assert muted.baseline_db == before.baseline_db
        assert muted.loudness_db == features.FLOOR_DB
        assert not muted.speaking and not muted.shouting
        mic.feed(25_000, 2)  # Playback between the last tick and unmute.
    mic.now -= 0.02
    mic.feed(25_000, 1)  # Frame overlapping the unmute boundary.
    mic.feed(1000, 6)
    resumed = monitor.measure(mic.now)
    assert resumed.baseline_db == before.baseline_db
    assert resumed.speaking and not resumed.shouting
    assert vad.is_speech.call_count == 16
    assert len(mic.read_since(reply_cursor)) == 48


async def test_audio_features_publish_at_five_hz():
    mic = SimulatedMic()
    monitor = features.CabinMonitor(mic, AudioSettings(None, None, 10))
    times = []
    received = asyncio.Event()

    async def publish(topic, data):
        assert topic == "audio.features"
        AudioFeatures.model_validate(data)
        times.append(perf_counter())
        if len(times) == 6:
            received.set()

    worker = asyncio.create_task(monitor.publish(Mock(publish=publish)))
    try:
        await asyncio.wait_for(received.wait(), 3)
        intervals = [later - earlier for earlier, later in zip(times, times[1:], strict=False)]
        assert 0.18 <= mean(intervals) <= 0.22
    finally:
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)
