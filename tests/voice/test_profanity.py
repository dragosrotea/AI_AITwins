import asyncio
from concurrent.futures import Future, ThreadPoolExecutor
from statistics import mean
from threading import Event
from time import perf_counter
from unittest.mock import Mock

import pytest

from backend.audio import features, profanity
from backend.audio.devices import AudioSettings
from backend.audio.profanity import CHUNK_FRAMES, CabinProfanity, load_pattern
from backend.schemas import AudioFeatures
from backend.voice.intents import normalize
from backend.voice.stt import SpeechExecutor, Transcription
from tests.voice.test_audio_features import SimulatedMic


@pytest.mark.parametrize(
    ("text", "count"),
    [
        ("Fuck!", 1),
        ("This fucking thing", 1),
        ("Oh, shit.", 1),
        ("Bullshit.", 1),
        ("Dámn that noise.", 1),
        ("Goddamn, it's late.", 1),
        ("You bastard.", 1),
        ("Son of a—bitch!", 1),
        ("PISS OFF!", 1),
        ("That asshole cut in.", 1),
        ("Shit, shit; DAMN.", 3),
        ("Son of a bitch, what a bitch.", 2),
        ("Shiitake mushrooms.", 0),
        ("A dam on the river.", 0),
        ("The light is green.", 0),
        ("First-class ticket.", 0),
        ("Good morning.", 0),
        ("This is a microphone.", 0),
        ("Someone is passing us.", 0),
        ("Pitch the tent.", 0),
        ("The guitarist tuned up.", 0),
        ("A Scunthorpe road sign.", 0),
        ("The word unbitching.", 0),
        ("Bullshittery.", 0),
    ],
)
def test_english_words_and_phrases(text, count):
    assert len(load_pattern().findall(normalize(text))) == count


class ImmediateWorker:
    def submit_if_idle(self, fn, arg):
        future = Future()
        future.set_result(fn(arg))
        return future

    def cancel_queued_background(self):
        pass


def monitor_with(mic, stt, worker, monkeypatch):
    monkeypatch.setattr(features, "Vad", lambda _mode: Mock(is_speech=lambda pcm, _rate: any(pcm)))
    tracker = CabinProfanity(stt, worker)
    return features.CabinMonitor(mic, AudioSettings(None, None, 10), tracker), tracker


def test_cabin_chunks_skip_silence_and_reset_on_playback_reply_and_gap(monkeypatch):
    mic = SimulatedMic()

    def now():
        return mic.now

    monkeypatch.setattr(features, "perf_counter", now)
    monkeypatch.setattr(profanity, "perf_counter", now)
    stt = Mock(transcribe=Mock(return_value=Transcription("Damn it.", 1)))
    monitor, tracker = monitor_with(mic, stt, ImmediateWorker(), monkeypatch)

    def feed(amplitude, count):
        mic.feed(amplitude, count)
        monitor.measure(mic.now)

    feed(0, CHUNK_FRAMES)
    feed(1000, 60)
    tracker.pause()  # An alert arrives before the first cabin chunk completes.
    with monitor.suppress_playback():
        feed(2000, CHUNK_FRAMES)
    feed(3000, CHUNK_FRAMES)  # Reply capture must remain excluded after unmuting.
    mic.feed(3000, 10)  # Buffered reply frames are read only after the alert finishes.
    tracker.resume()
    feed(1000, 61)
    stt.transcribe.assert_not_called()

    mic.now += 0.2  # Hardware timestamp gap.
    feed(1000, CHUNK_FRAMES - 1)
    stt.transcribe.assert_not_called()
    feed(1000, 1)
    pcm, lang = stt.transcribe.call_args.args
    assert len(pcm) == 3 * features.SAMPLE_RATE * 2 and lang == "en"
    assert monitor.measure(mic.now).swear_count_60s == 1
    assert monitor.measure(mic.now + 60).swear_count_60s == 0


def test_pending_result_uses_capture_time_even_if_inference_finishes_late(monkeypatch):
    mic = SimulatedMic()
    future = Future()
    worker = Mock(submit_if_idle=Mock(return_value=future))
    monitor, _tracker = monitor_with(mic, Mock(), worker, monkeypatch)
    mic.feed(1000, CHUNK_FRAMES)
    assert monitor.measure(mic.now).swear_count_60s == 0
    future.set_result(3)
    assert monitor.measure(mic.now + 59.9).swear_count_60s == 3
    assert monitor.measure(mic.now + 60).swear_count_60s == 0


def test_queued_cabin_job_yields_to_reply(monkeypatch):
    cabin = Future()
    reply = Future()
    monkeypatch.setattr(ThreadPoolExecutor, "submit", Mock(side_effect=[cabin, reply]))
    with SpeechExecutor() as worker:
        assert worker.submit_if_idle(Mock()) is cabin
        assert worker.submit(Mock()) is reply
        assert cabin.cancelled()
        assert worker.submit_if_idle(Mock()) is None


@pytest.mark.parametrize("fails", [False, True])
async def test_slow_or_failed_cabin_stt_keeps_5hz_and_never_queues_ahead_of_reply(
    monkeypatch, fails
):
    entered = Event()
    release = Event()
    times = []
    counts = []
    published = asyncio.Event()
    mic = SimulatedMic()

    def transcribe(_pcm, _lang):
        entered.set()
        assert release.wait(5)
        if fails:
            raise RuntimeError("sensitive transcript should never be logged")
        return Transcription("Damn.", 1)

    async def publish(topic, data):
        assert topic == "audio.features"
        AudioFeatures.model_validate(data)
        times.append(perf_counter())
        counts.append(data["swear_count_60s"])
        if len(times) == 6:
            published.set()

    with SpeechExecutor() as worker:
        monitor, tracker = monitor_with(mic, Mock(transcribe=transcribe), worker, monkeypatch)
        mic.now = perf_counter()
        mic.feed(1000, CHUNK_FRAMES)
        monitor.measure(mic.now)
        assert await asyncio.to_thread(entered.wait, 2)
        tracker.pause()
        reply = worker.submit(lambda: "reply")
        assert not reply.done()  # Native inference already running is not interruptible.
        assert worker.submit_if_idle(Mock()) is None
        publishing = asyncio.create_task(monitor.publish(Mock(publish=publish)))
        try:
            await asyncio.wait_for(published.wait(), 3)
            assert 0.18 <= mean(b - a for a, b in zip(times, times[1:], strict=False)) <= 0.22
            assert counts == [0] * 6
            release.set()
            assert await asyncio.wrap_future(reply) == "reply"
            assert tracker.count(mic.now) == (0 if fails else 1)
        finally:
            release.set()
            publishing.cancel()
            await asyncio.gather(publishing, return_exceptions=True)
