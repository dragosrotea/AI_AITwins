"""Cabin audio features derived from the shared microphone ring buffer."""

import asyncio
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from math import isclose, log10
from time import perf_counter

import numpy as np
from webrtcvad import Vad

from backend.bus.client import BusClient
from backend.schemas import AudioFeatures

from .devices import FRAME_MS, SAMPLE_RATE, AudioSettings, Microphone
from .profanity import CabinProfanity

PERIOD_S = 0.2
FRAME_S = FRAME_MS / 1000
FLOOR_DB = -96.0


def dbfs(power: float) -> float:
    return max(FLOOR_DB, 10 * log10(power)) if power else FLOOR_DB


class CabinMonitor:
    def __init__(
        self, mic: Microphone, settings: AudioSettings, profanity: CabinProfanity | None = None
    ) -> None:
        self._mic = mic
        self._settings = settings
        self._profanity = profanity
        self._cursor = mic.cursor()
        self._vad = Vad(2)
        self._muted = False
        self._first_clean_frame = float("-inf")
        self._last_frame_at: float | None = None
        self._levels: deque[tuple[float, float]] = deque()
        self._total_db = 0.0
        self._shout_reference: float | None = None
        self._shout_started_at: float | None = None

    def _reset_shout(self) -> None:
        self._shout_reference = None
        self._shout_started_at = None

    @contextmanager
    def suppress_playback(self) -> Iterator[None]:
        self._muted = True
        self._reset_shout()
        if self._profanity is not None:
            self._profanity.reset()
        try:
            yield
        finally:
            self._cursor = self._mic.cursor()
            # A frame ending right after playback can still contain speaker audio.
            self._first_clean_frame = perf_counter() + FRAME_S
            self._last_frame_at = None
            self._vad = Vad(2)
            self._muted = False

    def _expire(self, now: float) -> None:
        cutoff = now - self._settings.cabin_baseline_s
        while self._levels and self._levels[0][0] <= cutoff:
            self._total_db -= self._levels.popleft()[1]
        if not self._levels:
            self._total_db = 0.0

    def _baseline(self) -> float:
        return self._total_db / len(self._levels) if self._levels else FLOOR_DB

    def measure(self, now: float) -> AudioFeatures:
        swear_count = self._profanity.count(now) if self._profanity is not None else 0
        powers: list[float] = []
        speaking = False
        oldest = max(self._first_clean_frame, now - self._settings.cabin_baseline_s)
        if not self._muted:
            for frame in self._mic.read_since(self._cursor):
                if frame.sequence != self._cursor + 1 or (
                    self._last_frame_at is not None
                    and frame.ended_at - self._last_frame_at > 2 * FRAME_S
                ):
                    self._reset_shout()
                    self._vad = Vad(2)
                    if self._profanity is not None:
                        self._profanity.reset()
                self._cursor = frame.sequence
                if frame.ended_at < oldest:
                    continue
                self._last_frame_at = frame.ended_at
                self._expire(frame.ended_at)
                samples = np.frombuffer(frame.pcm, dtype="<i2").astype(np.float64)
                power = float(np.mean(samples * samples)) / (32768**2)
                level = dbfs(power)
                powers.append(power)
                frame_speaking = self._vad.is_speech(frame.pcm, SAMPLE_RATE)
                speaking |= frame_speaking
                if self._profanity is not None:
                    self._profanity.feed(frame, frame_speaking)

                reference = self._shout_reference
                if reference is None and self._levels:
                    reference = self._baseline()
                if reference is not None and level - reference >= self._settings.cabin_shout_db:
                    # Keep the onset baseline for this uninterrupted excursion.
                    self._shout_reference = reference
                    if self._shout_started_at is None:
                        self._shout_started_at = frame.ended_at - FRAME_S
                else:
                    self._reset_shout()
                self._levels.append((frame.ended_at, level))
                self._total_db += level

        shout_duration = 0.0
        if powers and self._shout_started_at is not None and self._last_frame_at is not None:
            shout_duration = self._last_frame_at - self._shout_started_at
        self._expire(now)
        return AudioFeatures(
            loudness_db=round(dbfs(sum(powers) / len(powers)), 1) if powers else FLOOR_DB,
            baseline_db=round(self._baseline(), 1),
            speaking=speaking,
            shouting=shout_duration >= self._settings.cabin_shout_s
            or isclose(shout_duration, self._settings.cabin_shout_s),
            swear_count_60s=swear_count,
            anger_voice_p=None,
        )

    async def publish(self, bus: BusClient) -> None:
        next_tick = perf_counter()
        while True:
            await bus.publish("audio.features", self.measure(perf_counter()).model_dump())
            now = perf_counter()
            next_tick += PERIOD_S
            if next_tick <= now:
                next_tick = now + PERIOD_S
            await asyncio.sleep(next_tick - now)
