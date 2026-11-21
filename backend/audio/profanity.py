"""Count profanity in short local cabin recordings without retaining transcripts."""

import re
from collections import deque
from concurrent.futures import Future
from pathlib import Path
from time import perf_counter

from backend.voice.intents import normalize
from backend.voice.stt import SpeechExecutor, SpeechToText

from .devices import FRAME_MS, AudioFrame

LEXICON = Path(__file__).resolve().parents[2] / "config" / "lexicons" / "profanity_en.txt"
CHUNK_FRAMES = 3000 // FRAME_MS
WINDOW_S = 60.0


def load_pattern(path: Path = LEXICON) -> re.Pattern[str]:
    words = {
        normalize(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    } - {""}
    if not words:
        raise ValueError("Empty English profanity lexicon")
    # Prefer the complete phrase when it contains a shorter lexicon entry.
    alternatives = "|".join(re.escape(word) for word in sorted(words, key=lambda w: (-len(w), w)))
    return re.compile(rf"\b(?:{alternatives})\b")


class CabinProfanity:
    def __init__(self, stt: SpeechToText, worker: SpeechExecutor) -> None:
        self._stt = stt
        self._worker = worker
        self._pattern = load_pattern()
        self._frames: list[bytes] = []
        self._speaking = False
        self._after = float("-inf")
        self._pending: tuple[float, Future[int | None]] | None = None
        self._counts: deque[tuple[float, int]] = deque()

    def reset(self) -> None:
        self._frames.clear()
        self._speaking = False

    def pause(self) -> None:
        self._after = float("inf")
        self._worker.cancel_queued_background()
        self.reset()

    def resume(self) -> None:
        # Discard buffered reply frames and the frame crossing the window boundary.
        self._after = perf_counter() + FRAME_MS / 1000
        self.reset()

    def feed(self, frame: AudioFrame, speaking: bool) -> None:
        if frame.ended_at < self._after:
            return
        self._frames.append(frame.pcm)
        self._speaking |= speaking
        if len(self._frames) == CHUNK_FRAMES:
            if self._speaking and self._pending is None:
                future = self._worker.submit_if_idle(self._transcribe_count, b"".join(self._frames))
                if future is not None:
                    self._pending = (frame.ended_at, future)
            self.reset()

    def _transcribe_count(self, pcm: bytes) -> int | None:
        try:
            transcript = self._stt.transcribe(pcm, "en").text
            return sum(1 for _ in self._pattern.finditer(normalize(transcript)))
        except Exception:  # noqa: BLE001 - never retain or log a cabin STT traceback
            return None

    def count(self, now: float) -> int:
        cutoff = now - WINDOW_S
        if self._pending is not None:
            captured_at, future = self._pending
            if future.done():
                self._pending = None
                count = None if future.cancelled() else future.result()
                if count and captured_at > cutoff:
                    self._counts.append((captured_at, count))
        while self._counts and self._counts[0][0] <= cutoff:
            self._counts.popleft()
        return sum(count for _, count in self._counts)
