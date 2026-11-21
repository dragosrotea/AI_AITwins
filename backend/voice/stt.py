"""In-memory speech-to-text for 16 kHz mono microphone PCM."""

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from time import perf_counter
from typing import Literal

import numpy as np
from faster_whisper import WhisperModel

Language = Literal["en"]


class SpeechExecutor(ThreadPoolExecutor):
    """Single Whisper worker. Background work may start only from an empty queue."""

    def __init__(self) -> None:
        super().__init__(max_workers=1)
        self._pending: set[Future] = set()
        self._background: Future | None = None

    def submit(self, fn, /, *args, **kwargs):
        # Submissions happen on the event loop. A running future stays pending
        # even if the coroutine awaiting it has been cancelled.
        self.cancel_queued_background()
        self._pending = {future for future in self._pending if not future.done()}
        future = super().submit(fn, *args, **kwargs)
        self._pending.add(future)
        return future

    def submit_if_idle(self, fn, /, *args):
        if any(not future.done() for future in self._pending):
            return None
        future = self.submit(fn, *args)
        self._background = future
        return future

    def cancel_queued_background(self) -> None:
        if self._background is not None:
            self._background.cancel()
            self._background = None


@dataclass(frozen=True)
class Transcription:
    text: str
    elapsed_ms: float


class SpeechToText:
    """Load one multilingual model and reuse it for each transcription."""

    def __init__(self, *, offline: bool = False) -> None:
        started = perf_counter()
        self._model = WhisperModel(
            "base", device="cpu", compute_type="int8", local_files_only=offline
        )
        self.load_ms = (perf_counter() - started) * 1000

    def transcribe(self, pcm: bytes, lang: Language) -> Transcription:
        """Transcribe signed little-endian int16 samples with an explicit language."""
        started = perf_counter()
        audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32)
        if not audio.size:
            raise ValueError("No microphone samples to transcribe")
        audio /= np.float32(32768)
        segments, _info = self._model.transcribe(
            audio, language=lang, task="transcribe", beam_size=1
        )
        text = "".join(segment.text for segment in segments).strip()
        return Transcription(text, (perf_counter() - started) * 1000)
