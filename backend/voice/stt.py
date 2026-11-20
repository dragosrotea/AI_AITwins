"""In-memory speech-to-text for 16 kHz mono microphone PCM."""

from dataclasses import dataclass
from time import perf_counter
from typing import Literal

import numpy as np
from faster_whisper import WhisperModel

Language = Literal["ro", "en"]


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
