"""DriveMate audio device selection and shared microphone capture."""

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Any, Self

import sounddevice as sd
import yaml

SAMPLE_RATE = 16_000
FRAME_MS = 30
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000
CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "thresholds.yaml"


@dataclass(frozen=True)
class AudioSettings:
    input_device: int | None
    output_device: int | None
    ring_buffer_s: int

    @classmethod
    def load(cls, path: Path = CONFIG_PATH) -> "AudioSettings":
        with path.open(encoding="utf-8") as stream:
            voice = yaml.safe_load(stream)["voice"]
        for key in ("input_device", "output_device"):
            value = voice[key]
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"voice.{key} must be null or a nonnegative device index")
        duration = voice["ring_buffer_s"]
        if type(duration) is not int or duration <= 0:
            raise ValueError("voice.ring_buffer_s must be a positive integer")
        return cls(voice["input_device"], voice["output_device"], duration)


@dataclass(frozen=True)
class AudioFrame:
    sequence: int
    pcm: bytes
    ended_at: float = field(default_factory=perf_counter)


class Microphone:
    """Own one PortAudio stream. Readers keep independent sequence cursors."""

    def __init__(self, settings: AudioSettings) -> None:
        self._device = settings.input_device
        self._frames: deque[AudioFrame] = deque(maxlen=settings.ring_buffer_s * 1000 // FRAME_MS)
        self._lock = Lock()
        self._sequence = 0
        self._stream: sd.RawInputStream | None = None

    def _capture(self, data: Any, frames: int, _time: object, _status: object) -> None:
        if frames != FRAME_SAMPLES:
            return
        ended_at = perf_counter()
        pcm = bytes(data)
        with self._lock:
            self._sequence += 1
            self._frames.append(AudioFrame(self._sequence, pcm, ended_at))

    def cursor(self) -> int:
        """Mark the latest frame, for example when a reply window opens."""
        with self._lock:
            return self._sequence

    def read_since(self, sequence: int) -> tuple[AudioFrame, ...]:
        """Read retained frames after sequence. A gap means older frames expired."""
        with self._lock:
            return tuple(frame for frame in self._frames if frame.sequence > sequence)

    def __enter__(self) -> Self:
        if self._stream is not None:
            raise RuntimeError("Microphone capture is already running")
        stream = sd.RawInputStream(
            device=self._device,
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=FRAME_SAMPLES,
            callback=self._capture,
        )
        try:
            stream.start()
        except BaseException:
            stream.close()
            raise
        self._stream = stream
        return self

    def __exit__(self, *_exc: object) -> None:
        try:
            if self._stream is not None:
                self._stream.close()
        finally:
            self._stream = None
            with self._lock:
                self._frames.clear()
