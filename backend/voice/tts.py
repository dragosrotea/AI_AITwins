"""Speech synthesis and playback through DriveMate's configured output."""

import asyncio
from dataclasses import dataclass
from threading import Event
from time import perf_counter
from typing import Any

import miniaudio
import numpy as np
import sounddevice as sd

from backend.audio.devices import AudioSettings

VOICES = {"en": "en-US-MichelleNeural"}
RATES = {"en": "+0%"}
DEMO_LINES = {
    "en": "Hi, I'm DriveMate. How are you feeling behind the wheel?",
}
QUIET_PEAK = 32  # Approximately -60 dBFS for signed int16.
SPEECH_TAIL_MS = 40


@dataclass(frozen=True)
class PlaybackTiming:
    started_at: float
    ended_at: float
    output_device: int | None = None
    output_name: str | None = None


async def synthesize(text: str, lang: str) -> bytes:
    """Return the Edge MP3 in memory. Only synthesis uses the network."""
    import edge_tts

    audio = bytearray()
    async with asyncio.timeout(20):
        async for chunk in edge_tts.Communicate(text, VOICES[lang], rate=RATES[lang]).stream():
            if chunk["type"] == "audio":
                audio.extend(chunk["data"])
    if not audio:
        raise ValueError("Edge TTS returned no audio")
    return bytes(audio)


def play(
    audio: bytes,
    requested_at: float | None = None,
    *,
    stop: Event | None = None,
    chime: bytes | None = None,
) -> PlaybackTiming | None:
    """Decode MP3 or WAV and play at the selected device's default sample rate."""
    device = AudioSettings.load().output_device
    output = sd.query_devices(device, "output")
    rate = int(output["default_samplerate"])
    channels = min(2, output["max_output_channels"])
    pcm = miniaudio.decode(audio, nchannels=channels, sample_rate=rate).samples
    if not pcm:
        raise ValueError("Decoded audio is empty")
    if chime is not None:
        samples = np.asarray(pcm)
        active = np.flatnonzero((samples > QUIET_PEAK) | (samples < -QUIET_PEAK))
        del samples  # Release the array view before resizing PCM.
        if active.size:
            keep_frames = int(active[-1]) // channels + 1 + rate * SPEECH_TAIL_MS // 1000
            del pcm[keep_frames * channels :]
        pcm.extend(miniaudio.decode(chime, nchannels=channels, sample_rate=rate).samples)
    # PortAudio timestamps estimate when the first block reaches the DAC.
    offset = 0
    first_audio_at: float | None = None
    last_audio_at: float | None = None
    data = memoryview(pcm).cast("B")

    def fill(out: Any, frames: int, timing: Any, _status: object) -> None:
        nonlocal offset, first_audio_at, last_audio_at
        block_at = perf_counter() + (timing.outputBufferDacTime - timing.currentTime)
        if first_audio_at is None:
            first_audio_at = block_at
        count = min(frames * channels * 2, len(data) - offset)
        out[:count] = data[offset : offset + count]
        out[count:] = bytes(len(out) - count)
        offset += count
        if offset == len(data):
            last_audio_at = block_at + count / (2 * channels * rate)
            raise sd.CallbackStop

    finished = Event()
    if stop is not None and stop.is_set():
        return
    with sd.RawOutputStream(
        device=device,
        samplerate=rate,
        channels=channels,
        dtype="int16",
        callback=fill,
        finished_callback=finished.set,
    ) as stream:
        deadline = perf_counter() + len(pcm) / (rate * channels) + 5
        while not finished.wait(0.02):
            if stop is not None and stop.is_set():
                stream.abort()
                return
            if perf_counter() >= deadline:
                stream.abort()
                raise TimeoutError("Speaker playback did not complete")
    if first_audio_at is None or last_audio_at is None:
        raise RuntimeError("Speaker did not finish playing the audio")
    if requested_at is not None:
        elapsed_ms = (first_audio_at - requested_at) * 1000
        print(f"Request to playback start: {elapsed_ms:.0f} ms")
    return PlaybackTiming(first_audio_at, last_audio_at, stream.device, output.get("name"))


async def play_interruptible(audio: bytes, *, chime: bytes) -> PlaybackTiming:
    """On cancellation, abort and close the device before returning."""
    stop = Event()
    worker = asyncio.create_task(asyncio.to_thread(play, audio, stop=stop, chime=chime))
    try:
        timing = await asyncio.shield(worker)
        if timing is None:
            raise RuntimeError("Speaker playback stopped early")
        return timing
    except asyncio.CancelledError:
        stop.set()
        await asyncio.gather(worker, return_exceptions=True)
        raise
