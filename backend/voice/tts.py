"""Speech synthesis and playback through DriveMate's configured output."""

import asyncio
import logging
import subprocess
import sys
from dataclasses import dataclass
from locale import windows_locale
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Lock
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
LIVE_EDGE_TIMEOUT_S = 2.0
_system_voice_lock = Lock()
log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PlaybackTiming:
    started_at: float
    ended_at: float
    output_device: int | None = None
    output_name: str | None = None


async def synthesize_edge(text: str, lang: str) -> bytes:
    """Return the Edge MP3 in memory, allowing longer requests for asset rendering."""
    import edge_tts

    audio = bytearray()
    async with asyncio.timeout(20):
        async for chunk in edge_tts.Communicate(text, VOICES[lang], rate=RATES[lang]).stream():
            if chunk["type"] == "audio":
                audio.extend(chunk["data"])
    if not audio:
        raise ValueError("Edge TTS returned no audio")
    return bytes(audio)


def _require_audio(audio: bytes) -> bytes:
    decoded = miniaudio.decode(audio)
    if not decoded.num_frames or not any(abs(sample) > QUIET_PEAK for sample in decoded.samples):
        raise ValueError("Speech synthesis returned no audible samples")
    return audio


def _is_english_voice(languages: list[str | bytes]) -> bool:
    for language in languages:
        if isinstance(language, bytes):
            language = language.decode("ascii", errors="ignore")
        language = language.lstrip("\x02\x05").replace("_", "-").lower()
        if language == "en" or language.startswith("en-"):
            return True
        if sys.platform == "win32":
            # Older SAPI drivers expose hexadecimal locale IDs rather than language tags.
            try:
                if windows_locale.get(int(language.split(";")[0], 16), "").startswith("en_"):
                    return True
            except ValueError:
                continue
    return False


def _render_macos_say(text: str) -> bytes:
    """macOS: `say` writes a real 16-bit WAV (pyttsx3 writes AIFF that miniaudio cannot read)."""
    with TemporaryDirectory(prefix="drivemate-tts-") as directory:
        path = Path(directory) / "speech.wav"
        subprocess.run(
            ["say", "-o", str(path), "--data-format=LEI16@22050", text],
            check=True,
            capture_output=True,
            timeout=15,
        )
        return _require_audio(path.read_bytes())


def _render_system_voice(text: str) -> bytes:
    if sys.platform == "darwin":
        return _render_macos_say(text)
    from pyttsx3.engine import Engine

    # A fresh engine keeps Windows COM objects on the worker that created them.
    engine = Engine()
    voice = next((v for v in engine.getProperty("voices") if _is_english_voice(v.languages)), None)
    if voice is None:
        raise RuntimeError("No English system voice installed")
    engine.setProperty("voice", voice.id)
    with TemporaryDirectory(prefix="drivemate-tts-") as directory:
        path = Path(directory) / "speech.wav"
        engine.save_to_file(text, str(path))
        engine.runAndWait()
        return _require_audio(path.read_bytes())


def _synthesize_system(text: str, cancelled: Event) -> bytes:
    # Serialize access to the system speech driver across cancelled and replacement requests.
    with _system_voice_lock:
        if cancelled.is_set():
            return b""
        if sys.platform != "win32":
            return _render_system_voice(text)

        import pythoncom

        pythoncom.CoInitialize()
        try:
            return _render_system_voice(text)
        finally:
            pythoncom.CoUninitialize()


async def synthesize(text: str, lang: str) -> bytes:
    """Render live English speech with Edge first and an installed system voice second."""
    try:
        async with asyncio.timeout(LIVE_EDGE_TIMEOUT_S):
            audio = await synthesize_edge(text, lang)
            return await asyncio.to_thread(_require_audio, audio)
    except Exception as error:  # noqa: BLE001 - Edge failures have one offline fallback
        log.info("Edge TTS unavailable (%s), trying the English system voice", type(error).__name__)

    if lang != "en":
        raise ValueError("System speech is English only")
    cancelled = Event()
    try:
        audio = await asyncio.to_thread(_synthesize_system, text, cancelled)
    except asyncio.CancelledError:
        # The driver can finish rendering, but no result can reach playback after cancellation.
        cancelled.set()
        raise
    except Exception as error:  # noqa: BLE001 - missing or broken system speech must stay local
        raise RuntimeError("English system speech unavailable") from error
    log.info("Using English system speech for playback")
    return audio


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
