"""Interruptible alert playback, mic capture and reply handling."""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter

from webrtcvad import Vad

from backend.audio.devices import FRAME_MS, SAMPLE_RATE, AudioSettings, Microphone
from backend.bus.client import BusClient
from backend.llm import classify_reply
from backend.schemas import VoiceReply, VoiceSay, VoiceState, VoiceStatus

from .intents import classify_keyword
from .stt import SpeechToText
from .tts import play_interruptible, synthesize

ASSETS = Path(__file__).resolve().parents[2] / "assets" / "audio"
FRAME_S = FRAME_MS / 1000
END_SILENCE_S = 0.8
TRAILING_FRAMES = 150 // FRAME_MS
log = logging.getLogger(__name__)
capture_log = logging.getLogger("backend.voice.capture")


def read_clip(audio_key: str) -> bytes | None:
    name = audio_key if audio_key.endswith(".wav") else f"{audio_key}.wav"
    path = (ASSETS / name).resolve()
    if path.parent == ASSETS.resolve() and path.is_file():
        return path.read_bytes()
    return None


async def capture_reply(
    mic: Microphone, cursor: int, deadline: float, *, started_at: float
) -> tuple[bytes, float] | None:
    vad = Vad(2)
    frames: list[bytes] = []
    first_speech: float | None = None
    last_speech: float | None = None
    speech_frames = 0
    silent_frames = 0
    while True:
        for frame in mic.read_since(cursor):
            if frame.ended_at > deadline:
                break
            if frame.sequence != cursor + 1:
                raise RuntimeError("Microphone ring buffer overrun")
            cursor = frame.sequence
            frames.append(frame.pcm)
            if vad.is_speech(frame.pcm, SAMPLE_RATE):
                if first_speech is None:
                    first_speech = frame.ended_at
                last_speech = frame.ended_at
                speech_frames = len(frames)
                silent_frames = 0
            elif last_speech is not None:
                silent_frames += 1
                if silent_frames * FRAME_S >= END_SILENCE_S:
                    break
        remaining = deadline - perf_counter()
        if remaining <= 0 or silent_frames * FRAME_S >= END_SILENCE_S:
            break
        await asyncio.sleep(min(FRAME_S, remaining))
    if first_speech is None or last_speech is None:
        capture_log.info("VAD first_ms=none last_ms=none retained_ms=0 whisper_ms=skipped")
        return None
    retained = frames[: speech_frames + TRAILING_FRAMES]
    capture_log.info(
        "VAD first_ms=%.0f last_ms=%.0f retained_ms=%d",
        (first_speech - started_at) * 1000,
        (last_speech - started_at) * 1000,
        len(retained) * FRAME_MS,
    )
    return b"".join(retained), last_speech


async def status(bus: BusClient, alert_id: str, state: VoiceState) -> None:
    await bus.publish("voice.status", VoiceStatus(alert_id=alert_id, state=state).model_dump())


async def interact(
    bus: BusClient,
    mic: Microphone,
    stt: SpeechToText,
    inference: ThreadPoolExecutor,
    alert: VoiceSay,
) -> None:
    try:
        await status(bus, alert.alert_id, "speaking")
        clip = await asyncio.to_thread(read_clip, alert.audio_key) if alert.audio_key else None
        if clip is None:
            clip = await synthesize(alert.text, alert.lang)
        await play_interruptible(clip)
        chime = await asyncio.to_thread(read_clip, "chime")
        if chime is None:
            raise FileNotFoundError("assets/audio/chime.wav")
        await play_interruptible(chime)

        cursor = mic.cursor()
        started_at = perf_counter()
        deadline = started_at + alert.listen_s
        capture_log.info(
            "Capture start monotonic_s=%.3f cursor=%d window_ms=%.0f",
            started_at,
            cursor,
            alert.listen_s * 1000,
        )
        await status(bus, alert.alert_id, "listening")
        capture = await capture_reply(mic, cursor, deadline, started_at=started_at)
        if capture is None:
            reply = VoiceReply(
                alert_id=alert.alert_id, transcript="", intent="NONE", latency_ms=0, path="keyword"
            )
        else:
            pcm, speech_end = capture
            result = await asyncio.get_running_loop().run_in_executor(
                inference, stt.transcribe, pcm, alert.lang
            )
            capture_log.info("Whisper whisper_ms=%.0f", result.elapsed_ms)
            intent = classify_keyword(result.text, alert.lang)
            if intent is None:
                intent = await classify_reply(result.text, "confirmation") or "UNCLEAR"
                path = "llm"
            else:
                path = "keyword"
            reply = VoiceReply(
                alert_id=alert.alert_id,
                transcript=result.text,
                intent=intent,
                latency_ms=round((perf_counter() - speech_end) * 1000),
                path=path,
            )
        await bus.publish("voice.reply", reply.model_dump())
    except Exception as error:
        log.error("Voice interaction failed (%s)", type(error).__name__)
    finally:
        await status(bus, alert.alert_id, "idle")


async def run_live(bus: BusClient) -> None:
    settings = AudioSettings.load()
    stt = await asyncio.to_thread(SpeechToText, offline=True)
    ready = asyncio.Event()
    pending: VoiceSay | None = None
    active: asyncio.Task[None] | None = None

    def on_say(envelope: dict) -> None:
        nonlocal pending
        alert = VoiceSay.model_validate(envelope["data"])
        busy = pending is not None or (active is not None and not active.done())
        if busy and alert.tier != "T3":
            return
        pending = alert
        if active is not None and not active.done() and not active.cancelling():
            active.cancel()
        ready.set()

    # CTranslate2 inference already in progress cannot be cancelled. Keep one
    # inference worker, while allowing the replacement alert to speak promptly.
    with Microphone(settings) as mic, ThreadPoolExecutor(max_workers=1) as inference:
        bus.subscribe(["voice.say"], on_say)
        await bus.connect()
        while True:
            await ready.wait()
            ready.clear()
            alert, pending = pending, None
            if alert is None:
                continue
            active = asyncio.create_task(interact(bus, mic, stt, inference, alert))
            await asyncio.gather(active, return_exceptions=True)
