"""Interruptible alert playback, mic capture and reply handling."""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from functools import cache
from math import log10
from pathlib import Path
from time import perf_counter
from typing import TextIO

import numpy as np
from webrtcvad import Vad

from backend.audio.devices import FRAME_MS, FRAME_SAMPLES, SAMPLE_RATE, AudioSettings, Microphone
from backend.audio.features import CabinMonitor
from backend.audio.profanity import CabinProfanity
from backend.bus.client import BusClient
from backend.llm.engagement import engagement_clips
from backend.llm.language import is_english
from backend.llm.templates import english_fallback, english_templates
from backend.schemas import VoiceReply, VoiceSay, VoiceState, VoiceStatus

from .intents import ReplyClassification, classify_confirmation, classify_keyword
from .latency import LatencySample, elapsed_ms
from .stt import SpeechExecutor, SpeechToText
from .tts import DEMO_LINES, play_interruptible, synthesize

ASSETS = Path(__file__).resolve().parents[2] / "assets" / "audio"
FRAME_S = FRAME_MS / 1000
END_SILENCE_S = 0.8
TRAILING_FRAMES = 150 // FRAME_MS
ONSET_SIGNAL_PEAK = 328  # About -40 dBFS peak, matching the capture diagnostics.
ONSET_FRAMES = 3
log = logging.getLogger(__name__)
capture_log = logging.getLogger("backend.voice.capture")


@cache
def english_audio() -> dict[str, str]:
    return {
        "demo_en": DEMO_LINES["en"],
        **{key: text for variants in english_templates().values() for text, key in variants},
        **engagement_clips(),
    }


def english_alert(alert: VoiceSay) -> VoiceSay:
    key = (alert.audio_key or "").removesuffix(".wav")
    if key == "demo_ro":
        key = "demo_en"
    elif "_ro_" in key:
        key = key.replace("_ro_", "_en_")
    if key in english_audio():
        text = english_audio()[key]
    elif alert.lang == "en" and not key and is_english(alert.text):
        return alert
    else:
        text, key = (
            english_fallback("crash_candidate", 3)
            if alert.tier == "T3"
            else english_fallback("checkin_due")
        )
    return alert.model_copy(update={"lang": "en", "text": text, "audio_key": key})


def read_clip(audio_key: str) -> bytes | None:
    key = audio_key.removesuffix(".wav")
    if key not in {"chime", "alarm"} and key not in english_audio():
        return None
    path = (ASSETS / f"{key}.wav").resolve()
    if path.parent == ASSETS.resolve() and path.name == f"{key}.wav" and path.is_file():
        return path.read_bytes()
    return None


async def capture_reply(
    mic: Microphone,
    cursor: int,
    deadline: float,
    *,
    started_at: float,
    alert_id: str | None = None,
    end_silence_s: float = END_SILENCE_S,
) -> tuple[bytes, float] | None:
    vad = Vad(2)
    frames: list[bytes] = []
    first_positive: float | None = None
    last_positive: float | None = None
    onset_candidate: float | None = None
    accepted_onset: float | None = None
    last_speech: float | None = None
    frame_ends: list[float] | None = [] if capture_log.isEnabledFor(logging.INFO) else None
    first_signal: float | None = None
    last_signal: float | None = None
    capture_peak = 0
    onset_min_peak = 0
    onset_max_peak = 0
    masked_samples = 0
    speech_frames = 0
    positive_frames = 0
    onset_frames = 0
    endpoint_reached = False
    while True:
        for frame in mic.read_since(cursor):
            if frame.ended_at > deadline:
                break
            if frame.sequence != cursor + 1:
                raise RuntimeError("Microphone ring buffer overrun")
            cursor = frame.sequence
            if frame.ended_at <= started_at:
                continue
            pcm = frame.pcm
            before_start = FRAME_SAMPLES - round((frame.ended_at - started_at) * SAMPLE_RATE)
            if before_start > 0:
                # Keep any immediate reply in a frame shared with the end of the chime.
                skip_bytes = before_start * 2
                pcm = bytes(skip_bytes) + pcm[skip_bytes:]
                masked_samples += before_start
            frames.append(pcm)
            positive = vad.is_speech(pcm, SAMPLE_RATE)
            peak = 0
            if frame_ends is not None:
                frame_ends.append(frame.ended_at)
            if frame_ends is not None or (positive and accepted_onset is None):
                samples = np.frombuffer(pcm, dtype="<i2")
                peak = max(int(samples.max()), -int(samples.min()))
            has_signal = peak >= ONSET_SIGNAL_PEAK
            if frame_ends is not None:
                capture_peak = max(capture_peak, peak)
                if has_signal:
                    if first_signal is None:
                        first_signal = frame.ended_at
                    last_signal = frame.ended_at
            if positive:
                positive_frames += 1
                if first_positive is None:
                    first_positive = frame.ended_at
                last_positive = frame.ended_at
            if accepted_onset is None:
                if not (positive and has_signal):
                    onset_frames = 0
                    continue
                if onset_frames == 0:
                    onset_candidate = frame.ended_at
                    onset_min_peak = onset_max_peak = peak
                else:
                    onset_min_peak = min(onset_min_peak, peak)
                    onset_max_peak = max(onset_max_peak, peak)
                onset_frames += 1
                if onset_frames < ONSET_FRAMES:
                    continue
                accepted_onset = onset_candidate
            if positive:
                last_speech = frame.ended_at
                speech_frames = len(frames)
            elif last_speech is not None and frame.ended_at >= last_speech + end_silence_s:
                endpoint_reached = True
                break
        remaining = deadline - perf_counter()
        if remaining <= 0 or endpoint_reached:
            break
        await asyncio.sleep(min(FRAME_S, remaining))
    retained = frames[: speech_frames + TRAILING_FRAMES] if speech_frames else []

    def offset(at: float | None) -> str:
        return "none" if at is None else f"{(at - started_at) * 1000:.0f}"

    if frame_ends is not None:

        def dbfs(peak: int) -> str:
            return f"{20 * log10(peak / 32768):.1f}" if peak else "none"

        capture_log.info(
            "Input first_frame_end_ms=%s masked_ms=%.1f signal_first_frame_end_ms=%s "
            "signal_last_frame_end_ms=%s whisper_input_end_ms=%s peak_dbfs=%s "
            "onset_min_peak_dbfs=%s onset_max_peak_dbfs=%s alert_id=%r",
            offset(frame_ends[0] if frame_ends else None),
            masked_samples * 1000 / SAMPLE_RATE,
            offset(first_signal),
            offset(last_signal),
            offset(frame_ends[len(retained) - 1] if retained else None),
            dbfs(capture_peak),
            dbfs(onset_min_peak) if accepted_onset is not None else "none",
            dbfs(onset_max_peak) if accepted_onset is not None else "none",
            alert_id,
        )
    capture_log.info(
        "VAD first_ms=%s last_ms=%s positive_frames=%d retained_ms=%d accepted_first_ms=%s%s "
        "alert_id=%r",
        offset(first_positive),
        offset(last_positive),
        positive_frames,
        len(retained) * FRAME_MS,
        offset(accepted_onset),
        " whisper_ms=skipped" if last_speech is None else "",
        alert_id,
    )
    if last_speech is None:
        return None
    return b"".join(retained), last_speech


async def status(bus: BusClient, alert_id: str, state: VoiceState) -> None:
    await bus.publish("voice.status", VoiceStatus(alert_id=alert_id, state=state).model_dump())


async def interact(
    bus: BusClient,
    mic: Microphone,
    stt: SpeechToText,
    inference: ThreadPoolExecutor,
    alert: VoiceSay,
    monitor: CabinMonitor | None = None,
    latency_log: TextIO | None = None,
    *,
    end_silence_s: float = END_SILENCE_S,
    engagement: bool = False,
) -> None:
    timing = LatencySample()
    try:
        alert = await asyncio.to_thread(english_alert, alert)
        await status(bus, alert.alert_id, "speaking")
        chime = await asyncio.to_thread(read_clip, "chime")
        if chime is None:
            raise FileNotFoundError("assets/audio/chime.wav")
        requested_at = perf_counter()
        clip = await asyncio.to_thread(read_clip, alert.audio_key) if alert.audio_key else None
        if clip is None:
            timing.playback_source = "tts"
            requested_at = perf_counter()
            clip = await synthesize(alert.text, "en")
        else:
            timing.playback_source = "wav"
        with monitor.suppress_playback() if monitor is not None else nullcontext():
            playback = await play_interruptible(clip, chime=chime)
        timing.playback_start_ms = elapsed_ms(requested_at, playback.started_at)

        started_at = playback.ended_at
        cursor = mic.cursor(after=started_at)
        deadline = started_at + alert.listen_s
        capture_log.info(
            "Capture start monotonic_s=%.3f cursor=%d window_ms=%.0f capture_ready_ms=%.0f "
            "output_device=%s output_name=%r alert_id=%r",
            started_at,
            cursor,
            alert.listen_s * 1000,
            elapsed_ms(started_at, perf_counter()),
            playback.output_device,
            playback.output_name,
            alert.alert_id,
        )
        await status(bus, alert.alert_id, "listening")
        capture = await capture_reply(
            mic,
            cursor,
            deadline,
            started_at=started_at,
            alert_id=alert.alert_id,
            end_silence_s=end_silence_s,
        )
        capture_ended_at = perf_counter()
        timing.listening_ms = elapsed_ms(started_at, capture_ended_at)
        if capture is None:
            timing.outcome = "silence"
            reply = VoiceReply(
                alert_id=alert.alert_id, transcript="", intent="NONE", latency_ms=0, path="keyword"
            )
        else:
            pcm, speech_end = capture
            timing.speech_to_capture_end_ms = elapsed_ms(speech_end, capture_ended_at)
            result = await asyncio.get_running_loop().run_in_executor(
                inference, stt.transcribe, pcm, "en"
            )
            capture_log.info(
                "Whisper whisper_ms=%.0f alert_id=%r", result.elapsed_ms, alert.alert_id
            )
            timing.whisper_ms = result.elapsed_ms
            classify_started_at = perf_counter()
            if engagement and result.text.strip():
                intent = "HELP" if classify_keyword(result.text) == "HELP" else "ANSWER"
                classification = ReplyClassification(intent, "keyword")
            else:
                classification = await classify_confirmation(result.text)
            classified_at = perf_counter()
            timing.classification_ms = elapsed_ms(classify_started_at, classified_at)
            timing.speech_to_intent_ms = elapsed_ms(speech_end, classified_at)
            timing.path = classification.path
            if classification.fallback_failed:
                timing.outcome = "provider_failure"
            elif not result.text.strip():
                timing.outcome = "no_transcript"
            elif classification.intent == "UNCLEAR":
                timing.outcome = "unclear"
            else:
                timing.outcome = "reply"
            reply = VoiceReply(
                alert_id=alert.alert_id,
                transcript=result.text,
                intent=classification.intent,
                latency_ms=round(timing.speech_to_intent_ms),
                path=classification.path,
            )
        await bus.publish("voice.reply", reply.model_dump())
    except asyncio.CancelledError:
        timing.outcome = "interrupted"
        raise
    except Exception as error:
        timing.outcome = "error"
        log.error("Voice interaction failed (%s)", type(error).__name__)
    finally:
        if latency_log is not None:
            try:
                timing.write(latency_log)
            except (OSError, ValueError) as error:
                log.error("Voice timing log write failed (%s)", type(error).__name__)
        await status(bus, alert.alert_id, "idle")


async def run_live(
    bus: BusClient, latency_log: TextIO | None = None, *, end_silence_ms: int | None = None
) -> None:
    settings = AudioSettings.load()
    end_silence_s = END_SILENCE_S if end_silence_ms is None else end_silence_ms / 1000
    stt = await asyncio.to_thread(SpeechToText, offline=True)
    ready = asyncio.Event()
    pending: tuple[VoiceSay, bool] | None = None
    engagement_alert: str | None = None
    active: asyncio.Task[None] | None = None

    def on_alert_state(envelope: dict) -> None:
        nonlocal engagement_alert
        data = envelope["data"]
        if data.get("status") == "speaking":
            engagement_alert = (
                data["alert_id"]
                if data.get("type") == "zoning_out" and data.get("tier") == "T1"
                else None
            )

    def on_say(envelope: dict) -> None:
        nonlocal pending
        alert = VoiceSay.model_validate(envelope["data"])
        busy = pending is not None or (active is not None and not active.done())
        if busy and alert.tier != "T3":
            return
        pending = (alert, alert.alert_id == engagement_alert)
        profanity.pause()
        if active is not None and not active.done() and not active.cancelling():
            active.cancel()
        ready.set()

    # CTranslate2 inference already in progress cannot be cancelled. Keep one
    # inference worker, while allowing the replacement alert to speak promptly.
    with Microphone(settings) as mic, SpeechExecutor() as inference:
        if capture_log.isEnabledFor(logging.INFO):
            index, name = mic.describe_input()
            capture_log.info(
                "Capture input device=%d name=%s configured=%s", index, name, settings.input_device
            )
        profanity = CabinProfanity(stt, inference)
        monitor = CabinMonitor(mic, settings, profanity)
        bus.subscribe(["alert.state"], on_alert_state)
        bus.subscribe(["voice.say"], on_say)
        await bus.connect()
        async with asyncio.TaskGroup() as tasks:
            tasks.create_task(monitor.publish(bus))
            while True:
                await ready.wait()
                ready.clear()
                next_alert, pending = pending, None
                if next_alert is None:
                    continue
                alert, engagement = next_alert
                active = asyncio.create_task(
                    interact(
                        bus,
                        mic,
                        stt,
                        inference,
                        alert,
                        monitor,
                        latency_log,
                        end_silence_s=end_silence_s,
                        engagement=engagement,
                    )
                )
                await asyncio.gather(active, return_exceptions=True)
                if pending is None:
                    profanity.resume()
