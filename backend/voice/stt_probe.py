"""Capture an English sample from the configured microphone."""

import argparse
from time import monotonic, sleep

from backend.audio.devices import FRAME_MS, AudioSettings, Microphone

from .stt import Language, SpeechToText

SAMPLE_SECONDS = 3
SAMPLE_FRAMES = SAMPLE_SECONDS * 1000 // FRAME_MS
SAMPLES: tuple[tuple[Language, str], ...] = (("en", "Hello, I feel good today."),)


def capture_sample(mic: Microphone) -> bytes:
    cursor = mic.cursor()
    print("Speak now (3 seconds).", flush=True)
    deadline = monotonic() + SAMPLE_SECONDS + 2
    frames = []
    while len(frames) < SAMPLE_FRAMES:
        for frame in mic.read_since(cursor)[: SAMPLE_FRAMES - len(frames)]:
            if frame.sequence != cursor + 1:
                raise RuntimeError("Microphone buffer overran during capture")
            frames.append(frame.pcm)
            cursor = frame.sequence
        if len(frames) == SAMPLE_FRAMES:
            break
        if monotonic() >= deadline:
            raise TimeoutError(f"Captured only {len(frames)} of {SAMPLE_FRAMES} frames")
        sleep(FRAME_MS / 1000)
    return b"".join(frames)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--offline", action="store_true", help="Load the model from the local cache"
    )
    args = parser.parse_args()
    try:
        settings = AudioSettings.load()
        print("Loading multilingual Whisper base (CPU, int8)...", flush=True)
        stt = SpeechToText(offline=args.offline)
        print(f"Model startup: {stt.load_ms:.0f} ms")
        with Microphone(settings) as mic:
            for lang, example in SAMPLES:
                input(f"[{lang}] Press Enter when ready to say: {example}\n")
                result = stt.transcribe(capture_sample(mic), lang)
                print(f"[{lang}] Transcription: {result.elapsed_ms:.0f} ms")
                print(f"[{lang}] Transcript: {result.text or '(no speech recognized)'}")
    except (KeyboardInterrupt, EOFError):
        parser.exit(1, "STT probe cancelled.\n")
    except Exception as error:  # noqa: BLE001 - present model and device errors at the CLI
        parser.exit(1, f"STT probe failed: {error}\n")


if __name__ == "__main__":
    main()
