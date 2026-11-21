"""Render local English WAVs from templates.yaml or the demo line.

The catalog maps alert keys to language keys and lists of strings.
For example, distraction: {en: ["Are you okay?"]} yields distraction_en_1.wav.
"""

import argparse
import asyncio
import math
import struct
import wave
from pathlib import Path
from tempfile import NamedTemporaryFile

import miniaudio

from backend.llm.engagement import engagement_clips
from backend.llm.templates import CATALOG, english_templates

from .tts import DEMO_LINES, synthesize

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "assets" / "audio"
RATE = 24_000


def speech_entries(path: Path) -> list[tuple[str, str, str]]:
    entries = (
        [
            (key, "en", text)
            for variants in english_templates(path).values()
            for text, key in variants
        ]
        if path.exists()
        else []
    )
    return entries or [("demo_en", "en", DEMO_LINES["en"])]


def write_wav(name: str, pcm: bytes) -> None:
    destination = ASSETS / f"{name}.wav"
    with NamedTemporaryFile(dir=ASSETS, suffix=".tmp", delete=False) as temp:
        temporary = Path(temp.name)
    try:
        with wave.open(str(temporary), "wb") as file:
            file.setparams((1, 2, RATE, 0, "NONE", "not compressed"))
            file.writeframes(pcm)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Wrote {destination.relative_to(ROOT)}")


def tone(hz: int, duration_s: float) -> bytes:
    samples = round(RATE * duration_s)
    fade = RATE // 100
    return b"".join(
        struct.pack(
            "<h",
            int(
                5000
                * min(1, i / fade, (samples - 1 - i) / fade)
                * math.sin(math.tau * hz * i / RATE)
            ),
        )
        for i in range(samples)
    )


async def render(tones_only: bool, *, engagement_only: bool = False) -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    if engagement_only:
        entries = [(key, "en", text) for key, text in engagement_clips().items()]
    else:
        entries = [] if tones_only else speech_entries(CATALOG)
        write_wav("chime", tone(880, 0.20))
        write_wav("alarm", (tone(1200, 0.25) + bytes(RATE // 5 * 2)) * 2 + tone(1200, 0.25))
    for name, lang, text in entries:
        audio = await synthesize_template(name, text, lang)
        pcm = miniaudio.decode(audio, nchannels=1, sample_rate=RATE)
        write_wav(name, pcm.samples.tobytes())


async def synthesize_template(name: str, text: str, lang: str) -> bytes:
    from edge_tts.exceptions import NoAudioReceived

    for delay in (1, 2):
        try:
            return await synthesize(text, lang)
        except (NoAudioReceived, TimeoutError) as error:
            print(f"{name}: {type(error).__name__}. Retrying in {delay} s.", flush=True)
            await asyncio.sleep(delay)
    return await synthesize(text, lang)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--tones-only", action="store_true", help="Render tones offline")
    modes.add_argument(
        "--engagement-only", action="store_true", help="Render the offline English bank"
    )
    args = parser.parse_args()
    try:
        asyncio.run(render(args.tones_only, engagement_only=args.engagement_only))
    except Exception as error:  # noqa: BLE001 - present provider and file errors at the CLI
        parser.exit(1, f"Audio prerender failed: {error}\n")


if __name__ == "__main__":
    main()
