"""Score a local CSV manifest of DriveMate confirmation recordings."""

import argparse
import asyncio
import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import miniaudio

from backend.audio.devices import SAMPLE_RATE

from .intents import classify_confirmation
from .stt import SpeechToText

LABELS = {"OK", "NOT_OK", "HELP"}
EXPECTED_SPEAKERS = 4
CLIPS_PER_SPEAKER = 32
REQUIRED_CORRECT = 116


@dataclass(frozen=True)
class Recording:
    path: Path
    speaker_id: str
    expected_intent: str


def load_manifest(path: Path) -> list[Recording]:
    recordings = []
    seen = set()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"path", "lang", "speaker_id", "expected_intent"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError("Manifest requires path, lang, speaker_id, expected_intent columns")
        for row in reader:
            filename, lang, speaker, expected = (
                (row[key] or "").strip()
                for key in ("path", "lang", "speaker_id", "expected_intent")
            )
            if lang == "ro":
                continue  # Preserve the original manifest and its archived recordings.
            if not filename or lang != "en" or not speaker or expected not in LABELS:
                raise ValueError(f"Invalid recording entry on CSV line {reader.line_num}")
            wav = (path.parent / filename).resolve()
            if wav.suffix.lower() != ".wav" or wav in seen:
                raise ValueError(f"Expected a unique WAV path on CSV line {reader.line_num}")
            seen.add(wav)
            recordings.append(Recording(wav, speaker, expected))
    if not recordings:
        raise ValueError("Manifest contains no English recordings")
    return recordings


async def evaluate(recordings: list[Recording]) -> int:
    totals: dict[str, Counter] = {}
    available = []
    for entry in recordings:
        counts = totals.setdefault(entry.speaker_id, Counter())
        counts["listed"] += 1
        if entry.path.is_file():
            available.append(entry)
        else:
            counts["missing"] += 1
            print(f"MISSING en {entry.speaker_id}: {entry.path}")

    stt = SpeechToText(offline=True) if available else None
    for entry in available:
        assert stt is not None
        counts = totals[entry.speaker_id]
        label = f"en {entry.speaker_id} {entry.path.name}"
        try:
            audio = miniaudio.decode_file(str(entry.path), nchannels=1, sample_rate=SAMPLE_RATE)
            transcript = stt.transcribe(audio.samples.tobytes(), "en").text
        except Exception as error:  # noqa: BLE001 - continue scoring independent recordings
            counts["audio_errors"] += 1
            print(f"AUDIO ERROR {label}: {type(error).__name__}")
            continue
        result = await classify_confirmation(transcript)
        counts["scored"] += 1
        counts["correct"] += result.intent == entry.expected_intent
        if result.fallback_failed:
            counts["provider_failures"] += 1
            print(f"PROVIDER FALLBACK FAILED {label}: unavailable, timed out, or invalid output")
        if result.intent != entry.expected_intent:
            print(
                f"FAIL {label}: expected={entry.expected_intent} actual={result.intent}"
                f" path={result.path} transcript={transcript!r}"
            )
    overall: Counter = Counter()
    for counts in totals.values():
        overall.update(counts)
    for label, counts in [*sorted(totals.items()), ("English overall", overall)]:
        scored = counts["scored"]
        accuracy = f"{counts['correct'] / scored:.1%}" if scored else "n/a"
        print(
            f"{label}: listed={counts['listed']} scored={scored} correct={counts['correct']}"
            f" accuracy={accuracy} missing={counts['missing']}"
            f" audio_errors={counts['audio_errors']}"
            f" provider_failures={counts['provider_failures']}"
        )
    complete = len(totals) == EXPECTED_SPEAKERS and all(
        counts["listed"] == counts["scored"] == CLIPS_PER_SPEAKER for counts in totals.values()
    )
    passed = (
        complete and not overall["provider_failures"] and overall["correct"] >= REQUIRED_CORRECT
    )
    gate = (
        "PASS" if passed else "FAIL" if complete and not overall["provider_failures"] else "PENDING"
    )
    print(
        f"T11 gate: {gate}. {EXPECTED_SPEAKERS} speakers × {CLIPS_PER_SPEAKER} clips"
        f" = {EXPECTED_SPEAKERS * CLIPS_PER_SPEAKER}; {REQUIRED_CORRECT} correct required."
    )
    return int(not passed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    try:
        recordings = load_manifest(args.manifest)
    except (OSError, ValueError, csv.Error) as error:
        parser.exit(1, f"Manifest error: {error}\n")
    try:
        code = asyncio.run(evaluate(recordings))
    except Exception as error:  # noqa: BLE001 - report model startup failure at the CLI
        parser.exit(
            1, f"Evaluation stopped ({type(error).__name__}). Check the Whisper base cache.\n"
        )
    raise SystemExit(code)


if __name__ == "__main__":
    main()
