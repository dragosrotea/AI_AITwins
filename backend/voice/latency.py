"""Write and report live voice timing samples without retaining message content."""

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Literal, TextIO


@dataclass
class LatencySample:
    outcome: Literal[
        "reply",
        "silence",
        "interrupted",
        "provider_failure",
        "provider_timeout",
        "unclear",
        "no_transcript",
        "error",
    ] = "error"
    path: Literal["keyword", "llm"] | None = None
    playback_source: Literal["tts", "wav"] | None = None
    playback_start_ms: float | None = None
    listening_ms: float | None = None
    whisper_ms: float | None = None
    classification_ms: float | None = None
    speech_to_intent_ms: float | None = None
    speech_to_capture_end_ms: float | None = None

    def write(self, stream: TextIO) -> None:
        stream.write(json.dumps(asdict(self), allow_nan=False) + "\n")
        stream.flush()


def elapsed_ms(start: float, end: float) -> float:
    return max(0.0, (end - start) * 1000)


def report(samples: list[LatencySample]) -> str:
    outcomes = Counter(sample.outcome for sample in samples)
    lines = [
        f"Live alerts: {len(samples)}",
        "Outcomes: "
        + ", ".join(
            f"{name}={outcomes[name]}"
            for name in (
                "reply",
                "silence",
                "interrupted",
                "provider_failure",
                "provider_timeout",
                "unclear",
                "no_transcript",
                "error",
            )
        ),
        "Successful replies (milliseconds):",
    ]
    successful = [sample for sample in samples if sample.outcome == "reply"]

    def add(label: str, values: list[float | None]) -> None:
        measured = [value for value in values if value is not None]
        middle = f"{median(measured):.1f}" if measured else "n/a"
        lines.append(f"{label}: n={len(measured)}, median={middle}")

    for source, label in (("tts", "TTS request"), ("wav", "Local WAV request")):
        add(
            f"{label} to playback start",
            [sample.playback_start_ms for sample in successful if sample.playback_source == source],
        )
    add("Listening window", [sample.listening_ms for sample in successful])
    add("End of speech to capture end", [sample.speech_to_capture_end_ms for sample in successful])
    add("Whisper", [sample.whisper_ms for sample in successful])
    add("Intent classification", [sample.classification_ms for sample in successful])
    add("End of speech to intent (overall)", [sample.speech_to_intent_ms for sample in successful])
    for path in ("keyword", "llm"):
        cohort = [sample for sample in successful if sample.path == path]
        add(
            f"End of speech to capture end ({path})",
            [sample.speech_to_capture_end_ms for sample in cohort],
        )
        add(f"Whisper ({path})", [sample.whisper_ms for sample in cohort])
        add(f"Intent classification ({path})", [sample.classification_ms for sample in cohort])
        add(
            f"End of speech to intent ({path})",
            [sample.speech_to_intent_ms for sample in cohort],
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    args = parser.parse_args()
    try:
        with args.log.open(encoding="utf-8") as stream:
            samples = [LatencySample(**json.loads(line)) for line in stream if line.strip()]
    except (OSError, ValueError, TypeError) as error:
        parser.exit(1, f"Cannot read latency log ({type(error).__name__}).\n")
    print(report(samples))


if __name__ == "__main__":
    main()
