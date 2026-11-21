"""Speak a DriveMate demo line or play a local audio file."""

import argparse
import asyncio
from pathlib import Path
from time import perf_counter

from .tts import DEMO_LINES, play, synthesize


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--lang", choices=DEMO_LINES)
    action.add_argument("--play", type=Path, metavar="FILE")
    args = parser.parse_args()
    try:
        if args.play is not None:
            play(args.play.read_bytes())
        else:
            requested_at = perf_counter()
            play(
                asyncio.run(synthesize(DEMO_LINES[args.lang], args.lang)), requested_at
            )
    except Exception as error:  # noqa: BLE001 - present provider and device errors at the CLI
        parser.exit(1, f"Voice playback failed: {error}\n")


if __name__ == "__main__":
    main()
