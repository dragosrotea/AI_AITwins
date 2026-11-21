"""Run the DriveMate voice process against the team broker."""

import argparse
import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from contextlib import nullcontext
from pathlib import Path
from typing import Protocol, TextIO

FEATURE_PERIOD_S = 0.2
REPLY_DELAY_S = 2.0


class VoiceBus(Protocol):
    async def connect(self) -> None: ...

    def subscribe(self, topics: list[str], handler: Callable[[dict], Awaitable[None]]) -> None: ...

    async def publish(self, topic: str, data: dict) -> None: ...


async def publish_reply(bus: VoiceBus, alert_id: str) -> None:
    await asyncio.sleep(REPLY_DELAY_S)
    await bus.publish(
        "voice.reply",
        {
            "alert_id": alert_id,
            "transcript": "Yes, I'm okay.",
            "intent": "OK",
            "latency_ms": random.randint(250, 950),
            "path": "keyword",
        },
    )


async def run_fake(bus: VoiceBus) -> None:
    await bus.connect()
    loop = asyncio.get_running_loop()
    started = loop.time()
    next_tick = started
    async with asyncio.TaskGroup() as replies:

        async def on_say(envelope: dict) -> None:
            data = envelope["data"]
            replies.create_task(publish_reply(bus, data["alert_id"]))

        bus.subscribe(["voice.say"], on_say)

        while True:
            speaking = (loop.time() - started) % 5.0 < 1.6
            baseline_db = -45.0
            loudness_db = baseline_db + random.uniform(-2, 2) + (9 if speaking else 0)
            await bus.publish(
                "audio.features",
                {
                    "loudness_db": round(loudness_db, 1),
                    "baseline_db": baseline_db,
                    "shouting": False,
                    "speaking": speaking,
                    "swear_count_60s": 0,
                    "anger_voice_p": None,
                },
            )
            now = loop.time()
            next_tick += FEATURE_PERIOD_S
            if next_tick <= now:
                next_tick = now + FEATURE_PERIOD_S
            await asyncio.sleep(next_tick - now)


async def run(
    fake: bool, latency_log: TextIO | None = None, *, end_silence_ms: int | None = None
) -> None:
    from backend.bus.client import BusClient

    bus = BusClient("voice")
    try:
        if fake:
            await run_fake(bus)
        else:
            from .live import run_live

            await run_live(bus, latency_log, end_silence_ms=end_silence_ms)
    finally:
        await bus.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fake", action="store_true", help="Simulate voice without audio hardware")
    parser.add_argument(
        "--capture-diagnostics", action="store_true", help="Print live capture timings to stderr"
    )
    parser.add_argument("--latency-log", type=Path, help="Append live timing samples as JSONL")
    parser.add_argument(
        "--end-silence-ms", type=int, help="Override live reply endpoint wait (default: 800 ms)"
    )
    args = parser.parse_args()
    if args.fake and args.latency_log is not None:
        parser.error("--latency-log requires live mode")
    if args.end_silence_ms is not None and (args.fake or args.end_silence_ms <= 0):
        parser.error("--end-silence-ms requires live mode and a positive duration")
    if args.capture_diagnostics:
        logging.basicConfig(format="%(message)s")
        logging.getLogger("backend.voice.capture").setLevel(logging.INFO)

    try:
        with (
            args.latency_log.open("a", encoding="utf-8") if args.latency_log else nullcontext()
        ) as latency_log:
            asyncio.run(run(args.fake, latency_log, end_silence_ms=args.end_silence_ms))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
