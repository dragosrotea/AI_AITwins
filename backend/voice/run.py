"""Run the DriveMate voice process against the team broker."""

import argparse
import asyncio
import random
from collections.abc import Awaitable, Callable
from typing import Protocol

FEATURE_PERIOD_S = 0.2
REPLY_DELAY_S = 2.0


class VoiceBus(Protocol):
    async def connect(self) -> None: ...

    def subscribe(self, topics: list[str], handler: Callable[[dict], Awaitable[None]]) -> None: ...

    async def publish(self, topic: str, data: dict) -> None: ...


async def publish_reply(bus: VoiceBus, alert_id: str, lang: str) -> None:
    await asyncio.sleep(REPLY_DELAY_S)
    await bus.publish(
        "voice.reply",
        {
            "alert_id": alert_id,
            "transcript": "Da, sunt bine." if lang == "ro" else "Yes, I'm okay.",
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
            replies.create_task(publish_reply(bus, data["alert_id"], data["lang"]))

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


async def run(fake: bool) -> None:
    from backend.bus.client import BusClient

    bus = BusClient("voice")
    try:
        if fake:
            await run_fake(bus)
        else:
            from .live import run_live

            await run_live(bus)
    finally:
        await bus.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fake", action="store_true", help="Simulate voice without audio hardware")
    args = parser.parse_args()

    try:
        asyncio.run(run(args.fake))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
