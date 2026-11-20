"""Broker load test (LEAD-T02 acceptance).

Two clients each publish at --rate msg/s on their own topic and subscribe to the other's.
Prints received rates, loss and the broker's RSS every few seconds; memory must stay flat.

    python -m backend.bus.broker &
    python scripts/bus_bench.py --rate 1000 --seconds 30 --broker-pid $!
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import time

import websockets

PAYLOAD = {"face_present": True, "eye_closed_p": 0.1, "head": {"yaw": 1.0, "pitch": 2.0}}


def rss_mb(pid: int | None) -> str:
    if pid is None:
        return "n/a"
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True)
    return f"{int(out.stdout.strip()) / 1024:.1f} MB" if out.stdout.strip() else "gone"


async def peer(url: str, me: str, other: str, rate: int, seconds: float, counts: dict) -> None:
    async with websockets.connect(url) as ws:
        await ws.send(json.dumps({"op": "sub", "topics": [f"bench.{other}"]}))
        await asyncio.sleep(0.2)

        async def reader() -> None:
            async for _ in ws:
                counts[me] += 1

        task = asyncio.create_task(reader())
        start = time.perf_counter()
        sent = 0
        while (elapsed := time.perf_counter() - start) < seconds:
            due = int(elapsed * rate)
            while sent < due:
                msg = {"op": "pub", "topic": f"bench.{me}", "ts": time.time(), "src": "core"}
                await ws.send(json.dumps(msg | {"data": PAYLOAD | {"seq": sent}}))
                sent += 1
            await asyncio.sleep(0.001)
        counts[f"{me}_sent"] = sent
        await asyncio.sleep(0.5)  # let the tail arrive
        task.cancel()


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="ws://127.0.0.1:8765/bus")
    p.add_argument("--rate", type=int, default=1000, help="messages per second per client")
    p.add_argument("--seconds", type=float, default=30)
    p.add_argument("--broker-pid", type=int)
    args = p.parse_args()

    counts = {"a": 0, "b": 0}
    print(f"broker RSS at start: {rss_mb(args.broker_pid)}")
    run = asyncio.gather(
        peer(args.url, "a", "b", args.rate, args.seconds, counts),
        peer(args.url, "b", "a", args.rate, args.seconds, counts),
    )
    task = asyncio.ensure_future(run)
    last = dict(counts)
    t0 = time.perf_counter()
    while not task.done():
        await asyncio.sleep(5)
        rates = {k: (counts[k] - last[k]) / 5 for k in ("a", "b")}
        last = dict(counts)
        print(
            f"t={time.perf_counter() - t0:5.1f}s  a recv {rates['a']:.0f}/s  "
            f"b recv {rates['b']:.0f}/s  broker RSS {rss_mb(args.broker_pid)}"
        )
    await task
    for me, other in (("a", "b"), ("b", "a")):
        sent, got = counts[f"{other}_sent"], counts[me]
        print(f"{me} received {got}/{sent} from {other} ({100 * (sent - got) / sent:.2f}% lost)")
    print(f"broker RSS at end: {rss_mb(args.broker_pid)}")


if __name__ == "__main__":
    asyncio.run(main())
