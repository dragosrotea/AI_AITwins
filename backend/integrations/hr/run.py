"""Publish simulated heart-rate data to the DriveMate bus."""

from __future__ import annotations

import argparse
import asyncio
import math
import random
import time
from typing import Any

RATE_HZ = 1.0
DEFAULT_RESTING_BPM = 65.0


def heart_rate(elapsed_s: float, resting_bpm: float, scenario: str) -> float:
    """Return a deterministic, gently varying BPM for the requested test case."""
    if scenario == "stress":
        target = resting_bpm + 36.0
    elif scenario == "fatigue":
        target = resting_bpm * 0.85
    elif scenario == "normal":
        target = resting_bpm + 8.0
    else:
        # 20 s baseline, 65 s sustained stress, then 65 s low-HR monotony cue.
        phase = elapsed_s % 150.0
        target = (
            resting_bpm + 8.0
            if phase < 20.0
            else resting_bpm + 36.0
            if phase < 85.0
            else resting_bpm * 0.85
        )
    return round(max(35.0, target + 1.2 * math.sin(elapsed_s / 7.0)), 1)


async def run_sim(bus: Any, scenario: str) -> None:
    resting_bpm = DEFAULT_RESTING_BPM

    async def on_trip(envelope: dict[str, Any]) -> None:
        nonlocal resting_bpm
        data = envelope.get("data", {})
        pretrip = data.get("pretrip") or {}
        value = pretrip.get("resting_hr")
        if data.get("action") == "start" and isinstance(value, (int, float)) and value > 0:
            resting_bpm = float(value)

    bus.subscribe(["control.trip"], on_trip)
    await bus.connect()
    loop = asyncio.get_running_loop()
    started = loop.time()
    next_tick = started

    while True:
        elapsed = loop.time() - started
        bpm = heart_rate(elapsed, resting_bpm, scenario)
        # Approximate RR intervals (ms) around the current mean interval.
        mean_rr = 60_000.0 / bpm
        rr_ms = [round(mean_rr + random.uniform(-18.0, 18.0), 1) for _ in range(5)]
        await bus.publish("bio.hr", {"bpm": bpm, "rr_ms": rr_ms, "source": "sim"})
        next_tick += 1.0 / RATE_HZ
        next_tick = max(next_tick, loop.time())
        await asyncio.sleep(next_tick - loop.time())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim", action="store_true", required=True)
    parser.add_argument("--scenario", choices=("cycle", "normal", "stress", "fatigue"), default="cycle")
    args = parser.parse_args()

    from backend.bus.client import BusClient

    try:
        asyncio.run(run_sim(BusClient("hr"), args.scenario))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
