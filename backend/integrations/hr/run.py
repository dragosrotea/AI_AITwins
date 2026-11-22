"""Publish heart-rate data to the DriveMate bus (UI-T15).

Simulated heart rate (the default) or a replay of a Suunto .FIT workout file (--fit).
The live Bluetooth LE mode was removed on 22 Nov: it did not work on every laptop and the
demo uses the simulation.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import math
import random
from typing import Any

log = logging.getLogger("hr")

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
    """Run simulated heart-rate publisher (UI-T05 / CP0 fake)."""
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


async def run_fit_replay(bus: Any, fit_path: str, rate_hz: float = RATE_HZ) -> None:
    """Replay heart rate values from a Suunto .FIT workout file at rate_hz (1 Hz)."""
    from backend.integrations.hr.fit import parse_fit_heart_rate

    hr_values = parse_fit_heart_rate(fit_path)
    if not hr_values:
        raise ValueError(f"No valid heart rate records found in {fit_path}")

    log.info(
        "Loaded %d HR records from %s (min: %.1f, max: %.1f BPM)",
        len(hr_values),
        fit_path,
        min(hr_values),
        max(hr_values),
    )
    print(
        f"[hr] Loaded {len(hr_values)} Suunto HR records from {fit_path} "
        f"(BPM range: {min(hr_values):.0f} - {max(hr_values):.0f}). Replaying at {rate_hz} Hz..."
    )

    await bus.connect()
    loop = asyncio.get_running_loop()
    interval = 1.0 / rate_hz
    next_tick = loop.time()
    idx = 0

    while True:
        bpm = hr_values[idx]
        mean_rr = 60_000.0 / bpm
        rr_ms = [round(mean_rr + random.uniform(-10.0, 10.0), 1) for _ in range(3)]
        payload = {"bpm": float(bpm), "rr_ms": rr_ms, "source": "suunto"}
        await bus.publish("bio.hr", payload)

        idx = (idx + 1) % len(hr_values)
        next_tick += interval
        next_tick = max(next_tick, loop.time())
        await asyncio.sleep(next_tick - loop.time())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fit",
        "--replay",
        dest="fit_path",
        default=None,
        help="Path to a Suunto .FIT workout file to replay at 1 Hz",
    )
    parser.add_argument(
        "--sim",
        "--fake",
        dest="sim",
        action="store_true",
        help="Simulated heart rate (the default; kept so run_all --fake works)",
    )
    parser.add_argument(
        "--scenario",
        choices=("cycle", "normal", "stress", "fatigue"),
        default="cycle",
        help="Simulation scenario (default: cycle)",
    )
    args = parser.parse_args()

    from backend.bus.client import BusClient

    bus = BusClient("hr")
    try:
        if args.fit_path:
            asyncio.run(run_fit_replay(bus, args.fit_path))
        else:
            asyncio.run(run_sim(bus, args.scenario))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
