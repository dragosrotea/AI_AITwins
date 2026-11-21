"""Publish simulated road-demand context to the DriveMate bus."""

from __future__ import annotations

import argparse
import asyncio
from typing import Any

RATE_HZ = 1.0
ROAD_STATES = (
    {"load": "LOW", "traffic": "Free flow", "next_maneuver": "Continue straight", "dist_m": 900.0},
    {"load": "MEDIUM", "traffic": "Moderate", "next_maneuver": "Keep right", "dist_m": 350.0},
    {"load": "HIGH", "traffic": "Dense junction", "next_maneuver": "Turn left", "dist_m": 80.0},
    {"load": "MEDIUM", "traffic": "Moderate", "next_maneuver": "Continue straight", "dist_m": 500.0},
)


async def run_sim(bus: Any, scenario: str) -> None:
    await bus.connect()
    loop = asyncio.get_running_loop()
    started = loop.time()
    next_tick = started

    while True:
        elapsed = loop.time() - started
        if scenario == "cycle":
            # Rotate through all loads so CP0/CP2 can exercise HIGH-load holds.
            state = ROAD_STATES[int(elapsed // 15.0) % len(ROAD_STATES)]
        else:
            state = ROAD_STATES[{"low": 0, "medium": 1, "high": 2}[scenario]]
        await bus.publish("context.road", state)
        next_tick += 1.0 / RATE_HZ
        next_tick = max(next_tick, loop.time())
        await asyncio.sleep(next_tick - loop.time())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim", action="store_true", required=True)
    parser.add_argument("--scenario", choices=("cycle", "low", "medium", "high"), default="cycle")
    args = parser.parse_args()

    from backend.bus.client import BusClient

    try:
        asyncio.run(run_sim(BusClient("maps"), args.scenario))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
