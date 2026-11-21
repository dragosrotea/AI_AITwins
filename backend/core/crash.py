"""Crash candidate detection (LEAD-T22, CR-01) on core's own vehicle.motion samples.

A spike of accel_mag_g >= spike_g, followed by speed under stop_speed_kmh within stop_within_s,
is a crash candidate: core publishes detector.event crash_candidate (src core, T3). A spike
while the car keeps moving is dropped after stop_within_s (CR-01 acceptance). One candidate
per spike. In the demo the spike comes from the injector (control.inject crash).
"""

from __future__ import annotations

import itertools
import logging
from collections.abc import Callable
from typing import Any

log = logging.getLogger("core.crash")


class CrashDetector:
    def __init__(self, cfg: dict[str, Any], emit: Callable[[dict[str, Any]], None]) -> None:
        self.cfg = cfg
        self.emit = emit
        self._spike_at: float | None = None
        self._spike_g = 0.0
        self._ids = itertools.count(1)

    def on_vehicle(self, data: dict[str, Any], now: float) -> None:
        cfg = self.cfg
        g = float(data["accel_mag_g"])
        if g >= cfg["spike_g"] and self._spike_at is None:
            self._spike_at, self._spike_g = now, g
            log.info("spike %.1f g: waiting for a stop", g)
        if self._spike_at is None:
            return
        if now - self._spike_at > cfg["stop_within_s"]:
            log.info("spike %.1f g without a stop: ignored", self._spike_g)
            self._spike_at = None
            return
        if data["speed_kmh"] < cfg["stop_speed_kmh"]:
            stop_s = round(now - self._spike_at, 1)
            log.warning("crash candidate: %.1f g, stopped after %.1f s", self._spike_g, stop_s)
            self.emit(
                {
                    "event_id": f"crash-{next(self._ids)}",
                    "type": "crash_candidate",
                    "tier": "T3",
                    "score": 100.0,
                    "signals": {
                        "spike_g": round(self._spike_g, 2),
                        "stop_after_s": stop_s,
                        "lat": data.get("lat"),
                        "lon": data.get("lon"),
                    },
                }
            )
            self._spike_at = None
