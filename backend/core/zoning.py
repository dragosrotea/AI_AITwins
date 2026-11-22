"""Zoning-out fusion (LEAD-T19): raises a T1 zoning_out from CV's visual score and monotony.

ZO-01: score = weights.visual * zoning_visual_p (CV: fixed stare, low blink rate, little head
movement; 0 without a face) + weights.monotony when the drive is monotonous: every vehicle.motion
sample in the last monotony_window_s above monotony_speed_kmh, spread under monotony_spread_kmh.
At score >= threshold core publishes detector.event zoning_out (src core, tier T1), once per
episode. ZO-06: nothing is raised while context.road says HIGH load or stop-and-go traffic.

What is spoken (trivia, a fun fact, or a plain check for "quiet" drivers), the 3-in-30-min break
(ZO-05) and the grading (ZO-04) are handled where the event is routed (backend/core/events.py).
"""

from __future__ import annotations

import itertools
import logging
from collections import deque
from collections.abc import Callable
from typing import Any

log = logging.getLogger("core.zoning")


def road_blocked(data: dict[str, Any]) -> bool:
    """ZO-06: HIGH road load or stop-and-go traffic (same reading as the router's)."""
    traffic = (data.get("traffic") or "").casefold().replace("_", "-").replace(" ", "-")
    return data["load"] == "HIGH" or traffic == "stop-and-go"


class ZoningFusion:
    def __init__(self, cfg: dict[str, Any], emit: Callable[[dict[str, Any]], None]) -> None:
        self.cfg = cfg
        self.emit = emit
        self.visual = 0.0
        self.blocked = False
        self._speeds: deque[tuple[float, float]] = deque()  # (time, km/h)
        self._in_episode = False
        self._ids = itertools.count(1)

    def on_driver_state(self, data: dict[str, Any], now: float) -> None:
        self.visual = float(data["zoning_visual_p"]) if data["face_present"] else 0.0
        self.evaluate(now)

    def on_vehicle(self, data: dict[str, Any], now: float) -> None:
        self._speeds.append((now, float(data["speed_kmh"])))
        window = self.cfg["monotony_window_s"]
        while self._speeds and now - self._speeds[0][0] > window:
            self._speeds.popleft()

    def on_road(self, data: dict[str, Any]) -> None:
        self.blocked = road_blocked(data)

    def monotonous(self, now: float) -> bool:
        cfg = self.cfg
        if not self._speeds or now - self._speeds[0][0] < 0.95 * cfg["monotony_window_s"]:
            return False  # not enough history yet
        speeds = [v for _, v in self._speeds]
        return min(speeds) > cfg["monotony_speed_kmh"] and (
            max(speeds) - min(speeds) < cfg["monotony_spread_kmh"]
        )

    def score(self, now: float) -> float:
        w = self.cfg["weights"]
        return min(1.0, w["visual"] * self.visual + w["monotony"] * self.monotonous(now))

    def evaluate(self, now: float) -> float:
        score = self.score(now)
        if self.blocked or score < self.cfg["threshold"]:
            self._in_episode = False
            return score
        if not self._in_episode:
            self._in_episode = True
            monotony = self.monotonous(now)
            log.info("zoning_out %.2f (visual %.2f, monotony %s)", score, self.visual, monotony)
            self.emit(
                {
                    "event_id": f"zoning-{next(self._ids)}",
                    "type": "zoning_out",
                    "tier": "T1",
                    "score": round(100 * score, 1),
                    "signals": {"visual": round(self.visual, 2), "monotony": monotony},
                }
            )
        return score

    def inject(self) -> None:
        """Demo trigger (control.inject zoning_out): raise the same event the fusion would.

        Everything after the event stays real: opt-in, road hold, cooldowns, ZO-05 break, grading.
        """
        log.info("zoning_out injected")
        self.emit(
            {
                "event_id": f"zoning-{next(self._ids)}",
                "type": "zoning_out",
                "tier": "T1",
                "score": 100.0,
                "signals": {"injected": True},
            }
        )
