from __future__ import annotations

import math
import uuid
from collections import deque
from collections.abc import Mapping
from typing import Any

from backend.cv.settings import float_setting


def _float_value(settings: dict[str, Any], name: str) -> float:
    value = settings.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Missing or invalid CV setting {name}")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"CV setting {name} must be finite")
    return parsed


def _integer_value(settings: dict[str, Any], name: str, *, minimum: int = 0) -> int:
    value = settings.get(name)
    if type(value) is not int or value < minimum:
        raise ValueError(f"CV setting {name} must be an integer >= {minimum}")
    return value


class DriverStateMetrics:
    """Add live eye-closure duration and PERCLOS to a per-frame driver state."""

    def __init__(self, settings: dict[str, Any]) -> None:
        self._closed_probability_threshold = float_setting(
            settings, "eye_state", "closed_probability_threshold"
        )
        self._perclos_window_s = _float_value(settings, "perclos_window_s")
        if self._perclos_window_s <= 0:
            raise ValueError("CV setting cv.perclos_window_s must be positive")
        self._segments: deque[tuple[float, float, bool]] = deque()
        self._previous_timestamp: float | None = None
        self._previous_face_present = False
        self._previous_closed = False
        self._closed_since: float | None = None

    def update(self, data: Mapping[str, Any], now: float) -> dict[str, Any]:
        if not math.isfinite(now):
            raise ValueError("Driver state timestamp must be finite")
        state = dict(data)
        face_present = bool(state["face_present"])
        eye_closed_p = float(state["eye_closed_p"])
        if not math.isfinite(eye_closed_p) or not 0.0 <= eye_closed_p <= 1.0:
            raise ValueError("eye_closed_p must be finite and between 0 and 1")
        closed = face_present and eye_closed_p >= self._closed_probability_threshold

        if (
            self._previous_timestamp is not None
            and now >= self._previous_timestamp
            and self._previous_face_present
        ):
            self._segments.append((self._previous_timestamp, now, self._previous_closed))
        self._previous_timestamp = now
        self._previous_face_present = face_present
        self._previous_closed = closed

        if closed:
            if self._closed_since is None:
                self._closed_since = now
            state["eyes_closed_ms"] = max(0, round((now - self._closed_since) * 1000))
        else:
            self._closed_since = None
            state["eyes_closed_ms"] = 0

        cutoff = now - self._perclos_window_s
        while self._segments and self._segments[0][1] <= cutoff:
            self._segments.popleft()
        observed_s = closed_s = 0.0
        for start, end, was_closed in self._segments:
            duration = max(0.0, end - max(start, cutoff))
            observed_s += duration
            if was_closed:
                closed_s += duration
        state["perclos_60s"] = (
            closed_s / observed_s if observed_s >= self._perclos_window_s else 0.0
        )
        return state


class VisualEventDetector:
    """Convert driver-state signals into one bus event per sustained episode."""

    def __init__(self, settings: dict[str, Any]) -> None:
        self._yaw_threshold = _float_value(settings, "eyes_off_yaw_deg")
        self._pitch_threshold = _float_value(settings, "eyes_off_pitch_deg")
        self._distraction_hold_s = _float_value(settings, "eyes_off_s")
        self._microsleep_s = _float_value(settings, "microsleep_s")
        self._perclos_threshold = _float_value(settings, "perclos_threshold")
        self._yawn_count = _integer_value(settings, "yawn_count", minimum=1)
        self._no_face_warning_s = _float_value(settings, "no_face_warning_s")
        self._cooldown_s = float_setting(settings, "visual_events", "cooldown_s")
        for name, value in (
            ("eyes_off_s", self._distraction_hold_s),
            ("microsleep_s", self._microsleep_s),
            ("no_face_warning_s", self._no_face_warning_s),
            ("visual_events.cooldown_s", self._cooldown_s),
        ):
            if value <= 0:
                raise ValueError(f"CV setting {name} must be positive")
        self._condition_since: dict[str, float] = {}
        self._emitted_this_episode: set[str] = set()
        self._last_emitted: dict[str, float] = {}

    def update(self, state: Mapping[str, Any], now: float) -> list[dict[str, Any]]:
        if not math.isfinite(now):
            raise ValueError("Visual event timestamp must be finite")
        face_present = bool(state["face_present"])
        head = state["head"]
        yaw = float(head["yaw"])
        pitch = float(head["pitch"])
        eyes_closed_ms = int(state["eyes_closed_ms"])
        perclos = float(state["perclos_60s"])
        yawns = int(state["yawns_10m"])
        if not all(math.isfinite(value) for value in (yaw, pitch, perclos)):
            raise ValueError("Visual event signals must be finite")

        conditions = {
            "distraction": face_present
            and (abs(yaw) > self._yaw_threshold or pitch < self._pitch_threshold),
            "microsleep": face_present and eyes_closed_ms >= self._microsleep_s * 1000,
            "drowsiness": face_present
            and (perclos >= self._perclos_threshold or yawns >= self._yawn_count),
            "no_face": not face_present,
        }
        for event_type, active in conditions.items():
            if active:
                self._condition_since.setdefault(event_type, now)
            else:
                self._condition_since.pop(event_type, None)
                self._emitted_this_episode.discard(event_type)

        elapsed = {
            event_type: now - started for event_type, started in self._condition_since.items()
        }
        ready = {
            "distraction": elapsed.get("distraction", 0.0) >= self._distraction_hold_s,
            "microsleep": conditions["microsleep"],
            "drowsiness": conditions["drowsiness"],
            "no_face": elapsed.get("no_face", 0.0) >= self._no_face_warning_s,
        }
        tiers = {
            "distraction": "T2",
            "microsleep": "T3",
            "drowsiness": "T2",
            "no_face": "T2",
        }
        scores = {
            "distraction": 70.0,
            "microsleep": 95.0,
            "drowsiness": 70.0,
            "no_face": 70.0,
        }
        signals = {
            "distraction": {
                "yaw": yaw,
                "pitch": pitch,
                "held_s": elapsed.get("distraction", 0.0),
            },
            "microsleep": {"eyes_closed_ms": eyes_closed_ms},
            "drowsiness": {"perclos_60s": perclos, "yawns_10m": yawns},
            "no_face": {"face_absent_s": elapsed.get("no_face", 0.0)},
        }

        events = []
        for event_type, is_ready in ready.items():
            if not is_ready or event_type in self._emitted_this_episode:
                continue
            self._emitted_this_episode.add(event_type)
            last_emitted = self._last_emitted.get(event_type)
            if last_emitted is not None and now - last_emitted < self._cooldown_s:
                continue
            event = {
                "event_id": str(uuid.uuid4()),
                "type": event_type,
                "tier": tiers[event_type],
                "score": scores[event_type],
                "signals": signals[event_type],
            }
            events.append(event)
            self._last_emitted[event_type] = now
        return events
