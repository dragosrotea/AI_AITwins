"""Ground trip summaries in recorded events and bounded provider suggestions."""

import json
from collections import Counter
from math import isfinite
from statistics import median
from typing import Any, get_args

from backend.schemas import EventType

from .replies import _request

ALERT_TYPES = frozenset(get_args(EventType))
SUGGESTIONS = {
    "drowsiness": "Plan a rest stop before your next long drive.",
    "microsleep": "Rest before driving again if you are still sleepy.",
    "phone": "Set your phone to Do Not Disturb before setting off.",
    "distraction": "Finish setup before departing so it does not divide your attention.",
    "road_rage": "If traffic feels stressful, take a break at the next safe stop.",
    "break_due": "Schedule your first break before setting off.",
    "no_face": "Check the driver camera position while parked before your next trip.",
    "zoning_out": "Plan a stop to stretch if a long stretch of road feels monotonous.",
}


def trip_stats(log: list[dict]) -> dict[str, Any]:
    start = stop = None
    alerts: dict[str, str] = {}
    closed_times: dict[str, int] = {}
    reply_times: dict[str, int] = {}
    for entry in log:
        if not isinstance(entry, dict):
            continue
        topic, data = entry.get("topic"), entry.get("data", {})
        if not isinstance(data, dict):
            continue
        if topic == "control.trip":
            ts = entry.get("ts")
            if type(ts) in (float, int) and isfinite(ts):
                if data.get("action") == "start":
                    start = ts
                elif data.get("action") == "stop":
                    stop = ts
        alert_id = data.get("alert_id")
        if not isinstance(alert_id, str) or not alert_id:
            continue
        kind = data.get("type")
        if (
            topic in ("alert.state", "log.alert_created")
            and isinstance(kind, str)
            and kind in ALERT_TYPES
        ):
            alerts.setdefault(alert_id, kind)
        latency = data.get("latency_ms")
        if type(latency) is not int or latency < 0 or data.get("intent") in (None, "NONE"):
            continue
        if topic == "voice.reply":
            reply_times.setdefault(alert_id, latency)
        elif topic == "alert.state" and data.get("status") == "closed":
            closed_times[alert_id] = latency

    times = closed_times | reply_times
    return {
        "duration_s": round(stop - start)
        if start is not None and stop is not None and stop >= start
        else None,
        "alerts_by_type": dict(sorted(Counter(alerts.values()).items())),
        "response_times_ms": [times[alert_id] for alert_id in alerts if alert_id in times],
    }


def _suggestions(stats: dict[str, Any]) -> list[str]:
    counts = stats["alerts_by_type"]
    ranked = sorted(counts, key=lambda kind: (-counts[kind], kind))
    options = [SUGGESTIONS[kind] for kind in ranked if kind in SUGGESTIONS]
    if not options:
        options.append(
            "Before your next drive, set your route and silence notifications while parked."
        )
    return options


def render_summary(stats: dict[str, Any], suggestions: list[str] | None = None) -> str:
    duration = stats["duration_s"]
    elapsed = "Trip duration unavailable."
    if duration is not None:
        minutes, seconds = divmod(duration, 60)
        elapsed = f"Time driven: {minutes} min {seconds} sec."
    counts = stats["alerts_by_type"]
    breakdown = ", ".join(f"{kind.replace('_', ' ')}: {count}" for kind, count in counts.items())
    alerts = f"Alerts by type: {breakdown}." if breakdown else "No alerts recorded."
    times = stats["response_times_ms"]
    responses = "No response times recorded."
    if len(times) > 3:
        responses = (
            f"Recorded response times: {len(times)} replies, median {median(times):g} ms "
            f"(range {min(times)}–{max(times)} ms)."
        )
    elif times:
        responses = f"Recorded response times: {', '.join(map(str, times))} ms."
    return " ".join([elapsed, alerts, responses, *(suggestions or _suggestions(stats)[:2])])


async def summarize_trip(log: list[dict], timeout_s: float = 10.0) -> str | None:
    try:
        stats = trip_stats(log)
        options = _suggestions(stats)

        def validate_selection(output: str) -> list[str] | None:
            selected = json.loads(output)
            if not isinstance(selected, list) or not 1 <= len(selected) <= 2:
                return None
            if any(type(index) is not int or not 1 <= index <= len(options) for index in selected):
                return None
            if len(set(selected)) != len(selected):
                return None
            return [options[index - 1] for index in selected]

        chosen = await _request(
            "summarize_trip",
            {"stats": stats, "suggestions": options},
            timeout_s,
            10.0,
            validate_selection,
        )
        return render_summary(stats, chosen) if chosen else None
    except (AttributeError, TypeError, ValueError, OverflowError):
        return None
