"""Detector events to alerts (LEAD-T11: the first hard-coded alert, the CP1 integration test).

A CV detector.event of type `distraction` becomes a T2 alert with a fixed template in the
driver's language. Every other type is logged and skipped until LEAD-T15 routes them all, and
core's own fusion events (src "core") are ignored so they can never loop back.

Templates follow VR-01 and LLM-01..05: short, end in a question, never ask the driver to look at
or touch anything, never blame. LEAD-T13 replaces them with the LLM and
config/prompts/templates.yaml.
"""

from __future__ import annotations

import logging
from typing import Any

from backend.core.alerts import Alert, AlertManager

log = logging.getLogger("core.events")

# type -> tier -> lang -> (text, short text for the repeat)
TEMPLATES: dict[str, dict[str, dict[str, tuple[str, str]]]] = {
    "distraction": {
        "T2": {
            "en": (
                "Your attention drifted from the road for a moment. Are you with me?",
                "Are you with me?",
            ),
            "ro": (
                "Atenția ți-a fugit de la drum pentru o clipă. Ești cu mine?",
                "Ești cu mine?",
            ),
        },
    },
}


def handle_detector_event(envelope: dict[str, Any], manager: AlertManager) -> Alert | None:
    """Turn a validated detector.event envelope into an alert, or None if it is skipped."""
    if envelope.get("src") != "cv":
        return None
    data = envelope["data"]
    kind = data["type"]
    tier = "T2"  # LEAD-T11: fixed; LEAD-T15 maps each type to its tier
    templates = TEMPLATES.get(kind, {}).get(tier)
    if templates is None:
        log.info("detector.event %s from cv: not routed yet (LEAD-T15)", kind)
        return None
    text, short = templates.get(manager.lang) or templates["en"]
    signals = {"event_id": data["event_id"], "score": data["score"], **data.get("signals", {})}
    alert = manager.submit(kind, tier, text, short_text=short, signals=signals)
    if alert is not None:
        log.info(
            "detector.event %s (score %.0f) -> %s alert %s",
            kind,
            data["score"],
            tier,
            alert.alert_id,
        )
    return alert
