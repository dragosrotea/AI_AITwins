"""Detector events to alerts: CV visual event routing (LEAD-T11, LEAD-T15).

A CV detector.event becomes an alert at the tier from config core.routing.tiers: core decides
the tier, whatever tier the event carries (a mismatch is logged). Drowsiness and microsleep
first cancel queued trivia (core.routing.cancels, DD-04). Types without a tier or template are
logged and skipped, and core's own fusion events (src "core") are ignored so they never loop.

Templates follow VR-01 and the LLM rules: short, end in a question, never ask the driver to
look at or touch anything, never blame, never encourage driving on when tired. LEAD-T13 puts
the LLM first and config/prompts/templates.yaml as the fallback.
"""

from __future__ import annotations

import logging
from typing import Any

from backend.core.alerts import Alert, AlertManager

log = logging.getLogger("core.events")

# type -> lang -> (text, short text for the repeat)
TEMPLATES: dict[str, dict[str, tuple[str, str]]] = {
    "distraction": {  # DD-01
        "en": (
            "Your attention drifted from the road for a moment. Are you with me?",
            "Are you with me?",
        ),
        "ro": (
            "Atenția ți-a fugit de la drum pentru o clipă. Ești cu mine?",
            "Ești cu mine?",
        ),
    },
    "microsleep": {  # DD-02, scenario 3: confirm awake, recommend a stop
        "en": (
            "Your eyes closed for a moment. Please plan a stop at the next rest area. "
            "Are you awake?",
            "Are you awake?",
        ),
        "ro": (
            "Ți s-au închis ochii o clipă. Te rog să oprești la următoarea parcare. Ești treaz?",
            "Ești treaz?",
        ),
    },
    "drowsiness": {  # DD-03: with a break suggestion
        "en": (
            "You seem tired. A short break would help. Shall we plan one soon?",
            "Shall we plan a break?",
        ),
        "ro": (
            "Pari obosit. O pauză scurtă te-ar ajuta. Facem una în curând?",
            "Facem o pauză?",
        ),
    },
    "phone": {  # PH-02
        "en": ("The phone can wait until we stop. Can you leave it for now?", "Can it wait?"),
        "ro": ("Telefonul poate aștepta până oprim. Îl lași deocamdată?", "Poate aștepta?"),
    },
    "no_face": {  # CV-06
        "en": ("I can't see you at the moment. Is everything okay?", "Are you okay?"),
        "ro": ("Nu te mai văd în acest moment. Este totul în regulă?", "Ești bine?"),
    },
}


def handle_detector_event(
    envelope: dict[str, Any], manager: AlertManager, routing: dict[str, Any]
) -> Alert | None:
    """Turn a validated detector.event envelope into an alert, or None if it is skipped."""
    if envelope.get("src") != "cv":
        return None
    data = envelope["data"]
    kind = data["type"]
    tier = routing["tiers"].get(kind)
    templates = TEMPLATES.get(kind)
    if tier is None or templates is None:
        log.info("detector.event %s from cv: no route, skipped", kind)
        return None
    if data["tier"] != tier:
        log.info("detector.event %s: cv says %s, routing uses %s", kind, data["tier"], tier)
    cancels = routing["cancels"].get(kind, [])
    if cancels:
        manager.cancel_queued(cancels)
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
