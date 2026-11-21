"""Detector events to alerts: CV visual event routing (LEAD-T11, LEAD-T15).

A CV detector.event becomes an alert at the tier from config core.routing.tiers: core decides
the tier, whatever tier the event carries (a mismatch is logged). Drowsiness and microsleep
first cancel queued trivia (core.routing.cancels, DD-04). Types without a tier or template are
logged and skipped, and core's own fusion events (src "core") are ignored so they never loop.

The spoken text comes from AlertWriter (LEAD-T13): the LLM for T1/T2, else the voice engineer's
catalog. TEMPLATES below are the last-resort lines and supply the short repeat text (VR-05).
They follow VR-01 and the LLM rules: short, end in a question, never ask the driver to look at
or touch anything, never blame, never encourage driving on when tired.
"""

from __future__ import annotations

import logging
from typing import Any

from backend.core.alerts import Alert, AlertManager
from backend.core.wording import AlertWriter

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
    "road_rage": {  # RR-06: calm, non-judgmental, at most 20 words (from core's fusion, T17)
        "en": ("That sounded stressful. Shall we take a slow breath together?", "Are you okay?"),
        "ro": ("A fost un moment tensionat. Respirăm adânc împreună?", "Ești bine?"),
    },
    "checkin_due": {  # CI-05, LEAD-T20
        "en": ("Quick check-in. How are you feeling right now?", "How are you feeling?"),
        "ro": ("Verificare rapidă. Cum te simți acum?", "Cum te simți?"),
    },
    "break_due": {  # CI-03, LEAD-T20
        "en": ("Time for a short break. Can you stop somewhere safe soon?", "Can we stop soon?"),
        "ro": ("E timpul pentru o pauză scurtă. Poți opri undeva în siguranță?", "Oprim curând?"),
    },
    "no_face": {  # CV-06
        "en": ("I can't see you at the moment. Is everything okay?", "Are you okay?"),
        "ro": ("Nu te mai văd în acest moment. Este totul în regulă?", "Ești bine?"),
    },
}


class EventRouter:
    """Routes validated CV detector.event envelopes to the Alert Manager (LEAD-T11, T15, T13)."""

    def __init__(self, manager: AlertManager, routing: dict[str, Any], writer: AlertWriter) -> None:
        self.manager = manager
        self.routing = routing
        self.writer = writer
        self._wording: set[str] = set()  # types waiting on the LLM, so repeats are dropped

    async def handle(self, envelope: dict[str, Any]) -> Alert | None:
        data = envelope["data"]
        kind = data["type"]
        # CV's visual events, and core's own fusion events (road_rage, LEAD-T18); core never
        # routes anything else of its own, so no event can loop.
        sources = {"cv": self.routing["tiers"], "core": self.routing["fusion_tiers"]}
        tier = sources.get(envelope.get("src"), {}).get(kind)
        if tier is None or kind not in TEMPLATES:
            if envelope.get("src") == "cv":
                log.info("detector.event %s from cv: no route, skipped", kind)
            return None
        if data["tier"] != tier:
            log.info("detector.event %s: cv says %s, routing uses %s", kind, data["tier"], tier)
        cancels = self.routing["cancels"].get(kind, [])
        if cancels:
            self.manager.cancel_queued(cancels)  # DD-04, before any wait
        if kind in self._wording or any(a.type == kind for a in self.manager.pending()):
            return None  # one alert per type at a time
        reason = self.manager.blocked(kind, tier)
        if reason is not None:  # AL-02: logged, not spoken, and no LLM call spent on it
            self.manager.suppressed(kind, tier, reason)
            return None
        self._wording.add(kind)
        try:
            signals_in = data.get("signals", {})
            firm = bool(signals_in.get("firm"))  # RR-05: score >= 85
            ctx = {"event_type": kind, "tier": tier, "signals": signals_in, "firm": firm}
            wording = await self.writer.write(
                kind,
                tier,
                ctx,
                max_words=self.routing["max_words"].get(kind),
                variant=self.routing["firm_variant"].get(kind) if firm else None,
            )
        finally:
            self._wording.discard(kind)
        _text, short = TEMPLATES[kind]["en"]
        signals = {
            "event_id": data["event_id"],
            "score": data["score"],
            **data.get("signals", {}),
            "text_path": wording.path,  # AL-05: llm, template or builtin
            "llm_ms": wording.llm_ms,
        }
        alert = self.manager.submit(
            kind,
            tier,
            wording.text,
            short_text=short,
            signals=signals,
            lang=wording.lang,
            audio_key=wording.audio_key,
        )
        if alert is not None:
            log.info(
                "detector.event %s (score %.0f) -> %s alert %s, text from %s",
                kind,
                data["score"],
                tier,
                alert.alert_id,
                wording.path,
            )
        return alert
