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
        self.engagement = "quiet"
        self.road_blocked = False
        self._engagement_epoch = 0
        self._engagement_index = 0

    def on_trip(self, data: dict[str, Any]) -> None:
        self._engagement_epoch += 1
        self._engagement_index = 0
        self.manager.cancel_queued(["zoning_out"])
        self.engagement = (
            (data.get("pretrip") or {}).get("engagement", "quiet")
            if data["action"] == "start"
            else "quiet"
        )

    def on_road(self, data: dict[str, Any]) -> None:
        traffic = (data.get("traffic") or "").casefold().replace("_", "-").replace(" ", "-")
        self.road_blocked = data["load"] == "HIGH" or traffic == "stop-and-go"
        if self.road_blocked:
            self._engagement_epoch += 1
            self.manager.cancel_queued(["zoning_out"])

    def _can_engage(self) -> bool:
        return (
            self.engagement != "quiet"
            and not self.road_blocked
            and not any(alert.tier != "T1" for alert in self.manager.pending())
            and not any(kind != "zoning_out" for kind in self._wording)
        )

    async def handle(self, envelope: dict[str, Any]) -> Alert | None:
        if envelope.get("src") != "cv":
            return None
        data = envelope["data"]
        kind = data["type"]
        is_engagement = kind == "zoning_out"
        tier = "T1" if is_engagement else self.routing["tiers"].get(kind)
        if tier is None or (not is_engagement and kind not in TEMPLATES):
            log.info("detector.event %s from cv: no route, skipped", kind)
            return None
        if data["tier"] != tier:
            log.info("detector.event %s: cv says %s, routing uses %s", kind, data["tier"], tier)
        cancels = self.routing["cancels"].get(kind, [])
        if cancels:
            self._engagement_epoch += 1
            self.manager.cancel_queued(cancels)  # DD-04, before any wait
        if is_engagement and not self._can_engage():
            return None
        if kind in self._wording or any(a.type == kind for a in self.manager.pending()):
            return None  # one alert per type at a time
        reason = self.manager.blocked(kind, tier)
        if reason is not None:  # AL-02: logged, not spoken, and no LLM call spent on it
            self.manager.suppressed(kind, tier, reason)
            return None
        self._wording.add(kind)
        try:
            if is_engagement:
                return await self._engage(data)
            ctx = {"event_type": kind, "tier": tier, "signals": data.get("signals", {})}
            wording = await self.writer.write(kind, tier, ctx)
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

    async def _engage(self, data: dict[str, Any]) -> Alert | None:
        try:
            from backend.llm import generate_engagement
            from backend.llm.engagement import engagement_audio_key
        except ImportError:
            log.warning("Engagement dependencies unavailable")
            return None
        epoch = self._engagement_epoch
        item = await generate_engagement(
            {
                "engagement": self.engagement,
                "region": data.get("signals", {}).get("region"),
                "fallback_index": self._engagement_index,
            },
            self.writer.timeout_s,
        )
        if item is None or epoch != self._engagement_epoch or not self._can_engage():
            return None
        text = item["text"]
        alert = self.manager.submit(
            "zoning_out",
            "T1",
            text,
            short_text=text,
            signals={"event_id": data["event_id"], "score": data["score"]},
            lang="en",
            audio_key=engagement_audio_key(text),
            engagement_answer=item["answer"],
        )
        if alert is not None:
            self._engagement_index += 1
        return alert
