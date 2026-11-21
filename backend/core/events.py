"""Route CV visual events, Core fusion events, and eligible trivia alerts.

A CV detector.event becomes an alert at the tier from config core.routing.tiers: core decides
the tier, whatever tier the event carries (a mismatch is logged). Drowsiness and microsleep
first cancel queued trivia (core.routing.cancels, DD-04). Core events use fusion_tiers.
zoning_out (from core's zoning fusion, LEAD-T19, or CV) uses the T1 engagement route: trivia or
a fun fact per the pre-trip preference (nothing for "quiet"), never in HIGH road load (ZO-06),
and a break recommendation instead after max_engagements in engagement_window_s (ZO-05). A
graded trivia answer is answered aloud (ZO-04).

The spoken text comes from AlertWriter (LEAD-T13): the LLM for T1/T2, else the voice engineer's
catalog. TEMPLATES below are the last-resort lines and supply the short repeat text (VR-05).
They follow VR-01 and the LLM rules: short, end in a question, never ask the driver to look at
or touch anything, never blame, never encourage driving on when tired.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from collections.abc import Callable
from typing import Any

from backend.core.alerts import Alert, AlertManager
from backend.core.wording import AlertWriter
from backend.core.zoning import road_blocked

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
    "crash_candidate": {  # CR-02, LEAD-T22: from core's crash detector
        "en": ("The car stopped suddenly. Are you okay?", "Are you okay?"),
        "ro": ("Mașina s-a oprit brusc. Ești bine?", "Ești bine?"),
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

    def __init__(
        self,
        manager: AlertManager,
        routing: dict[str, Any],
        writer: AlertWriter,
        zoning: dict[str, Any] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.manager = manager
        self.routing = routing
        self.writer = writer
        self.zoning = zoning or {}
        self.clock = clock
        self._engagements: deque[float] = deque()  # ZO-05: times of spoken engagements
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
        self.road_blocked = road_blocked(data)
        if self.road_blocked:
            self._engagement_epoch += 1
            self.manager.cancel_queued(["zoning_out"])

    def _can_engage(self) -> bool:
        return (
            self.engagement != "quiet"  # the pre-trip preference (CI-01): no form, no trivia
            and not self.road_blocked
            and not any(alert.tier != "T1" for alert in self.manager.pending())
            and not any(kind != "zoning_out" for kind in self._wording)
        )

    async def handle(self, envelope: dict[str, Any]) -> Alert | None:
        data = envelope["data"]
        kind = data["type"]
        # CV's visual events, and core's own fusion events (road_rage, LEAD-T18); core never
        # routes anything else of its own, so no event can loop.
        source = envelope.get("src")
        sources = {"cv": self.routing["tiers"], "core": self.routing["fusion_tiers"]}
        is_engagement = kind == "zoning_out" and source in ("cv", "core")
        tier = "T1" if is_engagement else sources.get(source, {}).get(kind)
        if tier is None or (not is_engagement and kind not in TEMPLATES):
            if source == "cv":
                log.info("detector.event %s from cv: no route, skipped", kind)
            return None
        if data["tier"] != tier:
            log.info(
                "detector.event %s: %s says %s, routing uses %s", kind, source, data["tier"], tier
            )
        cancels = self.routing["cancels"].get(kind, [])
        if cancels:
            self._engagement_epoch += 1
            self.manager.cancel_queued(cancels)  # DD-04, before any wait
        if is_engagement and not self._can_engage():
            return None
        if is_engagement and self._too_many_engagements():
            # ZO-05: persistent zoning out means fatigue: recommend a break, not more trivia.
            log.info(
                "zoning_out: %d engagements in the window, recommending a break",
                len(self._engagements),
            )
            data = {**data, "type": "break_due"}
            kind, is_engagement = "break_due", False
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
        if alert is not None and kind == "zoning_out":
            self._engagements.append(self.clock())
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
            self._engagements.append(self.clock())
        return alert

    def _too_many_engagements(self) -> bool:
        window = self.zoning.get("engagement_window_s")
        limit = self.zoning.get("max_engagements")
        if window is None or limit is None:
            return False
        now = self.clock()
        while self._engagements and now - self._engagements[0] > window:
            self._engagements.popleft()
        return len(self._engagements) >= limit

    async def answer_feedback(self, alert: Alert, reply: dict[str, Any]) -> Alert | None:
        """ZO-04: grade a trivia answer leniently and say the result in one short sentence."""
        try:
            from backend.llm import grade_answer
        except ImportError:
            log.warning("grading unavailable")
            return None
        assert alert.engagement_answer is not None
        result = await grade_answer(alert.text, alert.engagement_answer, reply["transcript"])
        if result is None:
            return None
        feedback = result["feedback"].rstrip(". ")
        if not result["correct"]:
            feedback = f"{feedback}, it's {alert.engagement_answer}"
        question = self.zoning.get("feedback_question", "How are you feeling now?")
        text = f"{feedback}. {question}"
        # An answer to the driver: not limited (AL-02), and voice hears the reply as ANSWER,
        # which closes it without another grade (no engagement_answer).
        return self.manager.submit(
            "zoning_out",
            "T1",
            text,
            short_text=question,
            signals={"graded": alert.alert_id, "correct": result["correct"]},
            lang="en",
            bypass_limits=True,
        )
