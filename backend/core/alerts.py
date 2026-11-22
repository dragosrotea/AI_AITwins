"""Alert Manager v1 (LEAD-T10): one spoken alert at a time, queued by tier [AL-01, VR-05].

Each alert moves queued -> speaking -> listening -> closed, or escalates a tier after missed
replies. The manager is synchronous: it sends every message through `publish(topic, data)`
(BusClient.publish_nowait in core) and reads time from `clock`, so tests drive it with a fake
clock. Core calls `tick()` about 10 times a second for the timeouts.

Protocol with voice (backend/voice/live.py): voice.say -> voice.status speaking -> listening
(the window of listen_s opens after speech and a chime) -> voice.reply (intent NONE when nobody
spoke) -> idle. Voice drops a non-T3 voice.say while busy and a T3 cuts off what is playing.
Fake voice sends no voice.status, only a reply, so a reply is accepted while speaking too.

Timeouts: from voice.say the deadline is speak_max_s + listen_s + reply_grace_s; when voice
reports listening it resets to listen_s + reply_grace_s.

Gap: voice publishes its reply and only then goes idle, and drops a non-T3 voice.say that arrives
while it is busy. So after a reply or timeout the next voice.say (repeat, escalation or next
alert) waits voice_gap_s, run from tick(). A newly submitted T3 skips it: voice always takes T3.

Rate limits (LEAD-T14, AL-02): a new T1 or T2 alert is suppressed inside its type's
cooldown_s, a T1 within t1_min_interval_s of the last T1, and any T1/T2 within
noncritical_min_interval_s of the last T1/T2. T3 is never limited; repeats and escalations are
not new alerts. Suppressed alerts are not spoken but go to on_suppress (the trip log, RR-07).

Replies: OK and ANSWER close; NOT_OK closes and calls on_follow_up; HELP closes and calls
on_emergency. UNCLEAR, NONE or a timeout is a miss: the first max_repeats misses repeat the
short text, the next escalates T1 -> T2 -> T3; a miss at T3 after its repeats calls on_alarm
(and on_emergency after a crash) and starts a continuous alarm (VR-05): voice plays
alarm_audio_key and listens, again and again, until any reply or alarm_rounds rounds.

Busy roads (LEAD-T23, AL-03): while context.road says HIGH load, T1 alerts wait as "held"
(queued T1s are re-published as held, new ones arrive held) and play once the load drops. T2
and T3 are never held, and a T1 already playing finishes.

Countdown alerts (LEAD-T22, CR-02: types in `countdowns`, e.g. crash_candidate: 20 s): the
countdown starts when the alert is first spoken. A miss sounds the alarm again instead of
repeating or escalating; OK (or any non-HELP answer) closes it; HELP or the end of the countdown
publishes it as escalated (the dashboard's emergency banner) and calls on_emergency.
"""

from __future__ import annotations

import itertools
import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("core.alerts")

Publish = Callable[[str, dict[str, Any]], None]
Hook = Callable[["Alert", dict[str, Any] | None], None]

TIER_RANK = {"T1": 1, "T2": 2, "T3": 3}
NEXT_TIER = {"T1": "T2", "T2": "T3"}
MISSES = {"UNCLEAR", "NONE"}

# Last-resort lines; LEAD-T13 swaps these for config/prompts/templates.yaml and the LLM.
REPEAT_TEXT = {
    "en": "I didn't catch that. Are you okay?",
    "ro": "Nu am înțeles. Ești bine?",
}
ALARM_TEXT = {  # shown and logged; voice plays the alarm clip instead of speaking it
    "en": "Alarm. Please answer me. Are you okay?",
    "ro": "Alarmă. Te rog să-mi răspunzi. Ești bine?",
}
HELP_TEXT = {  # VR-06 without X3 (Twilio, LEAD-T22): a spoken confirmation
    "en": (
        "I hear you. I've logged an emergency. If you can, stop somewhere safe and call for help. "
        "Are you safe right now?",
        "Are you safe right now?",
    ),
    "ro": (
        "Te aud. Am înregistrat o urgență. Dacă poți, oprește în siguranță și cere ajutor. "
        "Ești în siguranță acum?",
        "Ești în siguranță acum?",
    ),
}
ESCALATE_TEXT = {
    "en": "Please answer me now. Are you okay?",
    "ro": "Te rog să-mi răspunzi acum. Ești bine?",
}


@dataclass
class Alert:
    type: str
    tier: str
    text: str
    lang: str
    short_text: str | None = None  # used for the repeat; falls back to REPEAT_TEXT
    signals: dict[str, Any] = field(default_factory=dict)
    audio_key: str | None = None  # pre-rendered clip for `text` (first say only)
    alert_id: str = field(default_factory=lambda: "a-" + uuid.uuid4().hex[:8])
    status: str = "queued"
    base_tier: str = ""  # tier at submit; escalation raises `tier` only
    misses: int = 0  # misses at the current tier
    spoken: str = ""  # text of the latest voice.say
    reply: str | None = None
    intent: str | None = None
    latency_ms: int | None = None
    deadline: float = 0.0
    seq: int = 0  # submission order, for FIFO within a tier
    engagement_answer: str | None = None
    alarm_rounds: int = 0  # VR-05: continuous alarm after a T3 miss
    countdown_until: float | None = None  # CR-02: emergency when this passes unanswered
    forced: bool = False  # demo button: not held on a busy road (AL-03)


def _noop(alert: Alert, reply: dict[str, Any] | None) -> None:
    pass


class AlertManager:
    def __init__(
        self,
        cfg: dict[str, Any],
        publish: Publish,
        clock: Callable[[], float] = time.monotonic,
        on_follow_up: Hook = _noop,
        on_emergency: Hook = _noop,
        on_alarm: Hook = _noop,
        on_submit: Callable[[Alert], None] | None = None,
        on_suppress: Callable[[str, str, str], None] | None = None,
        on_answer: Hook = _noop,
    ) -> None:
        self.cfg = cfg
        self.publish = publish
        self.clock = clock
        self.on_follow_up = on_follow_up
        self.on_emergency = on_emergency
        self.on_alarm = on_alarm
        self.on_submit = on_submit  # e.g. the trip log, for the trigger signals (AL-04)
        self.on_suppress = on_suppress  # (type, tier, reason) for alerts held back by AL-02
        self.on_answer = on_answer
        self._last_type: dict[str, float] = {}  # accepted alert times, for the rate limits
        self._last_t1 = -float("inf")
        self._last_noncritical = -float("inf")
        self.lang = cfg["default_lang"]
        self.queue: list[Alert] = []
        self.active: Alert | None = None
        self._seq = itertools.count()
        self.countdowns: dict[str, float] = {}  # type -> seconds (CR-02), set by core
        self.road_high = False  # AL-03: hold T1 prompts while the road load is HIGH
        self._resume: Callable[[], None] | None = None  # next voice.say, after voice_gap_s
        self._resume_at = 0.0

    # Input: detectors and scheduler

    def submit(
        self,
        type: str,
        tier: str,
        text: str,
        short_text: str | None = None,
        signals: dict[str, Any] | None = None,
        lang: str | None = None,
        audio_key: str | None = None,
        bypass_limits: bool = False,
        engagement_answer: str | None = None,
        forced: bool = False,
    ) -> Alert | None:
        """Queue an alert. Returns None if one of the same type is already queued or active,
        or a rate limit holds it back (AL-02). bypass_limits is for answers to the driver
        (a break offered after "I'm tired"), which are not unsolicited prompts."""
        if any(a.type == type for a in self.pending()):
            log.info("drop %s %s: one is already queued or active", tier, type)
            return None
        reason = None if bypass_limits else self.blocked(type, tier)
        if reason is not None:
            self.suppressed(type, tier, reason)
            return None
        now = self.clock()
        self._last_type[type] = now
        if tier != "T3":
            self._last_noncritical = now
        if tier == "T1":
            self._last_t1 = now
        alert = Alert(type, tier, text, lang or self.lang, short_text, signals or {}, audio_key)
        alert.engagement_answer = engagement_answer
        alert.forced = forced
        alert.seq = next(self._seq)
        alert.base_tier = tier
        if self.on_submit is not None:
            self.on_submit(alert)
        self._enqueue(alert)
        if tier == "T3" and self.active is not None and self.active.base_tier != "T3":
            # A genuine T3 (microsleep, crash) also cuts off an alert that only reached T3 by
            # escalating: in the 17:12 demo a microsleep otherwise waited 22 s behind one.
            self._preempt(alert)
        elif tier == "T3" and self.active is None:
            self._resume = None  # no gap for T3
            self._start_next()
        elif self.active is None and self._resume is None and not self._held(alert):
            self._start_next()
        return alert

    def cancel_queued(self, types: list[str]) -> list[Alert]:
        """Drop queued (not active) alerts of these types, published as closed. DD-04 uses it
        so tired drivers get a break, not trivia. The active alert is left to finish: voice
        ignores a non-T3 interruption, and a T3 pre-empts on its own."""
        dropped = [a for a in self.queue if a.type in types]
        for alert in dropped:
            self.queue.remove(alert)
            log.info("cancel queued %s %s", alert.tier, alert.type)
            self._set_status(alert, "closed")
        return dropped

    def blocked(self, type: str, tier: str) -> str | None:
        """Why a new alert of this type and tier would be suppressed now, or None (AL-02).
        Callers check it before spending time on wording."""
        if tier == "T3":
            return None
        now = self.clock()
        cooldown = self.cfg["cooldown_s"].get(type, 0)
        if now - self._last_type.get(type, -float("inf")) < cooldown:
            return f"{type} cooldown {cooldown} s"
        if tier == "T1" and now - self._last_t1 < self.cfg["t1_min_interval_s"]:
            return f"one T1 per {self.cfg['t1_min_interval_s']} s"
        if now - self._last_noncritical < self.cfg["noncritical_min_interval_s"]:
            return f"one non-critical prompt per {self.cfg['noncritical_min_interval_s']} s"
        return None

    def suppressed(self, type: str, tier: str, reason: str) -> None:
        """Record an alert held back by a rate limit: logged, never spoken (RR-07)."""
        log.info("suppress %s %s: %s", tier, type, reason)
        if self.on_suppress is not None:
            self.on_suppress(type, tier, reason)

    def on_road(self, data: dict[str, Any]) -> None:
        """context.road (AL-03): hold queued T1 prompts at HIGH load, release them after."""
        high = data["load"] == "HIGH"
        if high == self.road_high:
            return
        self.road_high = high
        for alert in [a for a in self.queue if a.tier == "T1"]:
            self._set_status(alert, "held" if high else "queued")
        log.info("road load %s: T1 prompts %s", data["load"], "held" if high else "released")
        if not high and self.active is None and self._resume is None:
            self._start_next()

    # Input: voice

    def _is_current(self, data: dict[str, Any]) -> bool:
        """True for messages about the alert being spoken now (not while waiting in the gap)."""
        alert = self.active
        return alert is not None and self._resume is None and data.get("alert_id") == alert.alert_id

    def on_voice_status(self, data: dict[str, Any]) -> None:
        alert = self.active
        if alert is None or not self._is_current(data):
            return
        if data.get("state") == "listening" and alert.status == "speaking":
            alert.deadline = self.clock() + self.cfg["listen_s"] + self.cfg["reply_grace_s"]
            self._set_status(alert, "listening")

    def on_voice_reply(self, data: dict[str, Any]) -> None:
        alert = self.active
        if alert is None or not self._is_current(data):
            return
        alert.reply = data.get("transcript") or None
        alert.intent = data.get("intent")
        alert.latency_ms = data.get("latency_ms")
        if alert.intent in MISSES:
            self._miss(alert)
            return
        if alert.countdown_until is not None and alert.intent == "HELP":
            self._emergency(alert, data)  # CR-02: "help" continues to the emergency
            return
        self._close(alert)
        if alert.intent == "NOT_OK":
            self.on_follow_up(alert, data)
        elif alert.intent == "HELP":
            self.on_emergency(alert, data)
        elif alert.intent == "ANSWER" and alert.engagement_answer is not None:
            self.on_answer(alert, data)

    def tick(self) -> None:
        active = self.active
        if (
            active is not None
            and active.countdown_until is not None
            and self.clock() >= active.countdown_until
        ):
            self._emergency(active, None)  # CR-02: silence until the countdown ends
            return
        if self._resume is not None:
            if self.clock() >= self._resume_at:
                resume, self._resume = self._resume, None
                resume()
            return
        alert = self.active
        if alert is not None and self.clock() >= alert.deadline:
            alert.reply, alert.intent, alert.latency_ms = None, "NONE", None
            self._miss(alert)

    # Internals

    def pending(self) -> list[Alert]:
        """Queued alerts plus the active one."""
        return self.queue + ([self.active] if self.active else [])

    def _enqueue(self, alert: Alert) -> None:
        alert.status = "held" if self._held(alert) else "queued"
        self.queue.append(alert)
        self.queue.sort(key=lambda a: (-TIER_RANK[a.tier], a.seq))
        self._publish_state(alert)

    def _held(self, alert: Alert) -> bool:
        return self.road_high and alert.tier == "T1" and not alert.forced

    def _start_next(self) -> None:
        ready = [a for a in self.queue if not self._held(a)]  # the queue is sorted by tier
        if not ready:
            self.publish("control.music", {"duck": False, "level": 1.0})
            return
        self.queue.remove(ready[0])
        self._speak(ready[0], first=True)

    def _later(self, action: Callable[[], None]) -> None:
        self._resume = action
        self._resume_at = self.clock() + self.cfg["voice_gap_s"]
        if self.active is not None:
            self.active.deadline = float("inf")  # no timeout while waiting to speak

    def _preempt(self, urgent: Alert) -> None:
        interrupted = self.active
        assert interrupted is not None
        log.info("T3 pre-empts %s %s", interrupted.tier, interrupted.type)
        self._resume = None  # a pending repeat of the interrupted alert is dropped
        self.active = None
        if interrupted.type == "zoning_out":
            self._set_status(interrupted, "closed")
        else:
            self._enqueue(interrupted)  # back in line; it starts again after the T3
        self.queue.remove(urgent)  # speak this one, even if the interrupted one sorts first
        self._speak(urgent, first=True)

    def _speak(self, alert: Alert, first: bool) -> None:
        self.active = alert
        if first:
            alert.spoken = alert.text
        if alert.countdown_until is None and alert.type in self.countdowns:
            alert.countdown_until = self.clock() + self.countdowns[alert.type]
        if alert.tier == "T3":
            level = self.cfg["t3_music_level"]
        else:  # per type first: road rage only lowers the music by 30% (RR-06)
            level = self.cfg["duck_level"].get(alert.type, self.cfg["music_duck_volume"] / 100)
        self.publish("control.music", {"duck": True, "level": level})
        alert.deadline = (
            self.clock()
            + self.cfg["speak_max_s"]
            + self.cfg["listen_s"]
            + self.cfg["reply_grace_s"]
        )
        self._set_status(alert, "speaking")
        say = {
            "alert_id": alert.alert_id,
            "text": alert.spoken,
            "tier": alert.tier,
            "lang": alert.lang,
            "listen_s": self.cfg["listen_s"],
        }
        if alert.spoken == ALARM_TEXT[alert.lang]:
            say["audio_key"] = self.cfg["alarm_audio_key"]  # VR-05: the alarm clip, not speech
        elif alert.audio_key and alert.spoken == alert.text:
            say["audio_key"] = alert.audio_key  # the clip matches only the original text
        self.publish("voice.say", say)

    def _miss(self, alert: Alert) -> None:
        if alert.countdown_until is not None:  # CR-02: keep the alarm going until it ends
            alert.spoken = ALARM_TEXT[alert.lang]
            self._later(lambda: self._speak(alert, first=False))
            return
        alert.misses += 1
        if alert.misses <= self.cfg["max_repeats"]:
            log.info("%s %s: miss %d, repeating", alert.tier, alert.type, alert.misses)
            alert.spoken = alert.short_text or REPEAT_TEXT[alert.lang]
            self._later(lambda: self._speak(alert, first=False))
            return
        next_tier = NEXT_TIER.get(alert.tier)
        self._set_status(alert, "escalated")
        if next_tier is None:  # missed at T3: continuous alarm, emergency flow after a crash
            if alert.alarm_rounds == 0:
                log.warning("T3 %s missed: alarm", alert.type)
                self.on_alarm(alert, None)
                if alert.type == "crash_candidate":
                    self.on_emergency(alert, None)
            if alert.alarm_rounds >= self.cfg["alarm_rounds"]:
                log.warning("T3 %s: no answer after %d alarms", alert.type, alert.alarm_rounds)
                self.active = None
                self._later(self._start_next)
                return
            alert.alarm_rounds += 1
            alert.misses = self.cfg["max_repeats"]  # every further miss sounds the alarm again
            alert.spoken = ALARM_TEXT[alert.lang]
            self._later(lambda: self._speak(alert, first=False))
            return
        log.info("%s %s escalates to %s", alert.tier, alert.type, next_tier)
        alert.tier, alert.misses = next_tier, 0
        alert.spoken = ESCALATE_TEXT[alert.lang]
        self._later(lambda: self._speak(alert, first=False))

    def _emergency(self, alert: Alert, reply: dict[str, Any] | None) -> None:
        log.warning("%s %s: emergency (%s)", alert.tier, alert.type, "help" if reply else "silence")
        self._resume = None
        self._set_status(alert, "escalated")
        self.active = None
        self.on_emergency(alert, reply)
        self._later(self._start_next)

    def _close(self, alert: Alert) -> None:
        self._set_status(alert, "closed")
        self.active = None
        self._later(self._start_next)

    def _set_status(self, alert: Alert, status: str) -> None:
        alert.status = status
        self._publish_state(alert)

    def _publish_state(self, alert: Alert) -> None:
        self.publish(
            "alert.state",
            {
                "alert_id": alert.alert_id,
                "type": alert.type,
                "tier": alert.tier,
                "status": alert.status,
                "text": alert.spoken or alert.text,
                "reply": alert.reply,
                "intent": alert.intent,
                "latency_ms": alert.latency_ms,
            },
        )
