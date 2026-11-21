"""Check-in scheduler (LEAD-T20): personalised check-ins and break reminders [CI-02..CI-05].

On control.trip start the interval comes from the pre-trip form (CI-02): base_interval_min,
halved when sleep is under short_sleep_h, halved when tiredness is high_tiredness or more,
times circadian_factor inside the circadian dips, clamped to min/max. Breaks come every
break_every_min, or break_risk_min when any CI-02 risk factor applies (CI-03).

Live adaptation (CI-04): each T2/T3 alert in the last event_lookback_min shortens the next
check-in by event_shorten; once the trip has run quiet_min, quiet_min without such alerts
lengthens it by quiet_lengthen. Every change is published on checkin.schedule.

A due check-in is spoken as a T1 checkin_due alert (LLM first, template fallback, CI-05). If the
AL-02 rate limits hold it back it tries again after retry_s, so it is never lost. A NOT_OK reply
to any alert shortens the interval like an event and offers a break at once; that offer answers
the driver, so it skips the rate limits.

Times are Unix seconds from `clock` (checkin.schedule carries Unix timestamps); time of day
uses the local clock.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

from backend.core.alerts import Alert, AlertManager
from backend.core.wording import AlertWriter

log = logging.getLogger("core.checkin")

Publish = Callable[[str, dict[str, Any]], None]


def _minutes_of_day(ts: float) -> int:
    t = time.localtime(ts)
    return t.tm_hour * 60 + t.tm_min


def _in_dips(ts: float, dips: list[str]) -> bool:
    now = _minutes_of_day(ts)
    for dip in dips:
        start, end = (int(h) * 60 + int(m) for h, m in (part.split(":") for part in dip.split("-")))
        if start <= now < end:
            return True
    return False


def base_interval(
    pretrip: dict[str, Any], now_ts: float, cfg: dict[str, Any]
) -> tuple[float, bool]:
    """CI-02: the check-in interval in minutes, and whether any risk factor applies."""
    minutes = float(cfg["base_interval_min"])
    short_sleep = pretrip["sleep_h"] < cfg["short_sleep_h"]
    tired = pretrip["tiredness"] >= cfg["high_tiredness"]
    dip = _in_dips(now_ts, cfg["circadian_dips"])
    if short_sleep:
        minutes /= 2
    if tired:
        minutes /= 2
    if dip:
        minutes *= cfg["circadian_factor"]
    minutes = min(max(minutes, cfg["min_interval_min"]), cfg["max_interval_min"])
    return minutes, short_sleep or tired or dip


class CheckinScheduler:
    def __init__(
        self,
        cfg: dict[str, Any],
        manager: AlertManager,
        writer: AlertWriter,
        builtin: dict[str, dict[str, tuple[str, str]]],
        publish: Publish,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.cfg = cfg
        self.manager = manager
        self.writer = writer
        self.builtin = builtin
        self.publish = publish
        self.clock = clock
        self.pretrip: dict[str, Any] | None = None
        self.trip_start = 0.0
        self.anchor = 0.0  # last check-in (or trip start); the next one is anchor + interval
        self.interval_min = 0.0
        self.next_checkin = 0.0
        self.next_break = 0.0
        self.retry_at: dict[str, float] = {}  # per prompt type, after an AL-02 hold
        self.offer_break = False
        self._published: tuple[float, float, float] | None = None
        self.events: list[float] = []  # times of T2/T3 alerts (CI-04)

    @property
    def running(self) -> bool:
        return self.pretrip is not None

    # Inputs

    def on_trip(self, data: dict[str, Any]) -> None:
        if data["action"] == "stop":
            self.pretrip = None
            log.info("trip stopped: check-ins off")
            return
        if not data.get("pretrip"):
            return
        now = self.clock()
        self.pretrip = data["pretrip"]
        self.trip_start = self.anchor = now
        self.events.clear()
        self.offer_break = False
        _, risk = base_interval(self.pretrip, now, self.cfg)
        break_min = self.cfg["break_risk_min"] if risk else self.cfg["break_every_min"]
        self.next_break = now + break_min * 60
        self._reschedule()

    def on_alert(self, alert: Alert) -> None:
        """Every accepted alert; T2/T3 ones (not our own prompts) adapt the interval (CI-04)."""
        if not self.running or alert.type in ("checkin_due", "break_due"):
            return
        if alert.tier in ("T2", "T3"):
            self.events.append(self.clock())
            self._reschedule()

    def on_not_ok(self, alert: Alert, reply: dict[str, Any] | None) -> None:
        """A tired reply shortens the interval and offers a break now (CI-05, T20)."""
        if not self.running:
            return
        self.events.append(self.clock())
        self.offer_break = alert.type != "break_due"  # don't answer a break offer with another
        self._reschedule()

    # Schedule

    def adapted_interval(self, now: float) -> float:
        assert self.pretrip is not None
        cfg = self.cfg
        minutes, _ = base_interval(self.pretrip, now, cfg)
        recent = [t for t in self.events if now - t <= cfg["event_lookback_min"] * 60]
        minutes *= (1 - cfg["event_shorten"]) ** len(recent)
        quiet_for = now - max(self.events, default=self.trip_start)
        if now - self.trip_start >= cfg["quiet_min"] * 60 and quiet_for >= cfg["quiet_min"] * 60:
            minutes *= 1 + cfg["quiet_lengthen"]
        return min(max(minutes, cfg["min_interval_min"]), cfg["max_interval_min"])

    def _reschedule(self) -> None:
        now = self.clock()
        interval = round(self.adapted_interval(now), 2)
        next_checkin = max(self.anchor + interval * 60, now)
        self.interval_min, self.next_checkin = interval, next_checkin
        state = (interval, round(next_checkin, 1), round(self.next_break, 1))
        if state != self._published:
            self._published = state
            self.publish(
                "checkin.schedule",
                {
                    "interval_min": interval,
                    "next_checkin_ts": round(next_checkin, 1),
                    "next_break_ts": round(self.next_break, 1),
                },
            )

    # Called about once a second by core

    async def tick(self) -> None:
        if not self.running:
            return
        now = self.clock()
        if self.offer_break:
            self.offer_break = False
            await self._prompt("break_due", bypass=True)
        if self._due("break_due", self.next_break, now) and await self._prompt("break_due"):
            _, risk = base_interval(self.pretrip or {}, now, self.cfg)
            break_min = self.cfg["break_risk_min"] if risk else self.cfg["break_every_min"]
            self.next_break = now + break_min * 60
            self._reschedule()  # publishes the new break time
        if self._due("checkin_due", self.next_checkin, now) and await self._prompt("checkin_due"):
            self.anchor = now
            self._reschedule()

    def _due(self, kind: str, at: float, now: float) -> bool:
        return now >= at and now >= self.retry_at.get(kind, 0.0)

    async def _prompt(self, kind: str, bypass: bool = False) -> bool:
        """Speak a T1 prompt. False if it must wait (rate limit, or one already pending)."""
        if any(a.type == kind for a in self.manager.pending()):
            return False
        if not bypass:
            reason = self.manager.blocked(kind, "T1")
            if reason is not None:
                self.retry_at[kind] = self.clock() + self.cfg["retry_s"]
                log.info("%s waits %d s: %s", kind, self.cfg["retry_s"], reason)
                return False
        wording = await self.writer.write(kind, "T1", {"event_type": kind, "tier": "T1"})
        _text, short = self.builtin[kind]["en"]
        alert = self.manager.submit(
            kind,
            "T1",
            wording.text,
            short_text=short,
            signals={"text_path": wording.path, "llm_ms": wording.llm_ms},
            lang=wording.lang,
            audio_key=wording.audio_key,
            bypass_limits=bypass,
        )
        return alert is not None
