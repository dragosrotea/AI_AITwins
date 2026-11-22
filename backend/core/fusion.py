"""Road-rage fusion (LEAD-T17): Rage Score 0-100 from face, voice and driving [RR-04, RR-05, RR-07].

Modalities, each 0-1, weighted by core.rage.weights:
- face (RR-01): driver.state anger_p while a face is present; it *agrees* once anger_p has
  stayed at or above face_threshold for face_hold_s.
- voice (RR-02, RR-03): audio.features shouting, or swear_count_60s scaled by swears_full; it
  agrees for voice_window_s after a shout or a counted swear.
- driving (RR-04): harsh acceleration or braking (|g| above harsh_g) in the last harsh_window_s,
  from core's own vehicle.motion; it agrees while one is in the window.
Bonuses: heart rate more than stress_delta_bpm above resting for stress_hold_s without a harsh
event (SW-03), and music turned up by music_rise within music_window_s.

Trigger (RR-05): score >= trigger_score with at least min_modalities agreeing, or the face alone
angry for face_alone_s. Score >= firm_score asks for a firmer message. Swearing or shouting
alone never triggers: voice is one modality and its weight is below trigger_score.

Limits (RR-07): cooldown_s between interventions and at most max_per_window per max_window_s.
An episode triggers once, when the condition starts; inside the limits it is reported as
suppressed (logged, not spoken). The manager-side road_rage cooldown (T14) stays as a backstop.
"""

from __future__ import annotations

import itertools
import logging
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("core.fusion")

NEVER = -float("inf")


@dataclass
class RageReading:
    score: float
    face: float
    voice: float
    driving: float
    agreeing: list[str] = field(default_factory=list)
    face_alone: bool = False
    hr_stress: bool = False
    music_rise: bool = False
    triggers: bool = False


class RageFusion:
    def __init__(
        self,
        rage: dict[str, Any],
        hr: dict[str, Any],
        emit: Callable[[dict[str, Any]], None],
        on_suppress: Callable[[dict[str, Any], str], None] | None = None,
    ) -> None:
        self.cfg = rage
        self.hr_cfg = hr
        self.emit = emit  # receives detector.event data
        self.on_suppress = on_suppress
        self.resting_hr = float(rage["resting_hr"])
        self._ids = itertools.count(1)
        # Latest inputs and the times that matter
        self._anger = 0.0
        self._angry_since: float | None = None
        self._swears = 0
        self._shouting = False
        self._voice_until = NEVER
        self._harsh: deque[tuple[float, float]] = deque()  # (time, |g|)
        self._hr_high_since: float | None = None
        self._music: deque[tuple[float, float]] = deque()  # (time, volume)
        # Episodes and limits
        self._in_episode = False
        self._fired: deque[float] = deque()

    # Inputs (data dicts as on the bus, with the time they arrived)

    def on_driver_state(self, data: dict[str, Any], now: float) -> None:
        self._anger = float(data["anger_p"]) if data["face_present"] else 0.0
        if self._anger >= self.cfg["face_threshold"]:
            if self._angry_since is None:
                self._angry_since = now
        else:
            self._angry_since = None
        self.evaluate(now)

    def on_audio(self, data: dict[str, Any], now: float) -> None:
        swears = int(data["swear_count_60s"])
        if data["shouting"] or swears > self._swears:
            self._voice_until = now + self.cfg["voice_window_s"]
        self._swears, self._shouting = swears, bool(data["shouting"])
        self.evaluate(now)

    def on_vehicle(self, data: dict[str, Any], now: float) -> None:
        peak = max(abs(data["accel_long_g"]), abs(data["accel_lat_g"]))
        if peak > self.cfg["harsh_g"]:
            self._harsh.append((now, peak))
        self.evaluate(now)

    def on_hr(self, data: dict[str, Any], now: float) -> None:
        if data["bpm"] > self.resting_hr + self.hr_cfg["stress_delta_bpm"]:
            if self._hr_high_since is None:
                self._hr_high_since = now
        else:
            self._hr_high_since = None

    def on_music(self, data: dict[str, Any], now: float) -> None:
        self._music.append((now, float(data["volume"])))

    def on_trip(self, data: dict[str, Any]) -> None:
        if data["action"] == "start" and data.get("pretrip"):
            self.resting_hr = float(data["pretrip"]["resting_hr"])

    # Score

    def reading(self, now: float) -> RageReading:
        cfg = self.cfg
        while self._harsh and now - self._harsh[0][0] > cfg["harsh_window_s"]:
            self._harsh.popleft()
        while self._music and now - self._music[0][0] > cfg["music_window_s"]:
            self._music.popleft()

        # A modality that agrees counts in full, so any two reach the trigger (voice + driving
        # = 60 with the default weights) and one alone never does (at most 40).
        face_held = self._angry_since is not None and now - self._angry_since >= cfg["face_hold_s"]
        voice_agrees = self._shouting or now < self._voice_until
        driving_agrees = bool(self._harsh)
        face = 1.0 if face_held else self._anger
        voice = 1.0 if voice_agrees else min(1.0, self._swears / cfg["swears_full"])
        driving = 1.0 if driving_agrees else 0.0
        agreeing = [
            name
            for name, agrees in (
                ("face", face_held),
                ("voice", voice_agrees),
                ("driving", driving_agrees),
            )
            if agrees
        ]
        hr_stress = (
            self._hr_high_since is not None
            and now - self._hr_high_since >= self.hr_cfg["stress_hold_s"]
            and not any(now - t < 10 for t, _ in self._harsh)  # SW-03: not braking hard
        )
        volumes = [v for _, v in self._music]
        music_rise = len(volumes) > 1 and volumes[-1] - min(volumes) >= cfg["music_rise"]

        w = cfg["weights"]
        score = 100 * (w["face"] * face + w["voice"] * voice + w["driving"] * driving)
        score += cfg["hr_bonus"] * hr_stress + cfg["music_bonus"] * music_rise
        score = round(min(100.0, score), 1)
        face_alone = (
            self._angry_since is not None and now - self._angry_since >= cfg["face_alone_s"]
        )
        two_agree = score >= cfg["trigger_score"] and len(agreeing) >= cfg["min_modalities"]
        return RageReading(
            score,
            round(face, 2),
            round(voice, 2),
            round(driving, 2),
            agreeing,
            face_alone=face_alone,
            hr_stress=hr_stress,
            music_rise=music_rise,
            triggers=two_agree or face_alone,
        )

    def snapshot(self, now: float) -> dict[str, Any]:
        """rage.score data: the live reading for the dashboard."""
        r = self.reading(now)
        return {
            "score": r.score,
            "face": r.face,
            "voice": r.voice,
            "driving": r.driving,
            "agreeing": r.agreeing,
            "hr_stress": r.hr_stress,
            "music_rise": r.music_rise,
            "trigger_score": float(self.cfg["trigger_score"]),
        }

    def evaluate(self, now: float) -> RageReading:
        reading = self.reading(now)
        if not reading.triggers:
            self._in_episode = False
            return reading
        if self._in_episode:
            return reading  # one intervention per episode
        self._in_episode = True
        data = self._event(reading)
        reason = self._limited(now)
        if reason is not None:
            log.info("road_rage %.0f suppressed: %s", reading.score, reason)
            if self.on_suppress is not None:
                self.on_suppress(data, reason)
            return reading
        self._fired.append(now)
        log.info("road_rage %.0f (%s)", reading.score, ", ".join(reading.agreeing) or "face")
        self.emit(data)
        return reading

    def _limited(self, now: float) -> str | None:
        cfg = self.cfg
        while self._fired and now - self._fired[0] >= cfg["max_window_s"]:
            self._fired.popleft()
        if self._fired and now - self._fired[-1] < cfg["cooldown_s"]:
            return f"rage cooldown {cfg['cooldown_s']} s"
        if len(self._fired) >= cfg["max_per_window"]:
            return f"at most {cfg['max_per_window']} per {cfg['max_window_s']} s"
        return None

    def _event(self, reading: RageReading) -> dict[str, Any]:
        return {
            "event_id": f"rage-{next(self._ids)}",
            "type": "road_rage",
            "tier": "T2",
            "score": reading.score,
            "signals": {
                "face": reading.face,
                "voice": reading.voice,
                "driving": reading.driving,
                "agreeing": reading.agreeing,
                "face_alone": reading.face_alone,
                "hr_stress": reading.hr_stress,
                "music_rise": reading.music_rise,
                "firm": reading.score >= self.cfg["firm_score"],
            },
        }
