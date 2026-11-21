from __future__ import annotations

import math
import uuid
from collections import deque
from collections.abc import Iterable, Mapping
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
    """Compute rolling eye, yawn, expression-fallback, and zoning metrics."""

    def __init__(self, settings: dict[str, Any]) -> None:
        self._closed_probability_threshold = float_setting(
            settings, "eye_state", "closed_probability_threshold"
        )
        self._perclos_window_s = _float_value(settings, "perclos_window_s")
        self._gaze_window_s = _float_value(settings, "gaze_window_s")
        self._blink_window_s = _float_value(settings, "blink_window_s")
        self._yawn_window_s = _float_value(settings, "yawn_window_s")
        self._yawn_rate_hz = float_setting(settings, "yawn_fallback", "rate_hz")
        self._yawn_score_threshold = float_setting(settings, "yawn_fallback", "score_threshold")
        self._yawn_mouth_aspect_threshold = float_setting(
            settings, "yawn_fallback", "mouth_aspect_threshold"
        )
        self._yawn_hold_s = float_setting(settings, "yawn_fallback", "hold_s")
        self._yawn_cooldown_s = float_setting(settings, "yawn_fallback", "cooldown_s")
        self._yawn_model_threshold = float_setting(
            settings, "activity_models", "drowsiness_threshold"
        )
        self._yawn_model_weight = float_setting(
            settings, "activity_models", "yawn_blend_weight"
        )
        if not 0.0 <= self._yawn_model_weight <= 1.0:
            raise ValueError("CV activity yawn blend weight must be between 0 and 1")
        self._anger_rate_hz = float_setting(settings, "anger_fallback", "rate_hz")
        self._anger_alpha = float_setting(settings, "anger_fallback", "smoothing_alpha")
        self._anger_weights = {
            "brow": float_setting(settings, "anger_fallback", "weights_brow"),
            "mouth": float_setting(settings, "anger_fallback", "weights_mouth"),
            "nose": float_setting(settings, "anger_fallback", "weights_nose"),
        }
        zoning = settings.get("zoning")
        if not isinstance(zoning, dict):
            raise ValueError("Missing or invalid CV setting zoning")
        self._zoning_window_s = _float_value(zoning, "window_s")
        self._zoning_gaze_limit = _float_value(zoning, "gaze_dispersion_deg")
        self._zoning_blink_norm = _float_value(zoning, "population_blink_rate_pm")
        self._zoning_blink_ratio = _float_value(zoning, "low_blink_ratio")
        self._zoning_head_limit = _float_value(zoning, "head_movement_deg")
        self._zoning_stare_s = _float_value(settings, "zoning_stare_s")
        weights = zoning.get("weights")
        if not isinstance(weights, dict):
            raise ValueError("Missing or invalid CV setting zoning.weights")
        self._zoning_weights = {
            name: _float_value(weights, name) for name in ("gaze", "blink", "head")
        }
        weight_total = sum(self._zoning_weights.values())
        if weight_total <= 0:
            raise ValueError("CV zoning weights must sum to a positive value")
        self._zoning_weights = {
            name: weight / weight_total for name, weight in self._zoning_weights.items()
        }
        for name, value in (
            ("perclos_window_s", self._perclos_window_s),
            ("gaze_window_s", self._gaze_window_s),
            ("blink_window_s", self._blink_window_s),
            ("yawn_window_s", self._yawn_window_s),
            ("yawn_fallback.rate_hz", self._yawn_rate_hz),
            ("yawn_fallback.hold_s", self._yawn_hold_s),
            ("yawn_fallback.cooldown_s", self._yawn_cooldown_s),
            ("anger_fallback.rate_hz", self._anger_rate_hz),
            ("zoning.window_s", self._zoning_window_s),
            ("zoning_stare_s", self._zoning_stare_s),
        ):
            if value <= 0:
                raise ValueError(f"CV setting cv.{name} must be positive")
        if not 0.0 < self._anger_alpha <= 1.0:
            raise ValueError("CV anger_fallback.smoothing_alpha must be in (0, 1]")
        if not 0.0 < self._yawn_score_threshold <= 1.0:
            raise ValueError("CV yawn_fallback.score_threshold must be in (0, 1]")
        if self._yawn_mouth_aspect_threshold <= 0:
            raise ValueError("CV yawn_fallback.mouth_aspect_threshold must be positive")
        self._segments: deque[tuple[float, float, bool]] = deque()
        self._gaze_samples: deque[tuple[float, float, float]] = deque()
        self._blink_times: deque[float] = deque()
        self._head_samples: deque[tuple[float, float, float]] = deque()
        self._yawn_times: deque[float] = deque()
        self._yawn_since: float | None = None
        self._yawn_emitted = False
        self._last_yawn = -math.inf
        self._anger_smoothed = 0.0
        self._yawn_score = 0.0
        self._last_yawn_sample = -math.inf
        self._last_anger_sample = -math.inf
        self._zoning_since: float | None = None
        self._previous_timestamp: float | None = None
        self._previous_face_present = False
        self._previous_closed = False
        self._closed_since: float | None = None

    def update(
        self,
        data: Mapping[str, Any],
        now: float,
        *,
        gaze: tuple[float, float] | None = None,
        blink: bool = False,
        head_pose: tuple[float, float] | None = None,
        mouth_aspect: float | None = None,
        blendshapes: Mapping[str, float] | None = None,
        yawn_model_p: float | None = None,
        model_blink: bool = False,
    ) -> dict[str, Any]:
        if not math.isfinite(now):
            raise ValueError("Driver state timestamp must be finite")
        state = dict(data)
        face_present = bool(state["face_present"])
        eye_closed_p = float(state["eye_closed_p"])
        if not math.isfinite(eye_closed_p) or not 0.0 <= eye_closed_p <= 1.0:
            raise ValueError("eye_closed_p must be finite and between 0 and 1")
        if yawn_model_p is not None and (
            not math.isfinite(yawn_model_p) or not 0.0 <= yawn_model_p <= 1.0
        ):
            raise ValueError("yawn_model_p must be finite and between 0 and 1")
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

        if face_present and gaze is not None:
            gaze_x, gaze_y = gaze
            if not math.isfinite(gaze_x) or not math.isfinite(gaze_y):
                raise ValueError("Gaze coordinates must be finite")
            self._gaze_samples.append(
                (now, math.degrees(math.atan(gaze_x)), math.degrees(math.atan(gaze_y)))
            )
        gaze_cutoff = now - self._gaze_window_s
        gaze_history_cutoff = now - max(self._gaze_window_s, self._zoning_window_s)
        while self._gaze_samples and self._gaze_samples[0][0] < gaze_history_cutoff:
            self._gaze_samples.popleft()
        state["gaze_dispersion_30s"] = self._gaze_dispersion(
            sample for sample in self._gaze_samples if sample[0] >= gaze_cutoff
        )

        if (blink or model_blink) and face_present:
            self._blink_times.append(now)
        blink_cutoff = now - self._blink_window_s
        while self._blink_times and self._blink_times[0] < blink_cutoff:
            self._blink_times.popleft()
        state["blink_rate_pm"] = len(self._blink_times) * 60.0 / self._blink_window_s

        if face_present and head_pose is not None:
            yaw, pitch = head_pose
            if not math.isfinite(yaw) or not math.isfinite(pitch):
                raise ValueError("Head pose must contain finite values")
            self._head_samples.append((now, yaw, pitch))
        head_cutoff = now - self._zoning_window_s
        while self._head_samples and self._head_samples[0][0] < head_cutoff:
            self._head_samples.popleft()

        if not face_present:
            self._yawn_since = None
            self._yawn_emitted = False
            self._yawn_score = 0.0
        elif now - self._last_yawn_sample >= 1.0 / self._yawn_rate_hz:
            self._last_yawn_sample = now
            blendshape_yawn_score = 0.0
            blendshape_yawn_active = False
            if mouth_aspect is not None and blendshapes is not None:
                jaw_open = self._blendshape(blendshapes, "jawOpen")
                mouth_stretch = (
                    self._blendshape(blendshapes, "mouthStretchLeft")
                    + self._blendshape(blendshapes, "mouthStretchRight")
                ) / 2.0
                mouth_funnel = self._blendshape(blendshapes, "mouthFunnel")
                blendshape_yawn_score = (
                    0.65 * jaw_open + 0.2 * mouth_stretch + 0.15 * mouth_funnel
                )
                blendshape_yawn_active = (
                    blendshape_yawn_score >= self._yawn_score_threshold
                    and mouth_aspect >= self._yawn_mouth_aspect_threshold
                )
            model_yawn_score = (
                min(1.0, max(0.0, yawn_model_p)) if yawn_model_p is not None else None
            )
            if model_yawn_score is None:
                self._yawn_score = blendshape_yawn_score
            else:
                self._yawn_score = (
                    (1.0 - self._yawn_model_weight) * blendshape_yawn_score
                    + self._yawn_model_weight * model_yawn_score
                )
            model_yawn_active = (
                model_yawn_score is not None
                and model_yawn_score >= self._yawn_model_threshold
            )
            if blendshape_yawn_active or model_yawn_active:
                if self._yawn_since is None:
                    self._yawn_since = now
                if now - self._yawn_since >= self._yawn_hold_s and not self._yawn_emitted:
                    if now - self._last_yawn >= self._yawn_cooldown_s:
                        self._yawn_times.append(now)
                        self._last_yawn = now
                    self._yawn_emitted = True
            else:
                self._yawn_since = None
                self._yawn_emitted = False
        yawn_cutoff = now - self._yawn_window_s
        while self._yawn_times and self._yawn_times[0] < yawn_cutoff:
            self._yawn_times.popleft()
        state["yawn_p"] = self._yawn_score
        state["yawns_10m"] = len(self._yawn_times)

        if now - self._last_anger_sample >= 1.0 / self._anger_rate_hz:
            self._last_anger_sample = now
            raw_anger = self._anger_score(blendshapes or {}) if face_present else 0.0
            self._anger_smoothed += self._anger_alpha * (raw_anger - self._anger_smoothed)
        state["anger_p"] = min(1.0, max(0.0, self._anger_smoothed))

        zoning_gaze_cutoff = now - self._zoning_window_s
        zoning_gaze_dispersion = self._gaze_dispersion(
            sample for sample in self._gaze_samples if sample[0] >= zoning_gaze_cutoff
        )
        gaze_dispersion = zoning_gaze_dispersion
        gaze_score = min(1.0, max(0.0, 1.0 - gaze_dispersion / self._zoning_gaze_limit))
        blink_score = min(
            1.0,
            max(
                0.0,
                (self._zoning_blink_norm * self._zoning_blink_ratio - state["blink_rate_pm"])
                / (self._zoning_blink_norm * self._zoning_blink_ratio),
            ),
        )
        if len(self._head_samples) > 1:
            yaw_values = [sample[1] for sample in self._head_samples]
            pitch_values = [sample[2] for sample in self._head_samples]
            movement = max(
                max(yaw_values) - min(yaw_values),
                max(pitch_values) - min(pitch_values),
            )
            head_score = min(1.0, max(0.0, 1.0 - movement / self._zoning_head_limit))
        else:
            head_score = 0.0
        zoning_active = (
            face_present
            and len(self._gaze_samples) > 1
            and gaze_score > 0
            and blink_score > 0
            and head_score > 0
        )
        if zoning_active:
            if self._zoning_since is None:
                self._zoning_since = now
            duration_score = min(1.0, (now - self._zoning_since) / self._zoning_stare_s)
        else:
            self._zoning_since = None
            duration_score = 0.0
        weighted_score = sum(
            self._zoning_weights[name] * score
            for name, score in (
                ("gaze", gaze_score),
                ("blink", blink_score),
                ("head", head_score),
            )
        )
        state["zoning_visual_p"] = min(1.0, max(0.0, weighted_score * duration_score))
        return state

    @staticmethod
    def _gaze_dispersion(
        samples: Iterable[tuple[float, float, float]],
    ) -> float:
        points = list(samples)
        if len(points) < 2:
            return 0.0
        gaze_xs = [sample[1] for sample in points]
        gaze_ys = [sample[2] for sample in points]
        mean_x = sum(gaze_xs) / len(gaze_xs)
        mean_y = sum(gaze_ys) / len(gaze_ys)
        variance = sum(
            (x - mean_x) ** 2 + (y - mean_y) ** 2 for x, y in zip(gaze_xs, gaze_ys, strict=True)
        ) / len(points)
        return math.sqrt(variance)

    @staticmethod
    def _blendshape(blendshapes: Mapping[str, float], name: str) -> float:
        value = float(blendshapes.get(name, 0.0))
        if not math.isfinite(value):
            raise ValueError(f"Blendshape {name} must be finite")
        return min(1.0, max(0.0, value))

    def _anger_score(self, blendshapes: Mapping[str, float]) -> float:
        brow = (
            self._blendshape(blendshapes, "browDownLeft")
            + self._blendshape(blendshapes, "browDownRight")
        ) / 2.0
        mouth = (
            self._blendshape(blendshapes, "mouthPressLeft")
            + self._blendshape(blendshapes, "mouthPressRight")
        ) / 2.0
        nose = (
            self._blendshape(blendshapes, "noseSneerLeft")
            + self._blendshape(blendshapes, "noseSneerRight")
        ) / 2.0
        return sum(
            self._anger_weights[name] * score
            for name, score in (("brow", brow), ("mouth", mouth), ("nose", nose))
        )


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
        self._phone_hold_s = _float_value(settings, "phone_hold_s")
        activity_models = settings.get("activity_models")
        if not isinstance(activity_models, dict):
            raise ValueError("Missing or invalid CV setting activity_models")
        self._model_distraction_threshold = float_setting(
            settings, "activity_models", "distraction_threshold"
        )
        self._model_drowsiness_threshold = float_setting(
            settings, "activity_models", "drowsiness_threshold"
        )
        self._phone_threshold = float_setting(settings, "activity_models", "phone_threshold")
        self._cooldown_s = float_setting(settings, "visual_events", "cooldown_s")
        for name, value in (
            ("eyes_off_s", self._distraction_hold_s),
            ("microsleep_s", self._microsleep_s),
            ("no_face_warning_s", self._no_face_warning_s),
            ("phone_hold_s", self._phone_hold_s),
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
        distraction_model_p = float(state.get("distraction_model_p", 0.0))
        drowsiness_model_p = float(state.get("drowsiness_model_p", 0.0))
        phone_model_p = float(state.get("phone_model_p", 0.0))
        if not all(
            math.isfinite(value)
            for value in (
                yaw,
                pitch,
                perclos,
                distraction_model_p,
                drowsiness_model_p,
                phone_model_p,
            )
        ):
            raise ValueError("Visual event signals must be finite")

        conditions = {
            "distraction": face_present
            and (
                abs(yaw) > self._yaw_threshold
                or pitch < self._pitch_threshold
                or distraction_model_p >= self._model_distraction_threshold
            ),
            "microsleep": face_present and eyes_closed_ms >= self._microsleep_s * 1000,
            "drowsiness": face_present
            and (
                perclos >= self._perclos_threshold
                or yawns >= self._yawn_count
                or drowsiness_model_p >= self._model_drowsiness_threshold
            ),
            "phone": face_present and phone_model_p >= self._phone_threshold,
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
            "phone": elapsed.get("phone", 0.0) >= self._phone_hold_s,
        }
        tiers = {
            "distraction": "T2",
            "microsleep": "T3",
            "drowsiness": "T2",
            "no_face": "T2",
            "phone": "T2",
        }
        scores = {
            "distraction": 70.0,
            "microsleep": 95.0,
            "drowsiness": 70.0,
            "no_face": 70.0,
            "phone": 85.0,
        }
        signals = {
            "distraction": {
                "yaw": yaw,
                "pitch": pitch,
                "model_p": distraction_model_p,
                "action_probabilities": state.get("distraction_action_probabilities", {}),
                "held_s": elapsed.get("distraction", 0.0),
            },
            "microsleep": {"eyes_closed_ms": eyes_closed_ms},
            "drowsiness": {
                "perclos_60s": perclos,
                "yawns_10m": yawns,
                "model_p": drowsiness_model_p,
                "class_probabilities": state.get("drowsiness_class_probabilities", {}),
            },
            "no_face": {"face_absent_s": elapsed.get("no_face", 0.0)},
            "phone": {"phone_p": phone_model_p, "held_s": elapsed.get("phone", 0.0)},
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
