from __future__ import annotations

import argparse
import base64
import logging
import math
import signal
import time
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.bus.client import ThreadPublisher
from backend.cv.camera import CameraCapture, CameraConfig, list_cameras
from backend.cv.emotion import EmotionModel
from backend.cv.eye_crops import crop_eyes
from backend.cv.eye_state import EyeStateModel
from backend.cv.face_landmarks import FaceLandmarker
from backend.cv.geometry import BlinkDetector, FaceGeometry, extract_geometry
from backend.cv.settings import (
    DEFAULT_CONFIG,
    float_setting,
    integer_setting,
    load_cv_settings,
    resolved_path,
    string_setting,
)
from backend.cv.visual_events import DriverStateMetrics, VisualEventDetector
from backend.schemas import validate

log = logging.getLogger("cv")


@dataclass(frozen=True)
class FakeDriverState:
    face_present: bool
    eye_closed_p: float
    eyes_closed_ms: int
    perclos_60s: float
    blink_rate_pm: float
    gaze_dispersion_30s: float
    yaw: float
    pitch: float
    roll: float
    yawn_p: float
    yawns_10m: int
    anger_p: float
    zoning_visual_p: float
    phone_in_hand: bool
    fps: float


def fake_sequences(sequence: str) -> Iterator[FakeDriverState]:
    base = FakeDriverState(
        face_present=True,
        eye_closed_p=0.02,
        eyes_closed_ms=0,
        perclos_60s=0.02,
        blink_rate_pm=18.0,
        gaze_dispersion_30s=3.0,
        yaw=1.0,
        pitch=-2.0,
        roll=0.0,
        yawn_p=0.02,
        yawns_10m=0,
        anger_p=0.02,
        zoning_visual_p=0.05,
        phone_in_hand=False,
        fps=30.0,
    )
    elapsed = 0
    while True:
        phase = (elapsed // 20) % 5
        requested = sequence
        if sequence == "all":
            requested = ("normal", "eyes_closing", "looking_down", "angry", "staring")[phase]

        state = base
        if requested == "eyes_closing":
            state = _replace(
                base,
                eye_closed_p=0.95,
                eyes_closed_ms=min(elapsed % 20 * 100, 1200),
                perclos_60s=0.18,
            )
        elif requested == "looking_down":
            state = _replace(base, pitch=-28.0, gaze_dispersion_30s=2.0)
        elif requested == "angry":
            state = _replace(base, anger_p=0.92)
        elif requested == "staring":
            state = _replace(base, gaze_dispersion_30s=0.8, blink_rate_pm=5.0, zoning_visual_p=0.82)

        yield state
        elapsed += 1


def _replace(base: FakeDriverState, **changes: object) -> FakeDriverState:
    values = base.__dict__.copy()
    values.update(changes)
    return FakeDriverState(**values)


def fake_state(sequence: str = "normal") -> FakeDriverState:
    return next(fake_sequences(sequence))


def state_dict(state: FakeDriverState) -> dict[str, object]:
    return {
        "face_present": state.face_present,
        "eye_closed_p": state.eye_closed_p,
        "eyes_closed_ms": state.eyes_closed_ms,
        "perclos_60s": state.perclos_60s,
        "blink_rate_pm": state.blink_rate_pm,
        "gaze_dispersion_30s": state.gaze_dispersion_30s,
        "head": {"yaw": state.yaw, "pitch": state.pitch, "roll": state.roll},
        "yawn_p": state.yawn_p,
        "yawns_10m": state.yawns_10m,
        "anger_p": state.anger_p,
        "zoning_visual_p": state.zoning_visual_p,
        "phone_in_hand": state.phone_in_hand,
        "fps": state.fps,
    }


class CVPublisher:
    def __init__(self, broker_url: str) -> None:
        self._publisher = ThreadPublisher("cv", broker_url)

    def start(self) -> None:
        self._publisher.start()

    def publish(self, topic: str, data: dict[str, object]) -> None:
        self._publisher.publish(topic, data)

    def stop(self) -> None:
        self._publisher.stop()


def _fake_camera_frame(width: int, jpeg_quality: int) -> dict[str, object]:
    import cv2
    import numpy as np

    if width <= 0:
        raise ValueError("Synthetic camera frame width must be positive")
    if not 1 <= jpeg_quality <= 100:
        raise ValueError("CV preview JPEG quality must be between 1 and 100")
    height = max(1, round(width * 9 / 16))
    frame = np.full((height, width, 3), (48, 54, 62), dtype=np.uint8)
    center = (width // 2, height // 2)
    scale = width / 320
    cv2.ellipse(
        frame,
        center,
        (round(32 * scale), round(40 * scale)),
        0,
        0,
        360,
        (180, 190, 202),
        -1,
    )
    for eye_x in (center[0] - round(11 * scale), center[0] + round(11 * scale)):
        cv2.circle(
            frame,
            (eye_x, center[1] - round(5 * scale)),
            max(1, round(2 * scale)),
            (40, 45, 50),
            -1,
        )
    cv2.ellipse(
        frame,
        (center[0], center[1] + round(12 * scale)),
        (round(8 * scale), max(1, round(3 * scale))),
        0,
        0,
        180,
        (60, 65, 72),
        2,
    )
    encoded_ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    if not encoded_ok:
        raise RuntimeError("Could not encode synthetic camera.frame preview")
    data = {"jpeg_b64": base64.b64encode(encoded).decode("ascii")}
    return validate("camera.frame", data).model_dump(mode="json")


def run_fake(
    sequence: str,
    broker_url: str,
    state_rate_hz: float,
    preview_rate_hz: float,
    preview_width: int,
    jpeg_quality: int,
    settings: dict[str, Any] | None = None,
) -> None:
    if not math.isfinite(state_rate_hz) or state_rate_hz <= 0:
        raise ValueError("state_rate_hz must be positive")
    if not math.isfinite(preview_rate_hz) or preview_rate_hz <= 0:
        raise ValueError("preview_rate_hz must be positive")
    preview = _fake_camera_frame(preview_width, jpeg_quality)
    publisher = CVPublisher(broker_url)
    publisher.start()
    generator = fake_sequences(sequence)
    event_detector = VisualEventDetector(settings or load_cv_settings())
    running = True

    def stop(_signum: int, _frame: object) -> None:
        nonlocal running
        running = False

    interval = 1.0 / state_rate_hz
    next_publish = time.monotonic()
    next_preview = next_publish
    try:
        signal.signal(signal.SIGINT, stop)
        signal.signal(signal.SIGTERM, stop)
        while running:
            state = next(generator)
            data = validate("driver.state", state_dict(state)).model_dump(mode="json")
            now = time.monotonic()
            for event in event_detector.update(data, now):
                validated_event = validate("detector.event", event).model_dump(mode="json")
                publisher.publish("detector.event", validated_event)
            publisher.publish("driver.state", data)
            if now >= next_preview:
                publisher.publish("camera.frame", preview)
                next_preview = now + 1.0 / preview_rate_hz
            next_publish += interval
            sleep_for = next_publish - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
    finally:
        publisher.stop()


def _camera_config(settings: dict[str, Any], index_override: int | None) -> CameraConfig:
    camera = settings.get("camera")
    if not isinstance(camera, dict):
        raise ValueError("config/thresholds.yaml must define cv.camera")
    index = integer_setting(settings, "camera", "index")
    if index_override is not None:
        index = index_override
    return CameraConfig(
        index=index,
        capture_width=integer_setting(settings, "camera", "capture_width", minimum=1),
        capture_height=integer_setting(settings, "camera", "capture_height", minimum=1),
        capture_fourcc=string_setting(settings, "camera", "capture_fourcc"),
        process_width=integer_setting(settings, "camera", "process_width", minimum=1),
        process_height=integer_setting(settings, "camera", "process_height", minimum=1),
    )


def _landmarker(settings: dict[str, Any]) -> FaceLandmarker:
    model_path = resolved_path(string_setting(settings, "face_landmarks", "model_path"))
    num_faces = integer_setting(settings, "face_landmarks", "num_faces", minimum=1)
    return FaceLandmarker(
        model_path=model_path,
        num_faces=num_faces,
        detection_confidence=float_setting(settings, "face_landmarks", "detection_confidence"),
        presence_confidence=float_setting(settings, "face_landmarks", "presence_confidence"),
        tracking_confidence=float_setting(settings, "face_landmarks", "tracking_confidence"),
    )


def _closed_probability_from_ear(
    eye_aspect_ratio: float, closed_threshold: float, open_threshold: float
) -> float:
    if not 0.0 < closed_threshold < open_threshold:
        raise ValueError("EAR thresholds must satisfy 0 < closed < open")
    return min(
        1.0, max(0.0, (open_threshold - eye_aspect_ratio) / (open_threshold - closed_threshold))
    )


def _empty_driver_data(fps: float) -> dict[str, object]:
    return {
        "face_present": False,
        "eye_closed_p": 0.0,
        "eyes_closed_ms": 0,
        "perclos_60s": 0.0,
        "blink_rate_pm": 0.0,
        "gaze_dispersion_30s": 0.0,
        "head": {"yaw": 0.0, "pitch": 0.0, "roll": 0.0},
        "yawn_p": 0.0,
        "yawns_10m": 0,
        "anger_p": 0.0,
        "zoning_visual_p": 0.0,
        "phone_in_hand": False,
        "fps": fps,
    }


def _next_deadline(deadline: float, now: float, interval: float) -> float:
    if not math.isfinite(deadline) or not math.isfinite(now) or not math.isfinite(interval):
        raise ValueError("Publish deadlines and intervals must be finite")
    if interval <= 0:
        raise ValueError("Publish interval must be positive")
    deadline += interval
    while deadline <= now:
        deadline += interval
    return deadline


def _driver_data(
    frame: Any,
    result: Any,
    settings: dict[str, Any],
    eye_model: EyeStateModel | None,
    blink_detector: BlinkDetector,
    fps: float,
    *,
    state_metrics: DriverStateMetrics | None = None,
    timestamp: float | None = None,
    last_eye_closed_p: float = 0.0,
    anger_probability: float | None = None,
) -> dict[str, object]:
    data = _empty_driver_data(fps)
    if not result.face_present:
        if state_metrics is not None:
            if timestamp is None:
                raise ValueError("timestamp is required when state_metrics is provided")
            return state_metrics.update(data, timestamp)
        return data

    geometry: FaceGeometry = extract_geometry(
        result.landmarks,
        result.transformation_matrix,
        pitch_sign=float_setting(settings, "face_landmarks", "pitch_sign"),
    )
    blink = blink_detector.update(geometry.eye_aspect_ratio)
    if blink:
        log.debug("Eye blink detected")
    if eye_model is not None:
        try:
            eye_crops = crop_eyes(
                frame,
                result.landmarks,
                padding_fraction=float_setting(settings, "eye_state", "crop_padding_fraction"),
                output_size=64,
            )
            eye_closed_p, _ = eye_model.predict(eye_crops)
        except Exception:
            # A turned-away or partially cropped face can produce degenerate
            # eye regions. Use the geometric fallback for this frame so stale
            # probabilities do not accumulate into a false microsleep alert.
            log.debug("Eye crop/model unavailable; using EAR fallback", exc_info=True)
            try:
                eye_closed_p = _closed_probability_from_ear(
                    geometry.eye_aspect_ratio,
                    float_setting(settings, "eye_state", "ear_closed_threshold"),
                    float_setting(settings, "eye_state", "ear_open_threshold"),
                )
            except (ValueError, ArithmeticError):
                eye_closed_p = last_eye_closed_p
    else:
        eye_closed_p = _closed_probability_from_ear(
            geometry.eye_aspect_ratio,
            float_setting(settings, "eye_state", "ear_closed_threshold"),
            float_setting(settings, "eye_state", "ear_open_threshold"),
        )
    data = {
        **data,
        "face_present": True,
        "eye_closed_p": eye_closed_p,
        "head": {"yaw": geometry.yaw, "pitch": geometry.pitch, "roll": geometry.roll},
    }
    if state_metrics is not None:
        if timestamp is None:
            raise ValueError("timestamp is required when state_metrics is provided")
        data = state_metrics.update(
            data,
            timestamp,
            gaze=(geometry.gaze_horizontal, geometry.gaze_vertical),
            blink=blink,
            head_pose=(geometry.yaw, geometry.pitch),
            mouth_aspect=geometry.mouth_aspect_ratio,
            blendshapes=result.blendshapes,
        )
    if anger_probability is not None:
        data["anger_p"] = min(1.0, max(0.0, anger_probability))
    return data


def run_live(
    settings: dict[str, Any],
    broker_url: str,
    camera_index: int | None = None,
) -> None:
    camera = CameraCapture(_camera_config(settings, camera_index))
    landmarker = _landmarker(settings)
    model_path = resolved_path(string_setting(settings, "eye_state", "model_path"))
    threshold = float_setting(settings, "eye_state", "closed_probability_threshold")
    max_model_size = integer_setting(settings, "eye_state", "max_model_size_bytes", minimum=1)
    fallback = settings.get("eye_state", {}).get("use_ear_fallback", False)
    if not isinstance(fallback, bool):
        raise ValueError("CV setting eye_state.use_ear_fallback must be boolean")
    eye_model: EyeStateModel | None
    if model_path.is_file():
        eye_model = EyeStateModel(
            model_path,
            closed_threshold=threshold,
            max_model_size_bytes=max_model_size,
        )
    elif fallback:
        eye_model = None
        log.warning("Eye-state model unavailable at %s; using configured EAR fallback", model_path)
    else:
        raise FileNotFoundError(f"Eye-state model not found: {model_path}")

    closed_ear = float_setting(settings, "eye_state", "ear_closed_threshold")
    open_ear = float_setting(settings, "eye_state", "ear_open_threshold")
    input_size = integer_setting(settings, "eye_state", "input_size", minimum=1)
    if input_size != 64:
        raise ValueError("Eye-state model input_size must be 64")
    blink_detector = BlinkDetector(closed_ear, open_ear)
    state_metrics = DriverStateMetrics(settings)
    event_detector = VisualEventDetector(settings)
    anger_model_path = resolved_path(string_setting(settings, "anger_model", "model_path"))
    anger_model: EmotionModel | None = None
    if anger_model_path.is_file():
        try:
            anger_model = EmotionModel(anger_model_path)
            log.info("Loaded anger-vs-other model from %s", anger_model_path)
        except (RuntimeError, ValueError, OSError):
            log.exception("Could not load anger model; using blendshape fallback")
    else:
        log.info("Anger model is not trained yet; using blendshape fallback")
    anger_rate_hz = float_setting(settings, "anger_model", "rate_hz")
    anger_smoothing = float_setting(settings, "anger_model", "smoothing_alpha")
    if anger_rate_hz <= 0 or not 0.0 < anger_smoothing <= 1.0:
        raise ValueError("CV anger model rate must be positive and smoothing alpha in (0, 1]")
    anger_probability = 0.0
    last_anger_sample = -math.inf
    state_rate_hz = float_setting(settings, "publisher", "state_rate_hz")
    preview_rate_hz = float_setting(settings, "preview", "rate_hz")
    preview_width = integer_setting(settings, "preview", "width", minimum=1)
    if state_rate_hz <= 0 or preview_rate_hz <= 0:
        raise ValueError("CV publish rates must be positive")

    publisher = CVPublisher(broker_url)
    running = True
    last_frame_timestamp: float | None = None
    next_state_publish = 0.0
    next_preview_publish = 0.0
    frame_times: deque[float] = deque()
    last_eye_closed_p = 0.0

    def stop(_signum: int, _frame: object) -> None:
        nonlocal running
        running = False

    try:
        landmarker.open()
        publisher.start()
        camera.start()
        signal.signal(signal.SIGINT, stop)
        signal.signal(signal.SIGTERM, stop)
        next_state_publish = time.monotonic()
        next_preview_publish = next_state_publish
        while running:
            frame, timestamp = camera.get_latest()
            if frame is None or timestamp is None:
                time.sleep(0.01)
                continue
            if timestamp == last_frame_timestamp:
                time.sleep(0.001)
                continue
            last_frame_timestamp = timestamp
            now = time.monotonic()
            frame_times.append(now)
            while frame_times and now - frame_times[0] > 1.0:
                frame_times.popleft()
            fps = 0.0
            if len(frame_times) > 1:
                elapsed = frame_times[-1] - frame_times[0]
                if elapsed > 0:
                    fps = (len(frame_times) - 1) / elapsed

            result = landmarker.process(frame)
            if (
                anger_model is not None
                and result.face_present
                and now - last_anger_sample >= 1.0 / anger_rate_hz
            ):
                last_anger_sample = now
                try:
                    observed_anger = anger_model.predict(frame, result.landmarks)
                    anger_probability += anger_smoothing * (observed_anger - anger_probability)
                except Exception:
                    log.debug("Anger model failed on frame; retaining last score", exc_info=True)
            data = _driver_data(
                frame,
                result,
                settings,
                eye_model,
                blink_detector,
                fps,
                state_metrics=state_metrics,
                timestamp=now,
                last_eye_closed_p=last_eye_closed_p,
                anger_probability=(anger_probability if anger_model is not None else None),
            )
            last_eye_closed_p = float(data["eye_closed_p"])
            for event in event_detector.update(data, now):
                validated_event = validate("detector.event", event).model_dump(mode="json")
                publisher.publish("detector.event", validated_event)
            if now >= next_state_publish:
                validated = validate("driver.state", data).model_dump(mode="json")
                publisher.publish("driver.state", validated)
                next_state_publish = _next_deadline(next_state_publish, now, 1.0 / state_rate_hz)
            if now >= next_preview_publish:
                _publish_preview(
                    publisher,
                    frame,
                    preview_width,
                    settings,
                    result.landmarks if result.face_present else [],
                )
                next_preview_publish = _next_deadline(
                    next_preview_publish, now, 1.0 / preview_rate_hz
                )
    finally:
        camera.stop()
        landmarker.close()
        publisher.stop()


def _publish_preview(
    publisher: CVPublisher,
    frame: Any,
    width: int,
    settings: dict[str, Any],
    landmarks: list[Any],
) -> None:
    import cv2

    height, source_width = frame.shape[:2]
    target_height = max(1, round(height * width / source_width))
    preview_frame = frame.copy()
    draw_landmarks = settings.get("preview", {}).get("draw_landmarks", False)
    if not isinstance(draw_landmarks, bool):
        raise ValueError("CV setting preview.draw_landmarks must be boolean")
    if draw_landmarks:
        for landmark in landmarks:
            point = (round(float(landmark.x) * source_width), round(float(landmark.y) * height))
            cv2.circle(preview_frame, point, 1, (0, 255, 0), -1)
    preview = cv2.resize(preview_frame, (width, target_height), interpolation=cv2.INTER_AREA)
    quality = integer_setting(settings, "preview", "jpeg_quality", minimum=1)
    if quality > 100:
        raise ValueError("CV preview JPEG quality must be <= 100")
    encoded_ok, encoded = cv2.imencode(
        ".jpg",
        preview,
        [cv2.IMWRITE_JPEG_QUALITY, quality],
    )
    if not encoded_ok:
        raise RuntimeError("Could not encode camera.frame preview")
    data = {"jpeg_b64": base64.b64encode(encoded).decode("ascii")}
    validated = validate("camera.frame", data).model_dump(mode="json")
    publisher.publish("camera.frame", validated)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DriveMate computer-vision process")
    parser.add_argument(
        "--fake", action="store_true", help="Publish synthetic driver state and camera frames"
    )
    parser.add_argument(
        "--sequence",
        choices=("normal", "eyes_closing", "looking_down", "angry", "staring", "all"),
        default=None,
    )
    parser.add_argument("--broker", default=None)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--camera", type=int, help="Override the configured camera index")
    parser.add_argument("--list-cameras", action="store_true", help="Print usable webcam indices")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.camera is not None and args.camera < 0:
        raise ValueError("--camera must be non-negative")
    settings = load_cv_settings(args.config)
    broker_url = args.broker or string_setting(settings, "publisher", "broker_url")
    if args.list_cameras:
        print(list_cameras(integer_setting(settings, "camera", "max_index", minimum=1)))
        return 0
    if args.fake:
        sequence = args.sequence or string_setting(settings, "fake", "sequence")
        if sequence not in {"normal", "eyes_closing", "looking_down", "angry", "staring", "all"}:
            raise ValueError(f"Unknown configured CV fake sequence: {sequence}")
        run_fake(
            sequence,
            broker_url,
            state_rate_hz=float_setting(settings, "publisher", "state_rate_hz"),
            preview_rate_hz=float_setting(settings, "preview", "rate_hz"),
            preview_width=integer_setting(settings, "preview", "width", minimum=1),
            jpeg_quality=integer_setting(settings, "preview", "jpeg_quality", minimum=1),
            settings=settings,
        )
        return 0
    run_live(settings, broker_url, args.camera)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
