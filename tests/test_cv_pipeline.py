from __future__ import annotations

import base64
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from backend.cv.eye_crops import crop_eyes, preprocess_eye
from backend.cv.eye_state import EyeStateModel
from backend.cv.geometry import BlinkDetector, extract_geometry
from backend.cv.run import (
    _closed_probability_from_ear,
    _driver_data,
    _fake_camera_frame,
    _publish_preview,
    run_fake,
)
from backend.cv.settings import load_cv_settings
from backend.schemas import validate
from training.eye_state.train import EyeSample, parse_mrl_filename, split_by_subject


def _landmarks() -> list[SimpleNamespace]:
    return [SimpleNamespace(x=0.5, y=0.5) for _ in range(478)]


def test_config_has_live_camera_and_eye_settings() -> None:
    settings = load_cv_settings()

    assert settings["camera"]["capture_width"] == 1280
    assert settings["camera"]["process_width"] == 640
    assert settings["eye_state"]["input_size"] == 64
    assert settings["preview"]["width"] == 320
    assert settings["eye_state"]["max_model_size_bytes"] == 5_000_000


def test_fake_camera_frame_is_valid_contract_sized_jpeg() -> None:
    import cv2

    data = _fake_camera_frame(320, 60)
    encoded = base64.b64decode(data["jpeg_b64"])
    frame = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)

    assert frame is not None
    assert frame.shape[1] == 320
    assert validate("camera.frame", data).jpeg_b64 == data["jpeg_b64"]


def test_fake_mode_publishes_driver_state_and_camera_frame(monkeypatch) -> None:
    messages = []
    signal_handlers = []

    class Publisher:
        def __init__(self, _broker_url: str) -> None:
            pass

        def start(self) -> None:
            pass

        def publish(self, topic: str, data: dict[str, object]) -> None:
            messages.append((topic, data))

        def stop(self) -> None:
            pass

    monkeypatch.setattr("backend.cv.run.CVPublisher", Publisher)
    monkeypatch.setattr(
        "backend.cv.run.signal.signal",
        lambda _signal, handler: signal_handlers.append(handler),
    )
    monkeypatch.setattr("backend.cv.run.time.monotonic", lambda: 0.0)
    monkeypatch.setattr(
        "backend.cv.run.time.sleep",
        lambda _duration: signal_handlers[-1](None, None),
    )

    run_fake("normal", "ws://localhost:8765/bus", 10, 5, 320, 60)

    assert [topic for topic, _data in messages] == ["driver.state", "camera.frame"]
    assert validate("camera.frame", messages[1][1]).jpeg_b64


def test_geometry_extracts_pose_iris_and_aspect_ratios() -> None:
    landmarks = _landmarks()
    left = (
        (33, 0.2, 0.5),
        (160, 0.22, 0.48),
        (158, 0.28, 0.48),
        (133, 0.3, 0.5),
        (153, 0.28, 0.52),
        (144, 0.22, 0.52),
    )
    right = (
        (362, 0.6, 0.5),
        (385, 0.62, 0.48),
        (387, 0.68, 0.48),
        (263, 0.7, 0.5),
        (373, 0.68, 0.52),
        (380, 0.62, 0.52),
    )
    for index, x, y in (*left, *right):
        landmarks[index] = SimpleNamespace(x=x, y=y)
    for index in (468, 469, 470, 471, 472):
        landmarks[index] = SimpleNamespace(x=0.25, y=0.5)
    for index in (473, 474, 475, 476, 477):
        landmarks[index] = SimpleNamespace(x=0.65, y=0.5)
    landmarks[61] = SimpleNamespace(x=0.4, y=0.7)
    landmarks[291] = SimpleNamespace(x=0.6, y=0.7)
    landmarks[13] = SimpleNamespace(x=0.5, y=0.69)
    landmarks[14] = SimpleNamespace(x=0.5, y=0.71)

    angle = math.radians(20)
    matrix = [
        [math.cos(angle), 0.0, math.sin(angle), 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-math.sin(angle), 0.0, math.cos(angle), 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    features = extract_geometry(landmarks, matrix)

    assert features.pitch == pytest.approx(-20.0)
    assert features.eye_aspect_ratio > 0.0
    assert features.mouth_aspect_ratio == pytest.approx(0.1)
    assert math.isfinite(features.gaze_horizontal)


def test_blink_detector_uses_hysteresis() -> None:
    detector = BlinkDetector(closed_threshold=0.18, open_threshold=0.23)

    assert detector.update(0.20) is False
    assert detector.update(0.17) is False
    assert detector.update(0.20) is False
    assert detector.update(0.24) is True
    assert detector.update(0.24) is False


def test_eye_preprocess_and_face_crops_share_64_square_float_input() -> None:
    frame = np.full((100, 160, 3), 127, dtype=np.uint8)
    landmarks = _landmarks()
    for indices, center_x in (
        ((33, 160, 158, 133, 153, 144), 0.35),
        ((362, 385, 387, 263, 373, 380), 0.65),
    ):
        for index, y_offset in zip(indices, (0.0, -0.02, -0.02, 0.0, 0.02, 0.02), strict=True):
            landmarks[index] = SimpleNamespace(
                x=center_x + (0.03 if index in indices[1:3] else -0.03), y=0.4 + y_offset
            )

    crops = crop_eyes(frame, landmarks)
    prepared = preprocess_eye(frame[35:45, 50:60])

    assert len(crops) == 2
    assert all(crop.shape == (64, 64) and crop.dtype == np.float32 for crop in crops)
    assert prepared.shape == (64, 64)
    assert all(0.0 <= float(crop.min()) <= float(crop.max()) <= 1.0 for crop in crops)


def test_mrl_label_parser_and_split_are_by_subject(tmp_path: Path) -> None:
    samples = [
        EyeSample(
            tmp_path / f"s{subject:04d}_{index:05d}_0_0_{closed}_0_0_01.png",
            f"s{subject:04d}",
            closed,
        )
        for subject in range(10)
        for index, closed in enumerate((0, 1))
    ]

    parsed = parse_mrl_filename(samples[0].path)
    train, validation = split_by_subject(samples, validation_fraction=0.2, seed=42)

    assert parsed.closed == 1
    assert parsed.subject == "s0000"
    assert {sample.subject for sample in train}.isdisjoint(
        {sample.subject for sample in validation}
    )
    assert len(validation) == 4


def test_eye_state_model_reports_closed_probability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_path = tmp_path / "eye.onnx"
    model_path.touch()

    class FakeSession:
        def __init__(self, _path: str, providers: list[str]) -> None:
            assert providers == ["CPUExecutionProvider"]

        def get_inputs(self) -> list[SimpleNamespace]:
            return [SimpleNamespace(name="eye")]

        def get_outputs(self) -> list[SimpleNamespace]:
            return [SimpleNamespace(name="logits")]

        def run(self, _outputs: list[str], inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
            assert inputs["eye"].shape == (2, 1, 64, 64)
            return [np.asarray([[0.0, 3.0], [3.0, 0.0]], dtype=np.float32)]

    monkeypatch.setitem(
        __import__("sys").modules,
        "onnxruntime",
        SimpleNamespace(InferenceSession=FakeSession),
    )
    model = EyeStateModel(
        model_path,
        closed_threshold=0.5,
        max_model_size_bytes=5_000_000,
    )
    average, probabilities = model.predict([np.zeros((64, 64), dtype=np.float32)] * 2)

    assert average == pytest.approx(0.5)
    assert probabilities[0] > probabilities[1]
    assert _closed_probability_from_ear(0.18, 0.18, 0.23) == 1.0


def test_live_driver_state_matches_shared_schema_with_ear_fallback() -> None:
    settings = load_cv_settings()
    result = SimpleNamespace(
        face_present=True,
        landmarks=_landmarks(),
        transformation_matrix=np.eye(4).tolist(),
    )
    blink_detector = BlinkDetector(
        settings["eye_state"]["ear_closed_threshold"],
        settings["eye_state"]["ear_open_threshold"],
    )

    data = _driver_data(
        np.zeros((360, 640, 3), dtype=np.uint8),
        result,
        settings,
        eye_model=None,
        blink_detector=blink_detector,
        fps=20.0,
    )
    validated = validate("driver.state", data)

    assert validated.face_present is True
    assert validated.eye_closed_p == 1.0
    assert validated.fps == 20.0


def test_camera_preview_is_320_pixels_wide_and_base64_encoded() -> None:
    settings = load_cv_settings()
    published: list[tuple[str, dict[str, str]]] = []

    class Publisher:
        def publish(self, topic: str, data: dict[str, str]) -> None:
            published.append((topic, data))

    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    _publish_preview(Publisher(), frame, 320, settings, [])

    topic, data = published[0]
    decoded = base64.b64decode(data["jpeg_b64"])
    import cv2

    preview = cv2.imdecode(np.frombuffer(decoded, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert topic == "camera.frame"
    assert preview.shape[1] == 320
