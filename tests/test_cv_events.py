from __future__ import annotations

from copy import deepcopy

import pytest

from backend.cv.run import _next_deadline, run_fake
from backend.cv.settings import load_cv_settings
from backend.cv.visual_events import DriverStateMetrics, VisualEventDetector
from backend.schemas import validate


@pytest.fixture
def settings() -> dict:
    return deepcopy(load_cv_settings())


def _state(
    *,
    face_present: bool = True,
    yaw: float = 0.0,
    pitch: float = 0.0,
    eyes_closed_ms: int = 0,
    perclos_60s: float = 0.0,
    yawns_10m: int = 0,
) -> dict:
    return {
        "face_present": face_present,
        "eye_closed_p": 0.0,
        "eyes_closed_ms": eyes_closed_ms,
        "perclos_60s": perclos_60s,
        "yawns_10m": yawns_10m,
        "head": {"yaw": yaw, "pitch": pitch, "roll": 0.0},
    }


def test_distraction_requires_continuous_two_second_pose(settings: dict) -> None:
    detector = VisualEventDetector(settings)
    distracted = _state(pitch=-25.0)

    assert detector.update(distracted, now=0.0) == []
    assert detector.update(distracted, now=1.99) == []
    event = detector.update(distracted, now=2.0)[0]

    assert event["type"] == "distraction"
    assert event["tier"] == "T2"
    assert event["signals"]["pitch"] == -25.0
    assert validate("detector.event", event)
    assert detector.update(distracted, now=3.0) == []


def test_microsleep_is_a_t3_event(settings: dict) -> None:
    detector = VisualEventDetector(settings)

    event = detector.update(_state(eyes_closed_ms=1000), now=1.0)[0]

    assert event["type"] == "microsleep"
    assert event["tier"] == "T3"
    assert event["signals"] == {"eyes_closed_ms": 1000}
    assert validate("detector.event", event)


@pytest.mark.parametrize(
    "state",
    [
        _state(perclos_60s=0.15),
        _state(yawns_10m=3),
    ],
)
def test_drowsiness_can_be_triggered_by_perclos_or_yawns(settings: dict, state: dict) -> None:
    event = VisualEventDetector(settings).update(state, now=1.0)[0]

    assert event["type"] == "drowsiness"
    assert event["tier"] == "T2"
    assert validate("detector.event", event)


def test_no_face_event_waits_five_seconds_and_is_once_per_episode(settings: dict) -> None:
    detector = VisualEventDetector(settings)
    no_face = _state(face_present=False)

    assert detector.update(no_face, now=0.0) == []
    assert detector.update(no_face, now=4.99) == []
    event = detector.update(no_face, now=5.0)[0]
    assert event["type"] == "no_face"
    assert detector.update(no_face, now=6.0) == []


def test_cooldown_suppresses_retriggered_episode(settings: dict) -> None:
    settings["visual_events"]["cooldown_s"] = 30
    detector = VisualEventDetector(settings)
    distracted = _state(yaw=35.0)
    clear = _state()

    detector.update(distracted, now=0.0)
    first = detector.update(distracted, now=2.0)[0]
    detector.update(clear, now=2.1)
    detector.update(distracted, now=3.0)
    assert detector.update(distracted, now=5.0) == []
    detector.update(clear, now=6.0)
    detector.update(distracted, now=32.1)
    second = detector.update(distracted, now=34.1)[0]

    assert first["event_id"] != second["event_id"]
    assert second["type"] == "distraction"


def test_eye_metrics_track_closure_and_wait_for_perclos_window(settings: dict) -> None:
    settings["perclos_window_s"] = 0.5
    metrics = DriverStateMetrics(settings)
    open_state = _state()
    closed_state = _state()
    closed_state["eye_closed_p"] = 0.9

    assert metrics.update(open_state, now=0.0)["perclos_60s"] == 0.0
    first_closed = metrics.update(closed_state, now=0.1)
    assert first_closed["eyes_closed_ms"] == 0
    assert first_closed["perclos_60s"] == 0.0

    held_closed = metrics.update(closed_state, now=0.6)
    assert held_closed["eyes_closed_ms"] == 500
    assert held_closed["perclos_60s"] == pytest.approx(1.0)


def test_gaze_dispersion_and_blink_rate_use_configured_rolling_windows(settings: dict) -> None:
    settings["gaze_window_s"] = 2.0
    settings["blink_window_s"] = 3.0
    metrics = DriverStateMetrics(settings)
    state = _state()

    first = metrics.update(state, now=0.0, gaze=(0.0, 0.0), blink=True)
    second = metrics.update(state, now=1.0, gaze=(1.0, 0.0), blink=True)
    third = metrics.update(state, now=4.0, gaze=(1.0, 0.0))

    assert first["blink_rate_pm"] == 20.0
    assert second["blink_rate_pm"] == 40.0
    assert second["gaze_dispersion_30s"] == pytest.approx(22.5)
    assert third["blink_rate_pm"] == 20.0
    assert third["gaze_dispersion_30s"] == 0.0


def test_yawn_fallback_requires_a_sustained_open_mouth(settings: dict) -> None:
    settings["yawn_window_s"] = 10.0
    settings["yawn_fallback"]["hold_s"] = 0.5
    settings["yawn_fallback"]["cooldown_s"] = 1.0
    metrics = DriverStateMetrics(settings)
    state = _state()
    features = {"jawOpen": 0.95, "mouthStretchLeft": 0.8, "mouthStretchRight": 0.8}

    assert metrics.update(state, 0.0, mouth_aspect=0.8, blendshapes=features)["yawns_10m"] == 0
    assert metrics.update(state, 0.3, mouth_aspect=0.8, blendshapes=features)["yawns_10m"] == 0
    closed = metrics.update(state, 0.4, mouth_aspect=0.1, blendshapes={})
    assert closed["yawns_10m"] == 0
    assert metrics.update(state, 1.0, mouth_aspect=0.8, blendshapes=features)["yawns_10m"] == 0
    yawn = metrics.update(state, 1.5, mouth_aspect=0.8, blendshapes=features)

    assert yawn["yawn_p"] > settings["yawn_fallback"]["score_threshold"]
    assert yawn["yawns_10m"] == 1


def test_anger_fallback_combines_and_smooths_blendshapes(settings: dict) -> None:
    settings["anger_fallback"]["rate_hz"] = 10
    settings["anger_fallback"]["smoothing_alpha"] = 0.5
    metrics = DriverStateMetrics(settings)
    state = _state()
    features = {
        "browDownLeft": 1.0,
        "browDownRight": 1.0,
        "mouthPressLeft": 1.0,
        "mouthPressRight": 1.0,
        "noseSneerLeft": 1.0,
        "noseSneerRight": 1.0,
    }

    first = metrics.update(state, 0.0, blendshapes=features)
    second = metrics.update(state, 0.1, blendshapes=features)
    decay = metrics.update(_state(face_present=False), 0.2)

    assert first["anger_p"] == pytest.approx(0.5)
    assert second["anger_p"] == pytest.approx(0.75)
    assert decay["anger_p"] == pytest.approx(0.375)


def test_fallback_models_respect_configured_inference_rates(settings: dict) -> None:
    settings["anger_fallback"]["smoothing_alpha"] = 1.0
    settings["yawn_fallback"]["hold_s"] = 0.05
    metrics = DriverStateMetrics(settings)
    state = _state()
    angry = {"browDownLeft": 1.0, "browDownRight": 1.0}
    open_mouth = {"jawOpen": 1.0}

    first = metrics.update(state, 0.0, mouth_aspect=0.8, blendshapes={**angry, **open_mouth})
    skipped = metrics.update(state, 0.05, mouth_aspect=0.1, blendshapes={})
    sampled = metrics.update(state, 0.1, mouth_aspect=0.1, blendshapes=angry)

    assert first["yawn_p"] > 0.0
    assert skipped["yawn_p"] == first["yawn_p"]
    assert skipped["anger_p"] == first["anger_p"]
    assert sampled["yawn_p"] == 0.0
    assert sampled["anger_p"] == 0.35


def test_zoning_score_requires_low_gaze_blink_and_head_motion(settings: dict) -> None:
    settings["gaze_window_s"] = 2.0
    settings["blink_window_s"] = 2.0
    settings["zoning_stare_s"] = 2.0
    settings["zoning"]["window_s"] = 2.0
    metrics = DriverStateMetrics(settings)
    state = _state()

    first = metrics.update(state, 0.0, gaze=(0.0, 0.0), head_pose=(0.0, 0.0))
    second = metrics.update(state, 1.0, gaze=(0.0, 0.0), head_pose=(0.0, 0.0))
    third = metrics.update(state, 2.0, gaze=(0.0, 0.0), head_pose=(0.0, 0.0))
    fourth = metrics.update(state, 3.0, gaze=(0.0, 0.0), head_pose=(0.0, 0.0))

    assert first["zoning_visual_p"] == 0.0
    assert second["zoning_visual_p"] == 0.0
    assert 0.0 < third["zoning_visual_p"] < 1.0
    assert fourth["zoning_visual_p"] == pytest.approx(1.0)


def test_fake_live_loop_publishes_visual_detector_events(monkeypatch) -> None:
    messages = []
    signal_handlers = []
    clock = 0.0
    sleep_calls = 0

    class Publisher:
        def __init__(self, _broker_url: str) -> None:
            pass

        def start(self) -> None:
            pass

        def publish(self, topic: str, data: dict[str, object]) -> None:
            messages.append((topic, data))

        def stop(self) -> None:
            pass

    def fake_sleep(duration: float) -> None:
        nonlocal clock, sleep_calls
        clock += duration
        sleep_calls += 1
        if sleep_calls == 25:
            signal_handlers[-1](None, None)

    monkeypatch.setattr("backend.cv.run.CVPublisher", Publisher)
    monkeypatch.setattr(
        "backend.cv.run.signal.signal",
        lambda _signal, handler: signal_handlers.append(handler),
    )
    monkeypatch.setattr("backend.cv.run.time.monotonic", lambda: clock)
    monkeypatch.setattr("backend.cv.run.time.sleep", fake_sleep)

    run_fake("looking_down", "ws://localhost:8765/bus", 10, 5, 320, 60)

    events = [data for topic, data in messages if topic == "detector.event"]
    assert len(events) == 1
    assert events[0]["type"] == "distraction"
    assert validate("detector.event", events[0])


def test_publish_deadline_keeps_phase_and_skips_missed_ticks() -> None:
    assert _next_deadline(0.1, 0.133, 0.1) == pytest.approx(0.2)
    assert _next_deadline(0.2, 0.36, 0.1) == pytest.approx(0.4)
