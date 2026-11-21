"""Contract tests (LEAD-T04)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from pydantic import ValidationError

from backend.schemas import TOPIC_MODELS, TOPICS, validate, validate_envelope

ROOT = Path(__file__).resolve().parent.parent

# The topic table in section 3 of the briefs.
BRIEF_TOPICS = {
    "driver.state", "camera.frame", "detector.event", "audio.features", "voice.say",
    "voice.status", "voice.reply", "alert.state", "vehicle.motion", "control.music",
    "music.volume", "bio.hr", "context.road", "checkin.schedule", "control.trip",
    "control.replay", "control.inject", "trip.summary",
}  # fmt: skip

DRIVER_STATE = {  # exactly what CV's --fake publishes (cv/cp0 state_dict)
    "face_present": True, "eye_closed_p": 0.95, "eyes_closed_ms": 1200, "perclos_60s": 0.18,
    "blink_rate_pm": 18.0, "gaze_dispersion_30s": 3.0,
    "head": {"yaw": 1.0, "pitch": -2.0, "roll": 0.0},
    "yawn_p": 0.02, "yawns_10m": 0, "anger_p": 0.02, "zoning_visual_p": 0.05,
    "phone_in_hand": False, "fps": 30.0,
}  # fmt: skip

PRETRIP = {
    "planned_min": 120, "sleep_h": 5, "awake_h": 14, "tiredness": 3, "caffeine": True,
    "engagement": "trivia", "lang": "ro", "emergency_contact": "+40700000000", "resting_hr": 62,
}  # fmt: skip

EXAMPLES = {
    "driver.state": DRIVER_STATE,
    "camera.frame": {"jpeg_b64": "/9j/4AAQ"},
    "detector.event": {
        "event_id": "e1", "type": "distraction", "tier": "T2", "score": 72,
        "signals": {"yaw": 34},
    },
    "audio.features": {  # exactly what voice's --fake publishes
        "loudness_db": -36.2, "baseline_db": -45.0, "shouting": False, "speaking": True,
        "swear_count_60s": 0, "anger_voice_p": None,
    },
    "voice.say": {"alert_id": "a1", "text": "Eyes on the road", "tier": "T2", "lang": "en"},
    "voice.status": {"alert_id": "a1", "state": "listening"},
    "voice.reply": {
        "alert_id": "a1", "transcript": "Da, sunt bine.", "intent": "OK", "latency_ms": 321,
        "path": "keyword",
    },
    "alert.state": {
        "alert_id": "a1", "type": "distraction", "tier": "T2", "status": "queued",
        "text": "Eyes on the road",
    },
    "vehicle.motion": {
        "accel_long_g": -0.3, "accel_lat_g": 0.05, "accel_mag_g": 0.31, "speed_kmh": 92.5,
        "lat": 40.41, "lon": -3.70, "road": "motorway", "source": "replay",
    },
    "control.music": {"duck": True, "level": 0.3},
    "music.volume": {"volume": 45, "playing": True},
    "bio.hr": {"bpm": 71, "rr_ms": [845, 850], "source": "sim"},
    "context.road": {"load": "HIGH", "traffic": "heavy", "next_maneuver": "exit", "dist_m": 400},
    "checkin.schedule": {"interval_min": 15, "next_checkin_ts": 1.7e9, "next_break_ts": 1.7e9},
    "control.trip": {"action": "start", "pretrip": PRETRIP},
    "control.replay": {"action": "seek", "trip": "motorway_normal", "segment": 120},
    "control.inject": {"event": "crash"},
    "trip.summary": {"text": "Calm drive, one break.", "stats": {"alerts": 3}},
}  # fmt: skip


def test_registry_matches_the_brief_topic_table():
    assert set(TOPICS) == BRIEF_TOPICS
    assert set(EXAMPLES) == BRIEF_TOPICS


@pytest.mark.parametrize("topic", sorted(BRIEF_TOPICS))
def test_example_validates(topic):
    model = validate(topic, EXAMPLES[topic])
    assert isinstance(model, TOPIC_MODELS[topic])


def test_defaults():
    say = validate("voice.say", EXAMPLES["voice.say"])
    assert say.listen_s == 8.0 and say.audio_key is None
    assert validate("control.trip", {"action": "stop"}).pretrip is None
    assert validate("voice.status", {"state": "idle"}).alert_id is None


@pytest.mark.parametrize(
    ("topic", "patch"),
    [
        ("driver.state", {"eye_closed_prob": 0.1}),  # renamed field (drift)
        ("driver.state", {"eye_closed_p": 1.5}),  # probability out of range
        ("detector.event", {"tier": "T4"}),
        ("detector.event", {"type": "sneezing"}),
        ("detector.event", {"score": 101}),
        ("voice.reply", {"intent": "MAYBE"}),
        ("voice.say", {"lang": "de"}),
        ("control.music", {"level": 2}),
        ("vehicle.motion", {"source": "live"}),
    ],
)
def test_bad_data_is_rejected(topic, patch):
    with pytest.raises(ValidationError):
        validate(topic, EXAMPLES[topic] | patch)


def test_missing_required_field_is_rejected():
    data = dict(DRIVER_STATE)
    del data["fps"]
    with pytest.raises(ValidationError):
        validate("driver.state", data)


def test_bad_pretrip_is_rejected():
    with pytest.raises(ValidationError):
        validate("control.trip", {"action": "start", "pretrip": PRETRIP | {"tiredness": 7}})


def test_unknown_topic():
    with pytest.raises(KeyError):
        validate("driver.mood", {})


def test_validate_envelope():
    env = {"topic": "control.inject", "ts": 1.7e9, "src": "ui", "data": {"event": "crash"}}
    assert validate_envelope(env).event == "crash"
    with pytest.raises(ValidationError):
        validate_envelope(env | {"src": "phone"})


def test_contract_json_is_up_to_date():
    path = ROOT / "scripts" / "export_schema.py"
    spec = importlib.util.spec_from_file_location("export_schema", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    contract = ROOT / "docs" / "contract.json"
    assert contract.read_text() == module.render(), "run python scripts/export_schema.py"
