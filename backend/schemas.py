"""DriveMate shared contract (LEAD-T04): one pydantic v2 model per bus topic.

This file is the single source of truth, owned by the lead. Adding an optional field is fine;
renaming or removing one needs the whole team's OK. After any change run
`python scripts/export_schema.py` so docs/contract.json (and the dashboard types) follow.

    from backend.schemas import validate
    state = validate("driver.state", data)      # -> DriverState, raises ValidationError
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

CONTRACT_VERSION = "1.0.0"

# Shared enums

Src = Literal["cv", "voice", "core", "ui", "hr", "maps"]
Tier = Literal["T1", "T2", "T3"]
Lang = Literal["ro", "en"]
EventType = Literal[
    "distraction",
    "microsleep",
    "drowsiness",
    "phone",
    "no_face",
    "road_rage",
    "zoning_out",
    "crash_candidate",
    "checkin_due",
    "break_due",
]
Intent = Literal["OK", "NOT_OK", "ANSWER", "HELP", "UNCLEAR", "NONE"]
VoiceState = Literal["speaking", "listening", "idle"]
AlertStatus = Literal["queued", "held", "speaking", "listening", "closed", "escalated"]
RoadLoad = Literal["LOW", "MEDIUM", "HIGH"]

Prob = Annotated[float, Field(ge=0.0, le=1.0)]


class Model(BaseModel):
    # Unknown fields are rejected so name drift between teammates fails loudly.
    model_config = ConfigDict(extra="forbid")


class Envelope(Model):
    topic: str
    ts: float = Field(description="Unix time in seconds")
    src: Src
    data: dict[str, Any]


# CV


class HeadPose(Model):
    yaw: float = Field(description="degrees")
    pitch: float = Field(description="degrees")
    roll: float = Field(description="degrees")


class DriverState(Model):
    """driver.state, 10 Hz. With no face, send 0 or the last value; face_present says why."""

    face_present: bool
    eye_closed_p: Prob
    eyes_closed_ms: int = Field(ge=0, description="current continuous eye closure")
    perclos_60s: Prob = Field(description="fraction of the last 60 s closed")
    blink_rate_pm: float = Field(ge=0.0, description="blinks per minute")
    gaze_dispersion_30s: float = Field(ge=0.0, description="degrees")
    head: HeadPose
    yawn_p: Prob
    yawns_10m: int = Field(ge=0)
    anger_p: Prob
    zoning_visual_p: Prob
    phone_in_hand: bool
    fps: float = Field(ge=0.0)


class CameraFrame(Model):
    """camera.frame, 5 Hz. 320 px wide JPEG; never written to disk."""

    jpeg_b64: str


class DetectorEvent(Model):
    """detector.event, on event. Visual events from CV, fusion events from core."""

    event_id: str
    type: EventType
    tier: Tier
    score: float = Field(ge=0.0, le=100.0)
    signals: dict[str, Any] = Field(default_factory=dict)


# Voice


class AudioFeatures(Model):
    """audio.features, 5 Hz."""

    loudness_db: float
    baseline_db: float
    shouting: bool
    speaking: bool
    swear_count_60s: int = Field(ge=0)
    anger_voice_p: Prob | None = None


class VoiceStatus(Model):
    """voice.status, on change."""

    alert_id: str | None = None
    state: VoiceState


class VoiceReply(Model):
    """voice.reply, on event. intent NONE means no reply within listen_s."""

    alert_id: str
    transcript: str
    intent: Intent
    latency_ms: int = Field(ge=0)
    path: Literal["keyword", "llm"]


# Core (lead)


class VoiceSay(Model):
    """voice.say, on event."""

    alert_id: str
    text: str
    tier: Tier
    lang: Lang
    listen_s: float = Field(default=8.0, ge=0.0)
    audio_key: str | None = None


class AlertState(Model):
    """alert.state, on every status change."""

    alert_id: str
    type: EventType
    tier: Tier
    status: AlertStatus
    text: str
    reply: str | None = None
    intent: Intent | None = None
    latency_ms: int | None = Field(default=None, ge=0)


class VehicleMotion(Model):
    """vehicle.motion, 10 Hz. source "replay" (UAH-DriveSet) or "sim" (core --fake)."""

    accel_long_g: float
    accel_lat_g: float
    accel_mag_g: float = Field(ge=0.0)
    speed_kmh: float = Field(ge=0.0)
    lat: float
    lon: float
    road: Literal["motorway", "secondary"]
    source: Literal["replay", "sim"]


class ControlMusic(Model):
    """control.music, on event."""

    duck: bool
    level: Prob


class CheckinSchedule(Model):
    """checkin.schedule, on change."""

    interval_min: float = Field(gt=0.0)
    next_checkin_ts: float
    next_break_ts: float | None = None


class TripSummary(Model):
    """trip.summary, at trip end."""

    text: str
    stats: dict[str, Any] = Field(default_factory=dict)


# UI and integrations


class MusicVolume(Model):
    """music.volume, on change."""

    volume: float = Field(ge=0.0, le=100.0)
    playing: bool


class BioHR(Model):
    """bio.hr, 1 Hz."""

    bpm: float = Field(gt=0.0)
    rr_ms: list[float] = Field(default_factory=list)
    source: Literal["suunto", "strap", "sim"]


class ContextRoad(Model):
    """context.road, 1 Hz."""

    load: RoadLoad
    traffic: str | None = None
    next_maneuver: str | None = None
    dist_m: float | None = Field(default=None, ge=0.0)


class Pretrip(Model):
    """Pre-trip form (CI-01). Departure time is the control.trip envelope ts."""

    planned_min: float = Field(gt=0.0)
    sleep_h: float = Field(ge=0.0, le=24.0)
    awake_h: float = Field(ge=0.0)
    tiredness: int = Field(ge=1, le=5)
    caffeine: bool = Field(description="caffeine in the last 2 h")
    engagement: Literal["trivia", "fact", "quiet"]
    lang: Lang
    emergency_contact: str = Field(description="phone number, E.164")
    resting_hr: float = Field(gt=0.0)


class ControlTrip(Model):
    """control.trip, on event. pretrip is required on start."""

    action: Literal["start", "stop"]
    pretrip: Pretrip | None = None


class ControlReplay(Model):
    """control.replay, on event. trip picks the replay trip; segment is the seek position."""

    action: Literal["play", "pause", "seek"]
    trip: str | None = None
    segment: float | None = Field(default=None, ge=0.0, description="seconds from trip start")


class ControlInject(Model):
    """control.inject, on event."""

    event: Literal["harsh_brake", "crash"]


# Topic registry


class TopicSpec(BaseModel):
    model: type[Model]
    producers: tuple[Src, ...]
    rate_hz: float | None  # None = on event / on change


TOPICS: dict[str, TopicSpec] = {
    "driver.state": TopicSpec(model=DriverState, producers=("cv",), rate_hz=10),
    "camera.frame": TopicSpec(model=CameraFrame, producers=("cv",), rate_hz=5),
    "detector.event": TopicSpec(model=DetectorEvent, producers=("cv", "core"), rate_hz=None),
    "audio.features": TopicSpec(model=AudioFeatures, producers=("voice",), rate_hz=5),
    "voice.status": TopicSpec(model=VoiceStatus, producers=("voice",), rate_hz=None),
    "voice.reply": TopicSpec(model=VoiceReply, producers=("voice",), rate_hz=None),
    "voice.say": TopicSpec(model=VoiceSay, producers=("core",), rate_hz=None),
    "alert.state": TopicSpec(model=AlertState, producers=("core",), rate_hz=None),
    "vehicle.motion": TopicSpec(model=VehicleMotion, producers=("core",), rate_hz=10),
    "control.music": TopicSpec(model=ControlMusic, producers=("core",), rate_hz=None),
    "checkin.schedule": TopicSpec(model=CheckinSchedule, producers=("core",), rate_hz=None),
    "trip.summary": TopicSpec(model=TripSummary, producers=("core",), rate_hz=None),
    "music.volume": TopicSpec(model=MusicVolume, producers=("ui",), rate_hz=None),
    "bio.hr": TopicSpec(model=BioHR, producers=("hr", "ui"), rate_hz=1),
    "context.road": TopicSpec(model=ContextRoad, producers=("maps", "ui"), rate_hz=1),
    "control.trip": TopicSpec(model=ControlTrip, producers=("ui",), rate_hz=None),
    "control.replay": TopicSpec(model=ControlReplay, producers=("ui",), rate_hz=None),
    "control.inject": TopicSpec(model=ControlInject, producers=("ui",), rate_hz=None),
}

TOPIC_MODELS: dict[str, type[Model]] = {name: spec.model for name, spec in TOPICS.items()}


def validate(topic: str, data: dict[str, Any]) -> Model:
    """Validate a message's data against its topic model. Raises KeyError for unknown topics
    and pydantic.ValidationError for bad data."""
    try:
        model = TOPIC_MODELS[topic]
    except KeyError:
        raise KeyError(f"unknown topic {topic!r}") from None
    return model.model_validate(data)


def validate_envelope(envelope: dict[str, Any]) -> Model:
    """Validate a whole envelope and its data; returns the data model."""
    env = Envelope.model_validate(envelope)
    return validate(env.topic, env.data)
