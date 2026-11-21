"""LEAD-T12: trip log in JSONL (AL-04)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from backend.config import load_config
from backend.core.alerts import AlertManager
from backend.core.triplog import TripLog

CFG = load_config("default")["core"]


def env(topic: str, data: dict[str, Any], src: str = "core", ts: float = 1.0) -> dict[str, Any]:
    return {"topic": topic, "ts": ts, "src": src, "data": data}


def lines(path: Path | None) -> list[dict[str, Any]]:
    assert path is not None
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def make(tmp_path: Path, topics: list[str] | None = None) -> TripLog:
    cfg = dict(CFG["triplog"], **({"topics": topics} if topics else {}))
    return TripLog(cfg, root=tmp_path, clock=lambda: 1_763_740_800.0)


def test_writes_envelopes_in_order_and_flushes(tmp_path: Path) -> None:
    triplog = make(tmp_path)
    triplog.write(env("detector.event", {"type": "distraction", "score": 72}, src="cv"))
    triplog.write(env("voice.reply", {"intent": "OK", "latency_ms": 420, "transcript": "da"}))
    # Flushed per line: readable while the file is still open.
    got = lines(triplog.path)
    assert [g["topic"] for g in got] == ["detector.event", "voice.reply"]
    assert got[0]["src"] == "cv" and got[1]["data"]["latency_ms"] == 420
    assert triplog.path is not None
    assert triplog.path.parent == tmp_path / "logs" and triplog.path.name.startswith("trip_")


def test_camera_frames_are_never_written(tmp_path: Path) -> None:
    triplog = make(tmp_path, topics=["camera.frame", "alert.state"])
    assert "camera.frame" not in triplog.topics
    triplog.write(env("camera.frame", {"jpeg_b64": "xxxx"}, src="cv"))
    assert triplog.path is None  # nothing written, no file created
    assert not (tmp_path / "logs").exists()


def test_trip_start_and_stop_split_files(tmp_path: Path) -> None:
    triplog = make(tmp_path)
    triplog.write(env("alert.state", {"status": "closed"}))
    first = triplog.path
    triplog.write(env("control.trip", {"action": "start"}, src="ui"))
    second = triplog.path
    triplog.write(env("voice.say", {"text": "Ești bine?"}))
    triplog.write(env("control.trip", {"action": "stop"}, src="ui"))
    assert first != second
    assert [g["topic"] for g in lines(first)] == ["alert.state"]
    assert [g["topic"] for g in lines(second)] == ["control.trip", "voice.say", "control.trip"]
    assert lines(second)[1]["data"]["text"] == "Ești bine?"  # UTF-8 kept readable
    triplog.write(env("alert.state", {"status": "queued"}))
    assert triplog.path not in (first, second)  # after a stop, the next line starts a new file


def test_alert_created_note_carries_trigger_signals(tmp_path: Path) -> None:
    """AL-04: the trigger signals of each alert are logged (alert.state does not carry them)."""
    triplog = make(tmp_path)

    def note(alert: Any) -> None:
        triplog.note("alert_created", {"alert_id": alert.alert_id, "signals": alert.signals})

    def publish(topic: str, data: dict[str, Any]) -> None:
        triplog.write(env(topic, data))

    manager = AlertManager(CFG["alerts"], publish, clock=lambda: 0.0, on_submit=note)
    alert = manager.submit("distraction", "T2", "Ești cu mine?", signals={"yaw": 41.0})
    assert alert is not None
    got = lines(triplog.path)
    assert got[0]["topic"] == "log.alert_created"
    assert got[0]["data"] == {"alert_id": alert.alert_id, "signals": {"yaw": 41.0}}
    assert [g["data"].get("status") for g in got if g["topic"] == "alert.state"] == [
        "queued",
        "speaking",
    ]
