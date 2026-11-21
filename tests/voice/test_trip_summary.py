import asyncio
import json
from unittest.mock import AsyncMock, Mock

import pytest

from backend.core import run
from backend.core.triplog import TripLog
from backend.llm import client, summarize_trip
from backend.schemas import TripSummary


def record(topic, ts, **data):
    return {
        "topic": topic,
        "src": "ui" if topic == "control.trip" else "core",
        "ts": ts,
        "data": data,
    }


ALERTS = [
    record("log.alert_created", 110, alert_id="a", type="drowsiness"),
    record("alert.state", 111, alert_id="a", type="drowsiness", status="speaking"),
    record("voice.reply", 112, alert_id="a", latency_ms=750, intent="OK", transcript="private"),
    record(
        "alert.state",
        113,
        alert_id="a",
        type="drowsiness",
        status="closed",
        latency_ms=750,
        intent="OK",
    ),
    record("alert.state", 120, alert_id="b", type="phone", status="closed", intent="NONE"),
    record("voice.reply", 121, alert_id="b", latency_ms=0, intent="NONE"),
    record(
        "alert.state", 130, alert_id="c", type="phone", status="closed", latency_ms=920, intent="OK"
    ),
    record("voice.reply", 131, alert_id="unknown", latency_ms=50, intent="OK"),
]


@pytest.mark.parametrize(
    ("records", "provider_output"), [(ALERTS, "[2, 1]"), ([], "[1]"), (ALERTS, None)]
)
async def test_trip_end_publishes_and_saves_summary(
    tmp_path, monkeypatch, records, provider_output
):
    topics = ["control.trip", "alert.state", "voice.reply", "trip.summary"]
    cfg = {"dir": "logs", "topics": topics}
    triplog = TripLog(cfg, root=tmp_path)
    monkeypatch.setattr(run, "TripLog", lambda _cfg: triplog)
    provider = AsyncMock(return_value=provider_output)
    monkeypatch.setattr(client, "request_text", provider)
    published = asyncio.get_running_loop().create_future()
    bus = Mock(publish_nowait=Mock(side_effect=lambda *args: published.set_result(args)))
    run.attach_triplog(bus, {"core": {"triplog": cfg}})
    subscribed, on_record = bus.subscribe.call_args.args
    assert "trip.summary" not in subscribed

    on_record(record("control.trip", 100, action="start"))
    for entry in records:
        on_record(entry)
    ended_path = triplog.path
    on_record(record("control.trip", 235, action="stop"))
    on_record(record("control.trip", 240, action="start"))
    topic, payload = await asyncio.wait_for(published, 3)
    assert topic == "trip.summary"
    summary = TripSummary.model_validate(payload)
    assert "Time driven: 2 min 15 sec." in summary.text
    assert summary.stats["duration_s"] == 135
    if records:
        assert summary.stats["alerts_by_type"] == {"drowsiness": 1, "phone": 2}
        assert summary.stats["response_times_ms"] == [750, 920]
        assert "750, 920 ms" in summary.text
        assert "Plan a rest stop" in summary.text and "Do Not Disturb" in summary.text
        assert (summary.text.index("Plan a rest stop") < summary.text.index("Do Not Disturb")) == (
            provider_output is not None
        )
    else:
        assert summary.stats["alerts_by_type"] == {} and summary.stats["response_times_ms"] == []
        assert "No alerts recorded. No response times recorded." in summary.text

    request = provider.await_args.args[0]
    assert json.loads(request)["stats"] == summary.stats
    assert "private" not in request
    assert triplog.read_completed(ended_path)[-1]["data"] == payload
    assert [entry["topic"] for entry in triplog.read_completed(triplog.path)] == ["control.trip"]
    triplog.close()


async def test_provider_timeout_and_unusable_output(monkeypatch):
    async def slow(*_args, **_kwargs):
        await asyncio.sleep(1)

    monkeypatch.setattr(client, "request_text", slow)
    assert await summarize_trip([], timeout_s=0.05) is None
    monkeypatch.setattr(client, "request_text", AsyncMock(return_value="[99]"))
    assert await summarize_trip([]) is None
