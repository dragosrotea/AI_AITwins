from __future__ import annotations

import asyncio
import copy
from typing import Any

from backend.config import load_config
from backend.core.run import FakeCore, vehicle_sample
from backend.schemas import validate


class RecordingBus:
    url = "memory://"

    def __init__(self) -> None:
        self.sent: list[tuple[str, dict[str, Any]]] = []
        self.handlers: list[Any] = []

    def subscribe(self, topics: list[str], handler: Any) -> None:
        self.handlers.append(handler)

    async def publish(self, topic: str, data: dict[str, Any]) -> None:
        validate(topic, data)  # every fake message must satisfy the contract
        self.sent.append((topic, data))


def fast_config() -> dict[str, Any]:
    cfg = copy.deepcopy(load_config("default"))
    cfg["core"]["fake"]["speak_s"] = 0
    cfg["core"]["alerts"]["listen_s"] = 0.05
    cfg["core"]["alerts"]["reply_grace_s"] = 0
    return cfg


def statuses(bus: RecordingBus) -> list[str]:
    return [d["status"] for t, d in bus.sent if t == "alert.state"]


def test_vehicle_samples_match_schema() -> None:
    for i in range(500):
        validate("vehicle.motion", vehicle_sample(i / 10, 100))
    assert vehicle_sample(0.5, 100)["accel_long_g"] < -0.4  # harsh brake marker


def test_alert_without_reply_closes_with_none() -> None:
    bus = RecordingBus()
    asyncio.run(FakeCore(bus, fast_config()).one_alert("distraction", "T2", "Hi?"))
    assert statuses(bus) == ["queued", "speaking", "listening", "closed"]
    assert bus.sent[-2][1]["intent"] == "NONE"
    assert [d["duck"] for t, d in bus.sent if t == "control.music"] == [True, False]


def test_reply_closes_alert() -> None:
    cfg = fast_config()
    cfg["core"]["alerts"]["listen_s"] = 5

    async def scenario() -> RecordingBus:
        bus = RecordingBus()
        core = FakeCore(bus, cfg)
        task = asyncio.create_task(core.one_alert("drowsiness", "T2", "Tired?"))
        while not core.replies:  # noqa: ASYNC110 (test polls the fake core)
            await asyncio.sleep(0)
        alert_id = next(iter(core.replies))
        reply = {"alert_id": alert_id, "transcript": "yes", "intent": "OK", "latency_ms": 300}
        core.on_reply({"data": reply})
        await asyncio.wait_for(task, 1)
        return bus

    bus = asyncio.run(scenario())
    closed = [d for t, d in bus.sent if t == "alert.state"][-1]
    assert closed["status"] == "closed" and closed["intent"] == "OK" and closed["latency_ms"] == 300
