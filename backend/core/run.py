"""DriveMate core process.

    python -m backend.core.run          # live: UAH-DriveSet replay on vehicle.motion (LEAD-T08)
    python -m backend.core.run --fake

--fake (LEAD-T06) publishes a synthetic drive on vehicle.motion at 10 Hz (source "sim") and
cycles one alert every core.fake.alert_every_s through queued, speaking, listening and closed,
with voice.say, control.music and checkin.schedule, so voice and the dashboard have a partner.
A voice.reply for the active alert closes it; otherwise it closes after listen_s + reply_grace_s.
Live mode replays UAH-DriveSet (backend/core/vehicle.py) and obeys control.replay; without the
dataset it falls back to the synthetic drive. The Alert Manager (LEAD-T10) runs alongside and CV
distraction events become T2 alerts (LEAD-T11, backend/core/events.py).
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import logging
import math
import random
import time
import uuid
from typing import Any

from pydantic import ValidationError

from backend.bus.client import BusClient
from backend.config import get_config
from backend.core.alerts import Alert, AlertManager
from backend.core.events import handle_detector_event
from backend.core.triplog import TripLog
from backend.core.vehicle import Injector, Player, load_trips
from backend.schemas import validate

log = logging.getLogger("core")

# Synthetic route start: UAH-DriveSet was recorded around Madrid.
START_LAT, START_LON = 40.4168, -3.7038

FAKE_ALERTS = [
    ("distraction", "T2", "Eyes back on the road, please. Are you with me?"),
    ("drowsiness", "T2", "You seem tired. Shall we plan a break soon?"),
    ("zoning_out", "T1", "Quick one: what is the capital of Portugal?"),
]


def vehicle_sample(t: float, speed_kmh: float) -> dict[str, Any]:
    """Synthetic motion at time t (s): gentle speed and lateral sway, a harsh brake every 40 s."""
    speed = speed_kmh + 8 * math.sin(t / 20)
    long_g = 0.05 * math.sin(t / 3) + random.gauss(0, 0.01)
    if t % 40 < 1.5:
        long_g = -0.55
    lat_g = 0.12 * math.sin(t / 5) + random.gauss(0, 0.01)
    km = speed_kmh * t / 3600
    return {
        "accel_long_g": round(long_g, 3),
        "accel_lat_g": round(lat_g, 3),
        "accel_mag_g": round(math.hypot(long_g, lat_g), 3),
        "speed_kmh": round(max(speed, 0.0), 1),
        "lat": round(START_LAT + km / 111.0, 6),
        "lon": round(START_LON + km / 85.0, 6),
        "road": "motorway",
        "source": "sim",
    }


class FakeCore:
    def __init__(self, bus: BusClient, cfg: dict[str, Any]) -> None:
        self.bus = bus
        self.fake = cfg["core"]["fake"]
        self.alerts = cfg["core"]["alerts"]
        self.checkin = cfg["core"]["checkin"]
        self.rate_hz = cfg["core"]["vehicle"]["rate_hz"]
        self.injector = attach_injector(bus, cfg)
        self.triplog = attach_triplog(bus, cfg)
        self.replies: dict[str, asyncio.Future[dict[str, Any]]] = {}

    def on_reply(self, envelope: dict[str, Any]) -> None:
        data = envelope["data"]
        future = self.replies.get(data.get("alert_id", ""))
        if future is not None and not future.done():
            future.set_result(data)

    async def run(self) -> None:
        self.bus.subscribe(["voice.reply"], self.on_reply)
        await self.bus.connect()
        log.info("core --fake connected to %s", self.bus.url)
        async with asyncio.TaskGroup() as tasks:
            tasks.create_task(self.vehicle_loop())
            tasks.create_task(self.alert_loop())

    async def vehicle_loop(self) -> None:
        started = time.monotonic()
        speed = self.fake["speed_kmh"]
        await publish_at_rate(
            self.bus,
            self.rate_hz,
            lambda: self.injector.apply(vehicle_sample(time.monotonic() - started, speed)),
        )

    async def publish_schedule(self) -> None:
        interval = self.checkin["base_interval_min"]
        now = time.time()
        await self.bus.publish(
            "checkin.schedule",
            {
                "interval_min": interval,
                "next_checkin_ts": now + interval * 60,
                "next_break_ts": now + self.checkin["break_every_min"] * 60,
            },
        )

    async def alert_loop(self) -> None:
        for kind, tier, text in itertools.cycle(FAKE_ALERTS):
            await self.publish_schedule()
            await asyncio.sleep(self.fake["alert_every_s"])
            await self.one_alert(kind, tier, text)

    async def one_alert(self, kind: str, tier: str, text: str) -> None:
        alert_id = uuid.uuid4().hex[:8]
        base = {"alert_id": alert_id, "type": kind, "tier": tier, "text": text}
        listen_s = self.alerts["listen_s"]
        future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self.replies[alert_id] = future
        try:
            await self.bus.publish("alert.state", {**base, "status": "queued"})
            await self.bus.publish(
                "control.music", {"duck": True, "level": self.alerts["music_duck_volume"] / 100}
            )
            await self.bus.publish("alert.state", {**base, "status": "speaking"})
            await self.bus.publish(
                "voice.say",
                {
                    "alert_id": alert_id,
                    "text": text,
                    "tier": tier,
                    "lang": self.fake["lang"],
                    "listen_s": listen_s,
                },
            )
            await asyncio.sleep(self.fake["speak_s"])
            await self.bus.publish("alert.state", {**base, "status": "listening"})
            try:
                timeout = listen_s + self.alerts["reply_grace_s"]
                reply = await asyncio.wait_for(asyncio.shield(future), timeout)
            except TimeoutError:
                reply = {"transcript": None, "intent": "NONE", "latency_ms": None}
            await self.bus.publish(
                "alert.state",
                {
                    **base,
                    "status": "closed",
                    "reply": reply.get("transcript"),
                    "intent": reply.get("intent"),
                    "latency_ms": reply.get("latency_ms"),
                },
            )
            await self.bus.publish("control.music", {"duck": False, "level": 1.0})
        finally:
            self.replies.pop(alert_id, None)


def checked(topic: str, handler: Any) -> Any:
    """Wrap a bus handler so it only sees data that matches the contract."""

    def on_message(envelope: dict[str, Any]) -> None:
        try:
            validate(topic, envelope["data"])
        except (ValidationError, KeyError) as error:
            log.warning("ignored bad %s: %s", topic, error)
            return
        handler(envelope["data"])

    return on_message


def attach_injector(bus: BusClient, cfg: dict[str, Any]) -> Injector:
    """control.inject triggers events (LEAD-T09); any control.replay clears them (e.g. a crash)."""
    injector = Injector(cfg["core"]["inject"])
    on_inject = checked("control.inject", lambda data: injector.inject(data["event"]))
    bus.subscribe(["control.inject"], on_inject)
    bus.subscribe(["control.replay"], checked("control.replay", lambda d: injector.clear()))
    return injector


async def publish_at_rate(bus: BusClient, rate_hz: float, next_sample: Any) -> None:
    """Publish next_sample() on vehicle.motion at rate_hz without drifting."""
    period = 1.0 / rate_hz
    next_tick = time.monotonic()
    while True:
        await bus.publish("vehicle.motion", next_sample())
        next_tick += period
        await asyncio.sleep(max(0.0, next_tick - time.monotonic()))


def log_hook(name: str) -> Any:
    """Placeholder hook until the follow-up and emergency flows exist (T16, X3)."""

    def hook(alert: Alert, reply: dict[str, Any] | None) -> None:
        log.warning("%s hook: %s %s (reply %s)", name, alert.tier, alert.type, reply)

    return hook


def attach_triplog(bus: BusClient, cfg: dict[str, Any]) -> TripLog:
    """Trip log (LEAD-T12): every configured topic, as it went over the bus (AL-04)."""
    triplog = TripLog(cfg["core"]["triplog"])
    bus.subscribe(triplog.topics, triplog.write)
    return triplog


def alert_created(triplog: TripLog) -> Any:
    def note(alert: Alert) -> None:
        triplog.note(
            "alert_created",
            {
                "alert_id": alert.alert_id,
                "type": alert.type,
                "tier": alert.tier,
                "lang": alert.lang,
                "text": alert.text,
                "signals": alert.signals,
            },
        )

    return note


def attach_alerts(
    bus: BusClient, cfg: dict[str, Any], triplog: TripLog | None = None
) -> AlertManager:
    """Alert Manager (LEAD-T10) on the bus; core must run tick_loop() alongside."""
    manager = AlertManager(
        cfg["core"]["alerts"],
        bus.publish_nowait,
        on_follow_up=log_hook("follow-up"),
        on_emergency=log_hook("emergency"),
        on_alarm=log_hook("alarm"),
        on_submit=alert_created(triplog) if triplog else None,
    )
    bus.subscribe(["voice.status"], checked("voice.status", manager.on_voice_status))
    bus.subscribe(["voice.reply"], checked("voice.reply", manager.on_voice_reply))

    def on_trip(data: dict[str, Any]) -> None:
        if data["action"] == "start" and data.get("pretrip"):
            manager.lang = data["pretrip"]["lang"]
            log.info("driver language: %s", manager.lang)

    bus.subscribe(["control.trip"], checked("control.trip", on_trip))

    def on_detector_event(envelope: dict[str, Any]) -> None:
        # Needs the envelope (src), so it validates here instead of through checked().
        try:
            validate("detector.event", envelope["data"])
        except (ValidationError, KeyError) as error:
            log.warning("ignored bad detector.event from %s: %s", envelope.get("src"), error)
            return
        handle_detector_event(envelope, manager, cfg["core"]["routing"])  # LEAD-T11, T15

    bus.subscribe(["detector.event"], on_detector_event)
    return manager


async def tick_loop(manager: AlertManager, period_s: float = 0.1) -> None:
    while True:
        manager.tick()
        await asyncio.sleep(period_s)


async def run_live(bus: BusClient, cfg: dict[str, Any]) -> None:
    vehicle = cfg["core"]["vehicle"]
    alerts = attach_alerts(bus, cfg, attach_triplog(bus, cfg))
    injector = attach_injector(bus, cfg)
    trips = load_trips(vehicle)
    if trips:
        player = Player(trips, vehicle["default_trip"], vehicle["loop"])
        bus.subscribe(["control.replay"], checked("control.replay", player.control))

        def next_sample() -> dict[str, Any]:
            return injector.apply(player.step())
    else:
        log.warning("no UAH-DriveSet trips found; publishing the synthetic drive (source sim)")
        started = time.monotonic()
        speed = cfg["core"]["fake"]["speed_kmh"]

        def next_sample() -> dict[str, Any]:
            return injector.apply(vehicle_sample(time.monotonic() - started, speed))

    await bus.connect()
    log.info("core live connected to %s", bus.url)
    async with asyncio.TaskGroup() as tasks:
        tasks.create_task(publish_at_rate(bus, vehicle["rate_hz"], next_sample))
        tasks.create_task(tick_loop(alerts))


def main() -> None:
    parser = argparse.ArgumentParser(description="DriveMate core process")
    parser.add_argument("--fake", action="store_true", help="synthetic vehicle and alert cycle")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    cfg = get_config()
    try:
        if args.fake:
            asyncio.run(FakeCore(BusClient("core"), cfg).run())
        else:
            asyncio.run(run_live(BusClient("core"), cfg))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
