"""DriveMate core process.

    python -m backend.core.run --fake

--fake (LEAD-T06) publishes a synthetic drive on vehicle.motion at 10 Hz (source "sim") and
cycles one alert every core.fake.alert_every_s through queued, speaking, listening and closed,
with voice.say, control.music and checkin.schedule, so voice and the dashboard have a partner.
A voice.reply for the active alert closes it; otherwise it closes after listen_s + reply_grace_s.
The live core (vehicle replay, Alert Manager) arrives with LEAD-T08 and LEAD-T10.
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

from backend.bus.client import BusClient
from backend.config import get_config

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
        period = 1.0 / self.rate_hz
        started = time.monotonic()
        next_tick = started
        while True:
            await self.bus.publish(
                "vehicle.motion", vehicle_sample(time.monotonic() - started, self.fake["speed_kmh"])
            )
            next_tick += period
            await asyncio.sleep(max(0.0, next_tick - time.monotonic()))

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


def main() -> None:
    parser = argparse.ArgumentParser(description="DriveMate core process")
    parser.add_argument("--fake", action="store_true", help="synthetic vehicle and alert cycle")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    if not args.fake:
        parser.exit(2, "core live mode arrives with LEAD-T08/T10; run with --fake for now.\n")
    try:
        asyncio.run(FakeCore(BusClient("core"), get_config()).run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
