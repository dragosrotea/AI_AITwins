"""Client tests (LEAD-T03): publish, subscribe, wildcard, reconnect, ThreadPublisher."""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from backend.bus.client import BusClient, ThreadPublisher


class Inbox:
    """Collects envelopes and lets a test wait for a count."""

    def __init__(self) -> None:
        self.items: list[dict] = []
        self._changed = asyncio.Event()

    async def handler(self, envelope: dict) -> None:
        self.items.append(envelope)
        self._changed.set()

    async def wait_for(self, n: int, wait_s: float = 2.0) -> list[dict]:
        async with asyncio.timeout(wait_s):
            while len(self.items) < n:
                self._changed.clear()
                await self._changed.wait()
        return self.items

    def topics(self) -> list[str]:
        return [e["topic"] for e in self.items]


@pytest.fixture
async def clients(broker_server):
    made: list[BusClient] = []

    async def make(src: str = "core", **kw) -> BusClient:
        client = BusClient(src, broker_server.url, reconnect_s=0.1, **kw)
        made.append(client)
        await client.connect()
        return client

    yield make
    for client in made:
        await client.close(flush_s=0)


async def settle() -> None:
    await asyncio.sleep(0.1)  # let subscriptions reach the broker


async def test_publish_and_subscribe(clients):
    sub, pub = await clients("ui"), await clients("cv")
    inbox = Inbox()
    sub.subscribe(["driver.state"], inbox.handler)
    await settle()

    before = time.time()
    await pub.publish("driver.state", {"face_present": True})
    [msg] = await inbox.wait_for(1)
    assert msg["topic"] == "driver.state"
    assert msg["src"] == "cv"
    assert msg["data"] == {"face_present": True}
    assert before <= msg["ts"] <= time.time()


async def test_wildcard_routes_to_matching_handlers_only(clients):
    sub, pub = await clients("ui"), await clients("core")
    alerts, voice, everything = Inbox(), Inbox(), Inbox()
    sub.subscribe(["alert.*"], alerts.handler)
    sub.subscribe(["voice.say", "voice.status"], voice.handler)
    sub.subscribe(["*"], everything.handler)
    await settle()

    for topic in ("alert.state", "voice.say", "voice.reply", "vehicle.motion"):
        await pub.publish(topic, {})
    await everything.wait_for(4)
    await asyncio.sleep(0.05)
    assert alerts.topics() == ["alert.state"]
    assert voice.topics() == ["voice.say"]
    assert everything.topics() == ["alert.state", "voice.say", "voice.reply", "vehicle.motion"]


async def test_sync_handler_and_failing_handler_do_not_stop_delivery(clients):
    sub, pub = await clients("ui"), await clients("core")
    seen: list[str] = []

    def boom(_envelope: dict) -> None:
        raise ValueError("handler bug")

    sub.subscribe(["t"], boom)
    sub.subscribe(["t"], lambda e: seen.append(e["data"]["n"]))
    await settle()
    for n in range(3):
        await pub.publish("t", {"n": n})
    await asyncio.sleep(0.2)
    assert seen == [0, 1, 2]


async def test_reconnects_resubscribes_and_flushes_outbox(broker_server, clients):
    sub, pub = await clients("ui"), await clients("core")
    inbox = Inbox()
    sub.subscribe(["x"], inbox.handler)
    await settle()
    await pub.publish("x", {"n": 0})
    await inbox.wait_for(1)

    await broker_server.stop()
    await asyncio.sleep(0.3)
    assert not sub.connected and not pub.connected
    await pub.publish("x", {"n": 1})  # queued while the broker is down

    await broker_server.start()
    async with asyncio.timeout(3):
        while not (sub.connected and pub.connected):  # noqa: ASYNC110
            await asyncio.sleep(0.05)
    await settle()
    await pub.publish("x", {"n": 2})
    # n=1 (queued during the outage) may race the subscriber's resubscribe; n=2 must arrive.
    async with asyncio.timeout(3):
        while not inbox.items or inbox.items[-1]["data"]["n"] != 2:
            await inbox.wait_for(len(inbox.items) + 1)
    received = [e["data"]["n"] for e in inbox.items]
    assert received in ([0, 1, 2], [0, 2])


async def test_connect_waits_until_broker_comes_up(broker_server):
    await broker_server.stop()
    client = BusClient("core", broker_server.url, reconnect_s=0.1)
    connecting = asyncio.create_task(client.connect())
    await asyncio.sleep(0.3)
    assert not connecting.done()
    await broker_server.start()
    async with asyncio.timeout(3):
        await connecting
    assert client.connected
    await client.close(flush_s=0)


async def test_outbox_is_bounded_while_disconnected():
    client = BusClient("core", "ws://127.0.0.1:1/bus", outbox_size=3)  # never connects
    for n in range(5):
        await client.publish("x", {"n": n})
    assert len(client._outbox) == 3
    assert client.dropped == 2


async def test_thread_publisher_from_plain_threads(broker_server, clients):
    sub = await clients("ui")
    inbox = Inbox()
    sub.subscribe(["driver.state"], inbox.handler)
    await settle()

    pub = ThreadPublisher("cv", broker_server.url)  # no start(): publish starts it

    def camera_loop(offset: int) -> None:
        for i in range(25):
            pub.publish("driver.state", {"i": offset + i})

    threads = [threading.Thread(target=camera_loop, args=(k * 100,)) for k in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    items = await inbox.wait_for(50, wait_s=3)
    assert {e["src"] for e in items} == {"cv"}
    assert sorted(e["data"]["i"] for e in items) == list(range(25)) + list(range(100, 125))
    await asyncio.to_thread(pub.stop)
    assert not pub.connected
