"""Broker tests (LEAD-T02): run a real uvicorn server on a free port."""

from __future__ import annotations

import asyncio
import json
import socket

import httpx
import pytest
import uvicorn
import websockets

from backend.bus.broker import Client, create_app, topic_matches


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
async def broker_url():
    port = free_port()
    app = create_app(queue_size=200)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    task = asyncio.create_task(server.serve())
    while not server.started:  # noqa: ASYNC110 (uvicorn exposes no started event)
        await asyncio.sleep(0.01)
    yield f"127.0.0.1:{port}", app.state.broker
    server.should_exit = True
    await task


async def connect(addr: str, topics: list[str] | None = None):
    ws = await websockets.connect(f"ws://{addr}/bus", close_timeout=0.2)
    if topics:
        await ws.send(json.dumps({"op": "sub", "topics": topics}))
    return ws


async def pub(ws, topic: str, data: dict | None = None, src: str = "core") -> None:
    msg = {"op": "pub", "topic": topic, "ts": 1.0, "src": src, "data": data or {}}
    await ws.send(json.dumps(msg))


async def recv(ws, wait_s: float = 1.0) -> dict:
    return json.loads(await asyncio.wait_for(ws.recv(), wait_s))


async def nothing(ws, wait_s: float = 0.2) -> bool:
    try:
        await asyncio.wait_for(ws.recv(), wait_s)
    except TimeoutError:
        return True
    return False


def test_topic_matches():
    assert topic_matches("driver.state", "driver.state")
    assert not topic_matches("driver.state", "driver.states")
    assert topic_matches("alert.*", "alert.state")
    assert not topic_matches("alert.*", "voice.say")
    assert topic_matches("*", "anything")


async def test_pub_sub_exact_and_wildcard(broker_url):
    addr, _ = broker_url
    exact = await connect(addr, ["driver.state"])
    wild = await connect(addr, ["voice.*"])
    sender = await connect(addr)
    await asyncio.sleep(0.05)  # let subscriptions land

    await pub(sender, "driver.state", {"fps": 10})
    await pub(sender, "voice.say", {"text": "hi"})

    got = await recv(exact)
    assert got == {"topic": "driver.state", "ts": 1.0, "src": "core", "data": {"fps": 10}}
    assert (await recv(wild))["topic"] == "voice.say"
    assert await nothing(exact)
    assert await nothing(wild)
    for ws in (exact, wild, sender):
        await ws.close()


async def test_unsub_and_bad_messages_are_ignored(broker_url):
    addr, _ = broker_url
    ws = await connect(addr, ["a.*"])
    await ws.send("not json")
    await ws.send(json.dumps({"op": "pub", "data": {}}))  # no topic
    await ws.send(json.dumps({"op": "bogus"}))
    await pub(ws, "a.x")
    assert (await recv(ws))["topic"] == "a.x"  # connection survived the junk
    await ws.send(json.dumps({"op": "unsub", "topics": ["a.*"]}))
    await pub(ws, "a.x")
    assert await nothing(ws)
    await ws.close()


async def test_disconnect_does_not_affect_others(broker_url):
    addr, broker = broker_url
    a = await connect(addr, ["t"])
    b = await connect(addr, ["t"])
    await asyncio.sleep(0.05)
    await a.close()
    await asyncio.sleep(0.05)
    assert len(broker.clients) == 1
    await pub(b, "t")
    assert (await recv(b))["topic"] == "t"
    await b.close()


async def test_stats_reports_rates_and_clients(broker_url):
    addr, _ = broker_url
    ws = await connect(addr)
    for _ in range(10):
        await pub(ws, "driver.state", src="cv")
    await asyncio.sleep(0.1)
    async with httpx.AsyncClient() as http:
        stats = (await http.get(f"http://{addr}/stats")).json()
    assert stats["topics"]["driver.state"]["total"] == 10
    assert stats["topics"]["driver.state"]["rate_hz"] == 10 / stats["window_s"]
    assert stats["clients"][0]["src"] == "cv"
    await ws.close()


class _StuckSocket:
    async def send_text(self, text: str) -> None:
        await asyncio.Event().wait()  # never completes


def test_slow_client_queue_is_bounded_and_drops_oldest():
    client = Client(1, _StuckSocket(), queue_size=5)  # type: ignore[arg-type]
    for i in range(12):
        client.enqueue(str(i))
    assert list(client.outbox) == ["7", "8", "9", "10", "11"]
    assert client.dropped == 7


async def test_slow_subscriber_does_not_block_fast_one(broker_url):
    addr, broker = broker_url
    slow = await connect(addr, ["x"])  # never reads
    fast = await connect(addr, ["x"])
    sender = await connect(addr)
    await asyncio.sleep(0.05)
    n = 300

    async def read_all() -> list[int]:
        return [(await recv(fast, wait_s=2.0))["data"]["i"] for _ in range(n)]

    reader = asyncio.create_task(read_all())
    for i in range(n):
        await pub(sender, "x", {"i": i})
    received = await reader
    assert received == list(range(n))
    for c in broker.clients.values():
        assert len(c.outbox) <= broker.queue_size
    for ws in (slow, fast, sender):
        await ws.close()
