"""DriveMate bus client (LEAD-T03).

Async code (core, voice):

    bus = BusClient("core")
    await bus.connect()                       # waits for the first connection
    bus.subscribe(["driver.state", "alert.*"], handler)   # handler(envelope), sync or async
    await bus.publish("voice.say", {...})

Plain threads (camera loop, audio callback):

    pub = ThreadPublisher("cv")               # start() is optional
    pub.publish("driver.state", {...})        # thread-safe, never blocks
    pub.stop()

The client reconnects every second while the broker is down and re-sends its subscriptions.
Messages published while disconnected wait in a bounded outbox (oldest dropped).
Handlers run one at a time in arrival order: keep them quick and spawn a task for slow work
(LLM calls, speech), or every other message waits behind it.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import threading
import time
from collections import deque
from collections.abc import Awaitable, Callable
from typing import Any

import websockets

from backend.bus.topics import topic_matches

log = logging.getLogger("bus.client")

DEFAULT_URL = "ws://localhost:8765/bus"
RECONNECT_S = 1.0
OUTBOX_SIZE = 200
MAX_MESSAGE_BYTES = 4 * 1024 * 1024  # camera.frame JPEGs

Envelope = dict[str, Any]
Handler = Callable[[Envelope], Awaitable[None] | None]


def bus_url() -> str:
    return os.environ.get("DRIVEMATE_BUS_URL", DEFAULT_URL)


class BusClient:
    def __init__(
        self,
        src: str,
        url: str | None = None,
        reconnect_s: float = RECONNECT_S,
        outbox_size: int = OUTBOX_SIZE,
    ) -> None:
        self.src = src
        self.url = url or bus_url()
        self.reconnect_s = reconnect_s
        self.dropped = 0
        self._subs: list[tuple[list[str], Handler]] = []
        self._outbox: deque[str] = deque(maxlen=outbox_size)
        self._wake: asyncio.Event | None = None
        self._connected: asyncio.Event | None = None
        self._ws: Any = None
        self._runner: asyncio.Task | None = None
        self._closing = False

    @property
    def connected(self) -> bool:
        return self._ws is not None

    async def connect(self, wait: bool = True) -> None:
        """Start the background connection; by default wait until the broker answers."""
        if self._runner is None:
            self._wake = asyncio.Event()
            self._connected = asyncio.Event()
            self._closing = False
            self._runner = asyncio.create_task(self._run(), name=f"bus-{self.src}")
        if wait:
            await self._connected.wait()

    def subscribe(self, topics: list[str], handler: Handler) -> None:
        self._subs.append((list(topics), handler))
        if self._ws is not None:
            asyncio.create_task(self._send_sub(self._ws, topics))

    async def publish(self, topic: str, data: dict, ts: float | None = None) -> None:
        self.publish_nowait(topic, data, ts)

    def publish_nowait(self, topic: str, data: dict, ts: float | None = None) -> None:
        """Queue a message; must be called on the client's event loop."""
        envelope = {"topic": topic, "ts": time.time() if ts is None else ts, "src": self.src}
        msg = {"op": "pub", **envelope, "data": data}
        if len(self._outbox) == self._outbox.maxlen:
            self.dropped += 1
        self._outbox.append(json.dumps(msg, separators=(",", ":")))
        if self._wake is not None:
            self._wake.set()

    async def close(self, flush_s: float = 1.0) -> None:
        """Send what is queued (up to flush_s), then disconnect."""
        if self._runner is None:
            return
        deadline = time.monotonic() + flush_s
        while self._outbox and self._ws is not None and time.monotonic() < deadline:  # noqa: ASYNC110
            await asyncio.sleep(0.01)
        self._closing = True
        self._runner.cancel()
        try:
            await self._runner
        except asyncio.CancelledError:
            pass
        self._runner = None

    # Internals

    async def _run(self) -> None:
        warned = False
        while not self._closing:
            try:
                async with websockets.connect(
                    self.url, max_size=MAX_MESSAGE_BYTES, close_timeout=1
                ) as ws:
                    for topics, _ in self._subs:
                        await self._send_sub(ws, topics)
                    self._ws = ws
                    self._connected.set()
                    log.info("%s connected to %s", self.src, self.url)
                    warned = False
                    tasks = {
                        asyncio.create_task(self._read_loop(ws)),
                        asyncio.create_task(self._write_loop(ws)),
                    }
                    try:
                        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                    finally:
                        for task in tasks:
                            task.cancel()
                        await asyncio.gather(*tasks, return_exceptions=True)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # broker down, refused, handshake failed...
                if not warned:
                    log.warning("%s cannot reach %s (%s); retrying", self.src, self.url, exc)
                    warned = True
            finally:
                if self._ws is not None and not self._closing:
                    log.warning("%s disconnected from the broker; reconnecting", self.src)
                self._ws = None
                self._connected.clear()
            if not self._closing:
                await asyncio.sleep(self.reconnect_s)

    @staticmethod
    async def _send_sub(ws: Any, topics: list[str]) -> None:
        try:
            await ws.send(json.dumps({"op": "sub", "topics": list(topics)}))
        except Exception:
            pass  # the reconnect re-sends every subscription

    async def _write_loop(self, ws: Any) -> None:
        while True:
            while self._outbox:
                await ws.send(self._outbox[0])
                self._outbox.popleft()  # only after a successful send
            self._wake.clear()
            await self._wake.wait()

    async def _read_loop(self, ws: Any) -> None:
        try:
            async for raw in ws:
                try:
                    envelope = json.loads(raw)
                    topic = envelope["topic"]
                except (ValueError, KeyError, TypeError):
                    continue
                await self._dispatch(topic, envelope)
        except websockets.ConnectionClosed:
            pass

    async def _dispatch(self, topic: str, envelope: Envelope) -> None:
        for topics, handler in self._subs:
            if not any(topic_matches(p, topic) for p in topics):
                continue
            try:
                result = handler(envelope)
                if inspect.isawaitable(result):
                    await result
            except Exception:
                log.exception("%s handler for %s failed", self.src, topic)


class ThreadPublisher:
    """Publish from normal threads. Runs a BusClient on its own loop in a daemon thread."""

    def __init__(self, src: str, url: str | None = None) -> None:
        self.src = src
        self.url = url
        self._loop: asyncio.AbstractEventLoop | None = None
        self._client: BusClient | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._lock = threading.Lock()

    @property
    def connected(self) -> bool:
        return self._client is not None and self._client.connected

    def start(self) -> None:
        """Idempotent; every caller returns only once the loop is running."""
        with self._lock:
            if self._thread is None:
                self._ready.clear()
                self._thread = threading.Thread(
                    target=self._main, name=f"bus-{self.src}", daemon=True
                )
                self._thread.start()
        self._ready.wait()

    def publish(self, topic: str, data: dict) -> None:
        if not self._ready.is_set():
            self.start()
        ts = time.time()  # stamp in the caller's thread, not when the loop gets to it
        self._loop.call_soon_threadsafe(self._client.publish_nowait, topic, data, ts)

    def stop(self, flush_s: float = 1.0) -> None:
        if self._thread is None or self._loop is None:
            return
        future = asyncio.run_coroutine_threadsafe(self._client.close(flush_s), self._loop)
        try:
            future.result(timeout=flush_s + 2)
        except Exception:
            pass
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=2)
        with self._lock:
            self._ready.clear()
            self._thread = None

    def _main(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        self._client = BusClient(self.src, self.url)
        loop.run_until_complete(self._client.connect(wait=False))
        self._ready.set()
        try:
            loop.run_forever()
        finally:
            loop.close()
