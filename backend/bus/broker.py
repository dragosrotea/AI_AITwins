"""DriveMate message broker (LEAD-T02).

WebSocket pub/sub at ws://<host>:<port>/bus plus per-topic rates at GET /stats.

Client -> broker:
    {"op": "sub", "topics": ["driver.state", "alert.*"]}   trailing * = prefix match, "*" = all
    {"op": "unsub", "topics": [...]}
    {"op": "pub", "topic": ..., "ts": ..., "src": ..., "data": {...}}
Broker -> client: the envelope without "op".

Every client has a bounded outbox; when it is full the oldest message is dropped, so a slow
consumer (e.g. the dashboard) never blocks a fast producer (e.g. the camera).

Run: python -m backend.bus.broker [--host 127.0.0.1] [--port 8765] [--queue-size 200]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from collections import defaultdict, deque
from itertools import count

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from backend.bus.topics import topic_matches

log = logging.getLogger("bus.broker")

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
DEFAULT_QUEUE_SIZE = 200
RATE_WINDOW_S = 5.0


class Client:
    """One WebSocket connection: its subscriptions and a bounded, drop-oldest outbox."""

    def __init__(self, cid: int, ws: WebSocket, queue_size: int) -> None:
        self.cid = cid
        self.ws = ws
        self.patterns: set[str] = set()
        self.outbox: deque[str] = deque(maxlen=queue_size)
        self.wakeup = asyncio.Event()
        self.dropped = 0
        self.src: str | None = None  # learned from the first published envelope

    def wants(self, topic: str) -> bool:
        return any(topic_matches(p, topic) for p in self.patterns)

    def enqueue(self, text: str) -> None:
        if len(self.outbox) == self.outbox.maxlen:
            self.dropped += 1  # deque drops the oldest item on append
        self.outbox.append(text)
        self.wakeup.set()

    async def send_loop(self) -> None:
        try:
            while True:
                await self.wakeup.wait()
                self.wakeup.clear()
                while self.outbox:
                    await self.ws.send_text(self.outbox.popleft())
        except asyncio.CancelledError:
            raise
        except Exception:  # socket closed mid-send; the receive loop cleans up
            log.debug("client %s send loop stopped", self.cid)


class TopicRates:
    """Message counts per topic over a sliding window."""

    def __init__(self, window_s: float = RATE_WINDOW_S) -> None:
        self.window_s = window_s
        self.stamps: dict[str, deque[float]] = defaultdict(deque)
        self.totals: dict[str, int] = defaultdict(int)

    def add(self, topic: str, now: float) -> None:
        self.stamps[topic].append(now)
        self.totals[topic] += 1
        self._prune(topic, now)

    def _prune(self, topic: str, now: float) -> None:
        stamps = self.stamps[topic]
        while stamps and stamps[0] < now - self.window_s:
            stamps.popleft()

    def snapshot(self, now: float) -> dict[str, dict[str, float]]:
        out = {}
        for topic in sorted(self.totals):
            self._prune(topic, now)
            out[topic] = {
                "rate_hz": round(len(self.stamps[topic]) / self.window_s, 2),
                "total": self.totals[topic],
            }
        return out


class Broker:
    def __init__(self, queue_size: int = DEFAULT_QUEUE_SIZE) -> None:
        self.queue_size = queue_size
        self.clients: dict[int, Client] = {}
        self.rates = TopicRates()
        self.started = time.time()
        self._ids = count(1)

    def publish(self, envelope: dict) -> None:
        topic = envelope["topic"]
        text = json.dumps(envelope, separators=(",", ":"))
        self.rates.add(topic, time.monotonic())
        for client in self.clients.values():
            if client.wants(topic):
                client.enqueue(text)

    def handle(self, client: Client, raw: str) -> None:
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            log.warning("client %s sent invalid JSON", client.cid)
            return
        if not isinstance(msg, dict):
            return
        op = msg.pop("op", None)
        if op == "pub":
            if not isinstance(msg.get("topic"), str) or not msg["topic"]:
                log.warning("client %s published without a topic", client.cid)
                return
            msg.setdefault("ts", time.time())
            if client.src is None and isinstance(msg.get("src"), str):
                client.src = msg["src"]
            self.publish(msg)
        elif op in ("sub", "unsub"):
            topics = msg.get("topics")
            if not isinstance(topics, list):
                return
            patterns = {t for t in topics if isinstance(t, str) and t}
            if op == "sub":
                client.patterns |= patterns
            else:
                client.patterns -= patterns
        else:
            log.warning("client %s sent unknown op %r", client.cid, op)

    async def serve(self, ws: WebSocket) -> None:
        await ws.accept()
        client = Client(next(self._ids), ws, self.queue_size)
        self.clients[client.cid] = client
        log.info("client %s connected (%d total)", client.cid, len(self.clients))
        sender = asyncio.create_task(client.send_loop())
        try:
            while True:
                raw = await ws.receive_text()
                self.handle(client, raw)
                # receive_text() does not yield while frames are buffered; yield so the
                # subscribers' send loops drain before a burst overflows their outboxes.
                await asyncio.sleep(0)
        except WebSocketDisconnect:
            pass
        except Exception:  # one bad client must never take the broker down
            log.exception("client %s failed", client.cid)
        finally:
            sender.cancel()
            self.clients.pop(client.cid, None)
            log.info("client %s left (%d total)", client.cid, len(self.clients))

    def stats(self) -> dict:
        return {
            "uptime_s": round(time.time() - self.started, 1),
            "window_s": self.rates.window_s,
            "topics": self.rates.snapshot(time.monotonic()),
            "clients": [
                {
                    "id": c.cid,
                    "src": c.src,
                    "subs": sorted(c.patterns),
                    "queued": len(c.outbox),
                    "dropped": c.dropped,
                }
                for c in self.clients.values()
            ],
        }


def create_app(queue_size: int = DEFAULT_QUEUE_SIZE) -> FastAPI:
    broker = Broker(queue_size)
    app = FastAPI(title="DriveMate bus")
    app.state.broker = broker
    # The dashboard dev server runs on another port and fetches /stats.
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"])

    @app.websocket("/bus")
    async def bus(ws: WebSocket) -> None:
        await broker.serve(ws)

    @app.get("/stats")
    async def stats() -> dict:
        return broker.stats()

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="DriveMate WebSocket broker")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--queue-size", type=int, default=DEFAULT_QUEUE_SIZE)
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(name)s %(message)s")
    uvicorn.run(
        create_app(args.queue_size),
        host=args.host,
        port=args.port,
        log_level="warning",
        ws_max_size=4 * 1024 * 1024,  # camera.frame JPEGs
    )


if __name__ == "__main__":
    main()
