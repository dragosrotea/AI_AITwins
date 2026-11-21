"""Shared fixtures: a real broker on a free local port."""

from __future__ import annotations

import asyncio
import socket

import pytest
import uvicorn

from backend.bus.broker import create_app


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class BrokerServer:
    """A broker that tests can stop and start again on the same port."""

    def __init__(self, port: int) -> None:
        self.port = port
        self.addr = f"127.0.0.1:{port}"
        self.url = f"ws://{self.addr}/bus"
        self.broker = None
        self._server: uvicorn.Server | None = None
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        app = create_app(queue_size=200)
        self.broker = app.state.broker
        config = uvicorn.Config(app, host="127.0.0.1", port=self.port, log_level="warning")
        self._server = uvicorn.Server(config)
        self._task = asyncio.create_task(self._server.serve())
        while not self._server.started:  # noqa: ASYNC110 (uvicorn exposes no started event)
            await asyncio.sleep(0.01)

    async def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
            await self._task
            self._server = None


@pytest.fixture
async def broker_server():
    server = BrokerServer(free_port())
    await server.start()
    yield server
    await server.stop()


@pytest.fixture
async def broker_url(broker_server):
    yield broker_server.addr, broker_server.broker
