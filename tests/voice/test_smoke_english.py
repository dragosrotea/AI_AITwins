import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.voice import smoke_english


@pytest.mark.parametrize(
    ("mode", "short_reply", "intent", "path", "transcript", "passes"),
    [
        ("silence", None, "NONE", "keyword", "", True),
        ("silence", None, "OK", "keyword", "Okay.", False),
        ("silence", None, "UNCLEAR", "llm", "", False),
        ("keyword", "Yes", "OK", "keyword", "Yes.", True),
        ("keyword", "No", "NOT_OK", "keyword", "No.", True),
    ],
)
async def test_smoke_checks_expected_reply(
    monkeypatch, mode, short_reply, intent, path, transcript, passes
):
    alert_id = "en-smoke-12345678"
    replies = [
        {
            "topic": "voice.reply",
            "data": {
                "alert_id": alert_id,
                "intent": intent,
                "path": path,
                "transcript": transcript,
                "latency_ms": 0 if intent == "NONE" else 1500,
            },
        },
        {"topic": "voice.status", "data": {"alert_id": alert_id, "state": "idle"}},
    ]
    websocket = MagicMock(send=AsyncMock())
    websocket.__aiter__.return_value = [json.dumps(reply) for reply in replies]
    connection = MagicMock()
    connection.__aenter__.return_value = websocket
    monkeypatch.setattr(smoke_english.websockets, "connect", lambda _url: connection)
    monkeypatch.setattr(smoke_english, "uuid4", lambda: SimpleNamespace(hex="12345678"))
    if passes:
        await smoke_english.probe(mode, short_reply=short_reply)
    else:
        with pytest.raises(SystemExit, match="Expected NONE / keyword"):
            await smoke_english.probe(mode, short_reply=short_reply)
