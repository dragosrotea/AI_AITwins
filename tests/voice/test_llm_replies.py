import asyncio
import json
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from backend.llm import classify_reply, client, generate_alert, replies


@pytest.fixture
def provider(monkeypatch):
    config = {
        "LLM_PROVIDER": "groq",
        "LLM_MODEL": "openai/gpt-oss-20b",
        "GROQ_API_KEY": "fake",
    }
    monkeypatch.setattr(client, "dotenv_values", lambda *_args, **_kwargs: config)
    handler = AsyncMock()
    real_client = httpx.AsyncClient

    def make_client(**kwargs):
        return real_client(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(client.httpx, "AsyncClient", make_client)
    return handler


@pytest.mark.parametrize(
    ("lang", "text"),
    [
        ("en", "You sound tired. Can you pull over safely?"),
        ("ro", "Can you stop somewhere safe and rest?"),
    ],
)
async def test_generate_alert_valid(provider, lang, text):
    provider.return_value = completion(text)
    assert await generate_alert({"lang": lang, "type": "drowsiness"}) == text

    request = provider.await_args.args[0]
    assert str(request.url) == "https://api.groq.com/openai/v1/chat/completions"
    body = json.loads(request.content)
    assert body["model"] == "openai/gpt-oss-20b"
    system, user = body["messages"]
    assert system["role"] == "system" and "25 words" in system["content"]
    assert json.loads(user["content"]) == {"lang": "en", "type": "drowsiness"}


def completion(text):
    return httpx.Response(200, json={"choices": [{"message": {"content": text}}]})


@pytest.mark.parametrize(
    ("lang", "text"),
    [
        ("en", "Can you tap the screen?"),
        ("en", "Can you look at the dashboard?"),
        ("en", "It's your fault. Are you okay?"),
        ("en", "Can you keep driving?"),
        ("en", "Take a break."),
        ("en", "word " * 24 + "are you okay?"),
        ("ro", "Poți să apeși butonul?"),
        ("ro", "Uită-te în oglindă, poți?"),
        ("ro", "Este vina ta. Ești bine?"),
        ("ro", "Poți continua să conduci?"),
        ("ro", "Oprește în siguranță."),
        ("en", "Hai să luăm o pauză. Poți opri într-un loc sigur?"),
    ],
)
async def test_generate_alert_rejects_unsafe_or_invalid_text(provider, lang, text):
    provider.return_value = completion(text)
    assert await generate_alert({"lang": lang, "type": "drowsiness"}) is None


@pytest.mark.parametrize(
    ("transcript", "output", "expected"),
    [
        ("I'd like to stop for a rest.", "NOT_OK", "NOT_OK"),
        ("Am nevoie să vină cineva.", "HELP", "HELP"),
        ("I feel all right.", "OK, probably", None),
        ("Mă simt rău.", "TIRED", None),
        ("Mă simt rău.", "NONE", None),
    ],
)
async def test_classify_reply_validates_provider_label(provider, transcript, output, expected):
    provider.return_value = completion(output)
    assert await classify_reply(transcript, "confirmation") == expected
    request = provider.await_args.args[0]
    messages = json.loads(request.content)["messages"]
    assert messages[0]["role"] == "system"
    assert json.loads(messages[1]["content"]) == {
        "transcript": transcript,
        "expecting": "confirmation",
    }


@pytest.mark.parametrize("lang", ["ro", "en"])
async def test_functions_timeout_or_fail(provider, lang):
    async def slow_response(_request):
        await asyncio.sleep(5)

    provider.side_effect = slow_response
    transcript = "Mă simt amețit." if lang == "ro" else "I feel dizzy."
    assert await generate_alert({"lang": lang}, timeout_s=0.15) is None
    assert await classify_reply(transcript, "confirmation", timeout_s=0.15) is None
    assert provider.await_count == 2

    provider.side_effect = None
    provider.return_value = httpx.Response(429)
    assert await generate_alert({"lang": lang}) is None
    assert await classify_reply(transcript, "confirmation") is None


async def test_public_deadlines_are_passed_to_client(monkeypatch):
    request = AsyncMock(return_value=None)
    monkeypatch.setattr(replies.client, "request_text", request)
    await generate_alert({"lang": "en"})
    await classify_reply("Synthetic reply", "confirmation")
    assert [call.args[1] for call in request.await_args_list] == [2.0, 1.5]


async def test_live_scope_reuses_one_client(provider, monkeypatch):
    factory = Mock(wraps=client.httpx.AsyncClient)
    monkeypatch.setattr(client.httpx, "AsyncClient", factory)
    provider.return_value = completion("OK")
    async with client.reuse_http_client() as shared:
        assert await asyncio.create_task(classify_reply("first reply", "confirmation")) == "OK"
        assert await asyncio.create_task(classify_reply("second reply", "confirmation")) == "OK"
        assert not shared.is_closed
    assert shared.is_closed
    factory.assert_called_once_with(timeout=None)
    assert provider.await_count == 2
