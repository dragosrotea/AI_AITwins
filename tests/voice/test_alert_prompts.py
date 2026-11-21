import json
from unittest.mock import AsyncMock

import miniaudio
import pytest

from backend.config import load_config
from backend.core.alerts import AlertManager
from backend.core.checkin import CheckinScheduler
from backend.core.events import TEMPLATES, EventRouter
from backend.core.wording import AlertWriter
from backend.llm import client
from backend.llm.templates import english_fallback
from backend.schemas import VoiceSay
from backend.voice.live import read_clip


@pytest.mark.parametrize(
    ("kind", "firm", "generated", "fallback_variant"),
    [
        ("checkin_due", None, "Quick check-in. Are you feeling alert right now?", 1),
        ("road_rage", False, "Let's take a moment. Can you take a slow breath?", 1),
        ("road_rage", True, "Let's take a breather. Can you pull over somewhere safe?", 3),
    ],
)
@pytest.mark.parametrize("online", [True, False])
async def test_core_alert_prompt_and_offline_clip(
    monkeypatch, kind, firm, generated, fallback_variant, online
):
    provider = AsyncMock(return_value=generated if online else None)
    monkeypatch.setattr(client, "request_text", provider)
    core = load_config("default")["core"]
    sent = []

    def publish(topic, data):
        sent.append((topic, data))

    manager = AlertManager(core["alerts"], publish)
    writer = AlertWriter(core["alerts"]["llm_timeout_s"], TEMPLATES)
    if firm is None:
        scheduler = CheckinScheduler(core["checkin"], manager, writer, TEMPLATES, publish)
        scheduler.on_trip({"action": "start", "pretrip": {"sleep_h": 8, "tiredness": 1}})
        scheduler.next_checkin = scheduler.clock()
        await scheduler.tick()
        tier = "T1"
    else:
        router = EventRouter(manager, core["routing"], writer)
        await router.handle(
            {
                "src": "core",
                "data": {
                    "event_id": "t15-road-rage",
                    "type": kind,
                    "tier": "T2",
                    "score": 90 if firm else 60,
                    "signals": {"firm": firm},
                },
            }
        )
        tier = "T2"

    provider.assert_awaited_once()
    payload = json.loads(provider.await_args.args[0])
    assert (payload["event_type"], payload["tier"], payload["lang"]) == (kind, tier, "en")
    assert payload.get("firm") is firm
    assert generated in provider.await_args.kwargs["system_prompt"]
    [say] = [VoiceSay.model_validate(data) for topic, data in sent if topic == "voice.say"]
    assert (say.lang, say.tier) == ("en", tier)
    if online:
        assert manager.active.signals["text_path"] == "llm"
        assert say.text == generated and say.audio_key is None
    else:
        assert manager.active.signals["text_path"] == "template"
        fallback = english_fallback(kind, fallback_variant)
        assert (say.text, say.audio_key) == (fallback.text, fallback.audio_key)
        raw = read_clip(say.audio_key)
        assert raw is not None and miniaudio.decode(raw).num_frames > 0

    assert say.text.endswith("?") and len(say.text.split()) <= (20 if kind == "road_rage" else 25)
