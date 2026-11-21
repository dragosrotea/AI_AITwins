"""Request an offline trivia alert and check the live spoken reply."""

import asyncio
import json
import time
from uuid import uuid4

import websockets

from backend.bus.client import bus_url
from backend.llm import grade_answer


async def main() -> None:
    async with asyncio.timeout(30), websockets.connect(bus_url()) as ws:
        await ws.send(json.dumps({"op": "sub", "topics": ["voice.say", "voice.reply"]}))

        async def publish(topic, src, data):
            await ws.send(
                json.dumps(
                    {"op": "pub", "topic": topic, "src": src, "ts": time.time(), "data": data}
                )
            )

        await publish(
            "control.trip",
            "ui",
            {
                "action": "start",
                "pretrip": {
                    "planned_min": 60,
                    "sleep_h": 8,
                    "awake_h": 1,
                    "tiredness": 1,
                    "caffeine": False,
                    "engagement": "trivia",
                    "lang": "en",
                    "emergency_contact": "",
                    "resting_hr": 65,
                },
            },
        )
        await publish("context.road", "maps", {"load": "LOW"})
        await publish(
            "detector.event",
            "cv",
            {
                "event_id": f"t14-{uuid4().hex[:8]}",
                "type": "zoning_out",
                "tier": "T1",
                "score": 90,
                "signals": {"region": "France"},
            },
        )

        alert = None
        async for raw in ws:
            event = json.loads(raw)
            data = event.get("data", {})
            if (
                event.get("topic") == "voice.say"
                and data.get("audio_key") == "engagement_trivia_en_1"
            ):
                alert = data
                print(f"{data['text']} Say: I think it's Paris.", flush=True)
            if (
                alert is not None
                and event.get("topic") == "voice.reply"
                and data.get("alert_id") == alert["alert_id"]
            ):
                if data["intent"] != "ANSWER":
                    raise SystemExit(f"Expected ANSWER. Got {data['intent']}.")
                result = await grade_answer(alert["text"], "Paris", data["transcript"])
                if result is None or not result["correct"]:
                    raise SystemExit("The recorded reply did not grade as correct.")
                print(f"ANSWER / {data['path']}. {result['feedback']}")
                break


if __name__ == "__main__":
    asyncio.run(main())
