"""Manual English voice check against a running broker and live voice process."""

import argparse
import asyncio
import json
import time
from unittest.mock import patch
from uuid import uuid4

import websockets

from backend.bus.client import bus_url
from backend.llm import client, generate_alert
from backend.llm.templates import english_fallback


async def probe(mode: str, *, tts: bool = False, short_reply: str | None = None) -> None:
    if mode == "offline":

        async def unavailable(*_args, **_kwargs):
            await asyncio.sleep(5)

        with patch.object(client, "request_text", unavailable):
            if await generate_alert({"type": "drowsiness", "lang": "en"}, timeout_s=0.1):
                raise RuntimeError("Expected alert generator to time out")
        print("Simulated provider timeout. Publishing the English local fallback.")

    text, key = english_fallback("drowsiness")
    reply_text = short_reply or (
        "There's a rest area coming up. I'll stop there" if mode == "llm" else "Yes, I can"
    )
    expected_path = "llm" if mode == "llm" else "keyword"
    expected_intent = "NONE" if mode == "silence" else "NOT_OK" if short_reply == "No" else "OK"
    alert_id = f"en-smoke-{uuid4().hex[:8]}"
    reply = None
    async with asyncio.timeout(25), websockets.connect(bus_url()) as ws:
        await ws.send(json.dumps({"op": "sub", "topics": ["voice.status", "voice.reply"]}))
        await ws.send(
            json.dumps(
                {
                    "op": "pub",
                    "topic": "voice.say",
                    "ts": time.time(),
                    "src": "core",
                    "data": {
                        "alert_id": alert_id,
                        "text": text,
                        "audio_key": None if tts else key,
                        "lang": "en",
                        "tier": "T2",
                        "listen_s": 8,
                    },
                }
            )
        )
        print(f"Expect {'English TTS' if tts else key}: {text}")
        instruction = (
            "Stay silent through idle."
            if mode == "silence"
            else f"Say after the chime: {reply_text}"
        )
        print(instruction, flush=True)
        async for raw in ws:
            event = json.loads(raw)
            data = event.get("data", {})
            if data.get("alert_id") != alert_id:
                continue
            print(event["topic"], data, flush=True)
            if event["topic"] == "voice.reply":
                reply = data
            if event["topic"] == "voice.status" and data["state"] == "idle":
                break
    if not reply or (reply["intent"], reply["path"]) != (expected_intent, expected_path):
        raise SystemExit(f"Expected {expected_intent} / {expected_path}.")
    if mode == "silence" and (reply["transcript"] != "" or reply["latency_ms"] != 0):
        raise SystemExit("Expected an empty transcript and zero latency for silence.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("keyword", "llm", "offline", "silence"))
    parser.add_argument("--tts", action="store_true", help="Request live TTS instead of the WAV")
    parser.add_argument("--short-reply", choices=("Yes", "No"), help="Use a one-word keyword reply")
    args = parser.parse_args()
    if args.tts and args.mode == "offline":
        parser.error("--tts requires an online smoke mode")
    if args.short_reply and args.mode != "keyword":
        parser.error("--short-reply requires keyword mode")
    asyncio.run(probe(args.mode, tts=args.tts, short_reply=args.short_reply))
