"""Generate short alerts and classify spoken replies with bounded provider calls."""

import asyncio
import json
import re
import unicodedata
from collections.abc import Callable
from functools import cache
from math import isfinite
from pathlib import Path
from typing import TypeVar, get_args

from backend.schemas import Intent

from . import client
from .language import is_english

PROMPTS = Path(__file__).resolve().parents[2] / "config" / "prompts"
ALERT_FIELDS = ("event_type", "type", "tier", "signals", "trip_time", "preferences", "lang")
SPOKEN_INTENTS = set(get_args(Intent)) - {"NONE"}
WORDS = re.compile(r"\b[^\W_]+(?:['’][^\W_]+)*\b")
T = TypeVar("T")


@cache
def read_prompt(name: str) -> str:
    return (PROMPTS / f"{name}.md").read_text(encoding="utf-8")


@cache
def alert_blocklist() -> re.Pattern[str]:
    rules = (PROMPTS / "alert_blocklist.txt").read_text(encoding="utf-8").splitlines()
    return re.compile("|".join(rule for rule in rules if rule and not rule.startswith("#")))


def validate_alert(output: str) -> str | None:
    if not output.endswith("?") or not 1 <= len(WORDS.findall(output)) <= 25:
        return None
    normalized = unicodedata.normalize("NFKD", output.casefold())
    unaccented = "".join(char for char in normalized if not unicodedata.combining(char))
    words = " ".join(re.findall(r"[^\W_]+", unaccented))
    return None if alert_blocklist().search(words) else output


def validate_intent(output: str) -> str | None:
    return output if output in SPOKEN_INTENTS else None


async def _request(
    task: str,
    payload: dict,
    timeout_s: float,
    limit_s: float,
    validate: Callable[[str], T | None],
) -> T | None:
    try:
        if not isfinite(timeout_s) or timeout_s <= 0:
            return None
        budget = min(timeout_s, limit_s)
        async with asyncio.timeout(budget):
            prompt = await asyncio.to_thread(read_prompt, task)
            output = await client.request_text(
                json.dumps(payload, ensure_ascii=False), budget, system_prompt=prompt
            )
            return await asyncio.to_thread(validate, output.strip()) if output else None
    except Exception:  # noqa: BLE001 - caller supplies the fallback
        return None


async def generate_alert(ctx: dict, timeout_s: float = 2.0) -> str | None:
    if not isinstance(ctx, dict):
        return None
    payload = {key: ctx[key] for key in ALERT_FIELDS if key in ctx}
    payload["lang"] = "en"
    return await _request("generate_alert", payload, timeout_s, 2.0, validate_english_alert)


def validate_english_alert(output: str) -> str | None:
    return output if validate_alert(output) is not None and is_english(output) else None


async def classify_reply(transcript: str, expecting: str, timeout_s: float = 1.5) -> str | None:
    if not isinstance(transcript, str) or not transcript.strip():
        return None
    return await _request(
        "classify_reply",
        {"transcript": transcript, "expecting": expecting},
        timeout_s,
        1.5,
        validate_intent,
    )
