"""Short English engagement prompts and local answer grading."""

import json
import re
import unicodedata
from functools import cache
from pathlib import Path

from .replies import _request, validate_english_alert

BANK = Path(__file__).resolve().parents[2] / "config" / "trivia_fallback.json"
FACT_QUESTION = "Did you know that?"


@cache
def trivia_bank() -> list[dict]:
    return json.loads(BANK.read_text(encoding="utf-8"))


def fact_prompt(entry: dict) -> str:
    return f"{entry['fact']} {FACT_QUESTION}"


def engagement_clips() -> dict[str, str]:
    return {
        key: text
        for index, entry in enumerate(trivia_bank(), 1)
        for key, text in (
            (f"engagement_trivia_en_{index}", entry["question"]),
            (f"engagement_fact_en_{index}", fact_prompt(entry)),
        )
    }


@cache
def _clip_keys() -> dict[str, str]:
    return {text: key for key, text in engagement_clips().items()}


def engagement_audio_key(text: str) -> str | None:
    return _clip_keys().get(text)


def _validate_engagement(output: str, kind: str) -> dict | None:
    item = json.loads(output)
    if not isinstance(item, dict) or set(item) != {"kind", "text", "answer"}:
        return None
    text = item["text"]
    answer = item["answer"]
    if item["kind"] != kind or not isinstance(text, str) or validate_english_alert(text) is None:
        return None
    if kind == "fact":
        if answer is not None or not text.endswith(FACT_QUESTION) or text == FACT_QUESTION:
            return None
    elif not isinstance(answer, str) or not 1 <= len(_normalize(answer).split()) <= 3:
        return None
    return item


async def generate_engagement(ctx: dict, timeout_s: float = 2.0) -> dict | None:
    if not isinstance(ctx, dict):
        return None
    preference = ctx.get("engagement", ctx.get("kind"))
    if preference == "quiet":
        return None
    kind = "fact" if preference == "fact" else "trivia"
    region = ctx.get("region")
    region = region.strip()[:100] if isinstance(region, str) else ""
    topic = ctx.get("topic")
    topic = topic.strip()[:60] if isinstance(topic, str) else ""
    avoid = [q for q in ctx.get("avoid") or [] if isinstance(q, str)][-10:]
    avoided = {_normalize(q) for q in avoid}

    def validate(output: str) -> dict | None:
        item = _validate_engagement(output, kind)
        if item is not None and _normalize(item["text"]) in avoided:
            return None  # the model repeated a recent question: use the bank instead
        return item

    payload: dict = {"kind": kind, "region": region}
    if topic:
        payload["topic"] = topic
    if avoid:
        payload["avoid"] = avoid
    item = await _request("generate_engagement", payload, timeout_s, 2.0, validate)
    if item is not None:
        return item
    bank = trivia_bank()
    regional = [entry for entry in bank if region.casefold() in entry.get("regions", [])]
    candidates = regional or bank
    index = ctx.get("fallback_index", 0)
    index = index if isinstance(index, int) else 0
    # Next bank entry whose question was not asked recently.
    for step in range(len(candidates)):
        entry = candidates[(index + step) % len(candidates)]
        if _normalize(entry["question"]) not in avoided and (
            kind != "fact" or _normalize(fact_prompt(entry)) not in avoided
        ):
            break
    if kind == "fact":
        return {"kind": kind, "text": fact_prompt(entry), "answer": None}
    return {"kind": kind, "text": entry["question"], "answer": entry["accepted_answers"][0]}


def _normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.casefold())
    text = "".join(char for char in normalized if not unicodedata.combining(char))
    return " ".join(re.findall(r"[^\W_]+", text))


async def grade_answer(
    question: str, expected: str, reply: str, timeout_s: float = 1.5
) -> dict | None:
    if (
        not all(isinstance(value, str) for value in (question, expected, reply))
        or not expected.strip()
    ):
        return None
    question_key = _normalize(question)
    known = next(
        (entry for entry in trivia_bank() if question_key == _normalize(entry["question"])), None
    )
    accepted = known["accepted_answers"] if known else [expected]
    answer = _normalize(reply)
    answer = re.sub(
        r"^(?:(?:i think|i believe|my guess is) )?(?:it s |it is |the answer is )?", "", answer
    )
    answer = re.sub(r" (?:i think|i believe)$", "", answer)
    correct = bool(answer) and answer in {_normalize(value) for value in accepted}
    if (
        not correct
        and known is None
        and answer
        and not set(answer.split())
        & {
            "not",
            "no",
            "never",
            "wrong",
            "isn",
            "wasn",
            "don",
            "didn",
        }
    ):
        # For a novel online question, only ask about equivalence after local
        # matching. An absent or malformed grade leaves the answer incorrect.
        verdict = await _request(
            "grade_answer",
            {"question": question, "expected": expected, "reply": reply},
            timeout_s,
            1.5,
            lambda output: output if output in {"EQUIVALENT", "DIFFERENT"} else None,
        )
        correct = verdict == "EQUIVALENT"
    return {"correct": correct, "feedback": "That's right." if correct else "Not quite."}
