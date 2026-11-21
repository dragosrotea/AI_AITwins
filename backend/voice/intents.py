"""Fast, local reply classification using the configured intent phrases."""

import re
import unicodedata
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Literal

import yaml

from backend.llm.replies import classify_reply_with_status

LEXICON = Path(__file__).resolve().parents[2] / "config" / "lexicons" / "intents.yaml"
KeywordIntent = Literal["OK", "NOT_OK", "HELP"]
_DIRECT_CONFIRMATION = re.compile(
    r"(?:(?:yes|yeah|sure|ok|okay) )?i (?:am|m) "
    r"(?:feeling )?(?:ok|okay|fine|good|safe|alright|all right)(?: thanks| thank you)?"
)
_ABILITY = re.compile(r"(?:(?:yes|yeah|sure|ok|okay) )?i can(?: thanks| thank you)?")
_STOP_QUESTION = re.compile(r"\b(?:pull over|stop|park|take a break|take a rest)\b")
_STOP_COMMITMENT = re.compile(
    r"\bi (?:(?:ll|will|can|shall) |(?:m|am) (?:going to|about to) )"
    r"(?:pull (?:over|in|into|off (?:the road|at|there))|"
    r"(?:stop|park)(?: (?:there|at|in|soon|now|safely|somewhere|the car)\b|$)|"
    r"take (?:a break|a rest|the next exit))\b"
)
_REFUSAL = re.compile(
    r"\bi (?:won t|will not|can t|cannot|refuse|m not going to|am not going to)\b"
)
_UNCERTAIN = re.compile(r"\b(?:maybe|probably|might|perhaps|not sure|i think|i guess|try|if)\b")
_DIRECTION = re.compile(
    r"^(?:(?:yes|yeah|ok|okay) )?(?:turn|go|keep|take|make|head|drive|continue)\b"
)
_WELLNESS = re.compile(
    r"\b(?:i (?:m|am|feel) (?:feeling )?(?:quite |pretty |still |doing )?"
    r"(?:ok|okay|fine|good|well|safe|alright|all right|awake|alert|ready)"
    r"|i can (?:answer|respond|hear you)|all good|nothing to worry about|"
    r"everything is (?:ok|okay|fine|all right))\b"
)


def normalize(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    unaccented = "".join(char for char in decomposed if not unicodedata.combining(char))
    return " ".join(re.findall(r"[^\W_]+", unaccented))


@cache
def patterns() -> dict[str, re.Pattern[str]]:
    with LEXICON.open(encoding="utf-8") as file:
        lexicon = yaml.safe_load(file)
    return {
        label: re.compile(
            r"\b(?:" + "|".join(re.escape(normalize(phrase)) for phrase in phrases) + r")\b"
        )
        for label, phrases in lexicon["en"].items()
    }


def classify_keyword(transcript: str, *, needs_stop: bool = False) -> KeywordIntent | None:
    text = normalize(transcript)
    rules = patterns()
    if rules["HELP"].search(text):
        return "HELP"
    if rules["NOT_OK"].search(text) or (needs_stop and _REFUSAL.search(text)):
        return "NOT_OK"
    if (rules["OK"].search(text) or _WELLNESS.search(text)) and rules["negation"].search(text):
        return "NOT_OK"
    if _ABILITY.fullmatch(text) or text in {"yes", "yeah", "sure", "ok", "okay"}:
        return "OK"
    if not needs_stop and (
        rules["OK"].fullmatch(text)
        or _DIRECT_CONFIRMATION.fullmatch(text)
        or re.fullmatch(r"(?:(?:yes|yeah|sure|ok|okay) )?i can answer(?: thanks| thank you)?", text)
    ):
        return "OK"
    return None


@dataclass(frozen=True)
class ReplyClassification:
    intent: str
    path: Literal["keyword", "llm"]
    fallback_failed: bool = False
    timed_out: bool = False


async def classify_confirmation(
    transcript: str, *, question: str = "", timeout_s: float = 1.5
) -> ReplyClassification:
    needs_stop = bool(_STOP_QUESTION.search(normalize(question)))
    intent = classify_keyword(transcript, needs_stop=needs_stop)
    if intent is not None:
        return ReplyClassification(intent, "keyword")
    result, timed_out = await classify_reply_with_status(
        transcript, "confirmation", timeout_s, question=question
    )
    text = normalize(transcript)
    if result == "NOT_OK" and _DIRECTION.match(text) and not re.search(r"\bi\b", text):
        return ReplyClassification("UNCLEAR", "llm")
    if result == "OK":
        affirmative = (
            _STOP_COMMITMENT.search(text)
            if needs_stop
            else (_WELLNESS.search(text) or text == "absolutely")
        )
        if not affirmative or _UNCERTAIN.search(text) or patterns()["negation"].search(text):
            return ReplyClassification("UNCLEAR", "llm")
    if result == "ANSWER":
        return ReplyClassification("UNCLEAR", "llm")
    return ReplyClassification(
        result or "UNCLEAR", "llm", result is None and bool(transcript.strip()), timed_out
    )
