"""Fast, local reply classification using the configured intent phrases."""

import re
import unicodedata
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Literal

import yaml

from backend.llm import classify_reply

LEXICON = Path(__file__).resolve().parents[2] / "config" / "lexicons" / "intents.yaml"
KeywordIntent = Literal["OK", "NOT_OK", "HELP"]
_DIRECT_CONFIRMATION = re.compile(
    r"(?:(?:yes|yeah|sure|ok|okay) )?"
    r"(?:i (?:am|m) (?:ok|okay|fine|good)|i can answer)(?: thanks| thank you)?"
)
_FALLBACK_CONFIRMATION = re.compile(
    r"(?:(?:yes|yeah|sure|ok|okay) )?"
    r"(?:absolutely|all good(?: here)?|nothing to worry about|everything is (?:fine|okay|all right)"
    r"|i (?:am|m|feel) (?:all right|alright|well|fine|good|okay|awake|alert|ready)"
    r"|i can (?:answer|respond|hear you))(?: thanks| thank you)?"
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


def classify_keyword(transcript: str) -> KeywordIntent | None:
    text = normalize(transcript)
    rules = patterns()
    if rules["HELP"].search(text):
        return "HELP"
    if rules["NOT_OK"].search(text):
        return "NOT_OK"
    if rules["OK"].search(text) and rules["negation"].search(text):
        return "NOT_OK"
    if rules["OK"].fullmatch(text) or _DIRECT_CONFIRMATION.fullmatch(text):
        return "OK"
    return None


@dataclass(frozen=True)
class ReplyClassification:
    intent: str
    path: Literal["keyword", "llm"]
    fallback_failed: bool = False


async def classify_confirmation(transcript: str) -> ReplyClassification:
    intent = classify_keyword(transcript)
    if intent is not None:
        return ReplyClassification(intent, "keyword")
    result = await classify_reply(transcript, "confirmation")
    if result == "OK" and not _FALLBACK_CONFIRMATION.fullmatch(normalize(transcript)):
        return ReplyClassification("UNCLEAR", "llm")
    if result == "ANSWER":
        return ReplyClassification("UNCLEAR", "llm")
    return ReplyClassification(
        result or "UNCLEAR", "llm", result is None and bool(transcript.strip())
    )
