"""Fast, local reply classification using the configured intent phrases."""

import re
import unicodedata
from functools import cache
from pathlib import Path
from typing import Literal

import yaml

from backend.schemas import Lang

LEXICON = Path(__file__).resolve().parents[2] / "config" / "lexicons" / "intents.yaml"
KeywordIntent = Literal["OK", "NOT_OK", "HELP"]


def normalize(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    unaccented = "".join(char for char in decomposed if not unicodedata.combining(char))
    return " ".join(re.findall(r"[^\W_]+", unaccented))


@cache
def patterns() -> dict[Lang, dict[str, re.Pattern[str]]]:
    with LEXICON.open(encoding="utf-8") as file:
        lexicon = yaml.safe_load(file)
    return {
        lang: {
            label: re.compile(
                r"\b(?:" + "|".join(re.escape(normalize(phrase)) for phrase in phrases) + r")\b"
            )
            for label, phrases in groups.items()
        }
        for lang, groups in lexicon.items()
    }


def classify_keyword(transcript: str, lang: Lang) -> KeywordIntent | None:
    text = normalize(transcript)
    rules = patterns()[lang]
    if rules["HELP"].search(text):
        return "HELP"
    if rules["NOT_OK"].search(text):
        return "NOT_OK"
    if rules["OK"].search(text):
        return "NOT_OK" if rules["negation"].search(text) else "OK"
    return None
