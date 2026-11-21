"""Canonical English text and audio keys for alert producers and voice playback."""

import re
from functools import cache
from pathlib import Path
from typing import NamedTuple

import yaml

CATALOG = Path(__file__).resolve().parents[2] / "config" / "prompts" / "templates.yaml"


class Fallback(NamedTuple):
    text: str
    audio_key: str


@cache
def english_templates(path: Path = CATALOG) -> dict[str, tuple[Fallback, ...]]:
    catalog = yaml.safe_load(path.read_text(encoding="utf-8-sig")) or {}
    if not isinstance(catalog, dict):
        raise ValueError("Template catalog must map alert types to language maps")
    entries = {}
    for event, languages in catalog.items():
        if not isinstance(event, str) or not re.fullmatch(r"[a-z0-9_]+", event):
            raise ValueError(f"Invalid template event key: {event!r}")
        variants = languages.get("en") if isinstance(languages, dict) else None
        if (
            not isinstance(variants, list)
            or not variants
            or not all(isinstance(text, str) and text.strip() for text in variants)
        ):
            raise ValueError(f"{event} requires a nonempty list of English templates")
        entries[event] = tuple(
            Fallback(text, f"{event}_en_{index}") for index, text in enumerate(variants, 1)
        )
    return entries


def english_fallback(event_type: str, variant: int = 1) -> Fallback:
    """Return a matching text/key pair. Variant numbers start at 1."""
    variants = english_templates()[event_type]
    if not 1 <= variant <= len(variants):
        raise ValueError(f"Variant must be between 1 and {len(variants)}")
    return variants[variant - 1]
