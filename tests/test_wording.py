"""LEAD-T13: LLM-written alerts with a template fallback within 2 s (AL-05)."""

from __future__ import annotations

import asyncio
import time
from typing import Any, NamedTuple

import pytest

from backend.config import load_config
from backend.core.events import TEMPLATES
from backend.core.wording import AlertWriter
from backend.llm.templates import english_fallback, english_templates

TIMEOUT = load_config("default")["core"]["alerts"]["llm_timeout_s"]


class Entry(NamedTuple):
    text: str
    audio_key: str


def llm_says(text: str | None, delay: float = 0.0, calls: list | None = None) -> Any:
    async def generate(ctx: dict[str, Any], timeout_s: float) -> str | None:
        if calls is not None:
            calls.append((ctx, timeout_s))
        await asyncio.sleep(delay)
        return text

    return generate


def write(writer: AlertWriter, kind: str = "distraction", tier: str = "T2") -> Any:
    return asyncio.run(writer.write(kind, tier, {"event_type": kind, "tier": tier}))


def test_llm_text_is_used_when_it_answers() -> None:
    calls: list = []
    writer = AlertWriter(TIMEOUT, TEMPLATES, generate=llm_says("Eyes ahead, okay?", calls=calls))
    wording = write(writer)
    assert (wording.text, wording.path, wording.lang) == ("Eyes ahead, okay?", "llm", "en")
    assert wording.audio_key is None and wording.llm_ms is not None
    assert calls == [({"event_type": "distraction", "tier": "T2"}, TIMEOUT)]


def test_none_falls_back_to_the_catalog_with_audio() -> None:
    writer = AlertWriter(TIMEOUT, TEMPLATES, generate=llm_says(None))
    wording = write(writer)
    first = english_fallback("distraction", 1)
    assert (wording.text, wording.audio_key) == (first.text, first.audio_key)
    assert wording.path == "template"


def test_template_variants_rotate() -> None:
    writer = AlertWriter(TIMEOUT, TEMPLATES, generate=llm_says(None))
    keys = [write(writer).audio_key for _ in range(4)]
    assert keys == ["distraction_en_1", "distraction_en_2", "distraction_en_3", "distraction_en_1"]


def test_slow_llm_is_cut_off_at_the_timeout() -> None:
    writer = AlertWriter(0.2, TEMPLATES, generate=llm_says("too late?", delay=5))
    started = time.monotonic()
    wording = write(writer)
    assert wording.path == "template"
    assert time.monotonic() - started < 0.6  # AL-05: safety never waits on the network


def test_llm_errors_fall_back() -> None:
    async def broken(ctx: dict[str, Any], timeout_s: float) -> str | None:
        raise RuntimeError("provider down")

    assert write(AlertWriter(TIMEOUT, TEMPLATES, generate=broken)).path == "template"


def test_blank_llm_text_falls_back() -> None:
    assert write(AlertWriter(TIMEOUT, TEMPLATES, generate=llm_says("  "))).path == "template"


def test_t3_never_waits_on_the_llm() -> None:
    calls: list = []
    writer = AlertWriter(TIMEOUT, TEMPLATES, generate=llm_says("LLM?", delay=5, calls=calls))
    started = time.monotonic()
    wording = write(writer, "microsleep", "T3")
    assert calls == [] and wording.path == "template" and wording.llm_ms is None
    assert time.monotonic() - started < 0.05  # NF-01: microsleep alarm under 500 ms


def test_missing_catalog_type_uses_the_builtin_line() -> None:
    def catalog(kind: str, variant: int) -> Entry:
        raise KeyError(kind)

    writer = AlertWriter(TIMEOUT, TEMPLATES, generate=llm_says(None), fallback=catalog)
    wording = write(writer, "phone")
    assert (wording.text, wording.path) == (TEMPLATES["phone"]["en"][0], "builtin")


def test_short_catalog_wraps_to_the_first_variant() -> None:
    def catalog(kind: str, variant: int) -> Entry:
        if variant > 1:
            raise ValueError("only one variant")
        return Entry("Only line?", f"{kind}_en_1")

    writer = AlertWriter(TIMEOUT, TEMPLATES, generate=llm_says(None), fallback=catalog)
    assert [write(writer).audio_key for _ in range(3)] == ["distraction_en_1"] * 3


def test_every_routed_type_has_catalog_templates() -> None:
    routed = load_config("default")["core"]["routing"]["tiers"]
    catalog = english_templates()
    for kind in routed:
        assert len(catalog[kind]) >= 3, kind  # VOICE-T10: at least 3 variants


@pytest.mark.parametrize("kind", ["distraction", "microsleep"])
def test_catalog_audio_files_exist(kind: str) -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "assets" / "audio"
    for entry in english_templates()[kind]:
        assert (root / f"{entry.audio_key}.wav").is_file(), entry.audio_key
