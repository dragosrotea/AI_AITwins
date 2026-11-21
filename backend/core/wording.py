"""Alert wording: the LLM first, a template within the time limit (LEAD-T13, AL-05).

T1 and T2 ask llm.generate_alert(ctx) with core.alerts.llm_timeout_s (2 s). On None, an error
or a timeout, the text comes from the voice engineer's catalog (config/prompts/templates.yaml
via backend.llm.templates), rotating through its variants and passing the pre-rendered
audio_key. T3 (microsleep, crash) never waits on the network: it takes the template at once
(NF-01). If the catalog lacks the type, the built-in line in backend.core.events is used.

The catalog and the LLM prompt are English only (the voice engineer's choice), so this wording
is always spoken with lang "en". Which path was used is returned for the trip log (AL-05).
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

log = logging.getLogger("core.wording")

Generate = Callable[[dict[str, Any], float], Awaitable[str | None]]
Fallback = Callable[[str, int], Any]  # (event_type, 1-based variant) -> .text, .audio_key
VARIANTS = 3  # templates.yaml keeps 3 variants per type (VOICE-T10)


@dataclass(frozen=True)
class Wording:
    text: str
    lang: str
    path: str  # "llm", "template" or "builtin"
    audio_key: str | None = None
    llm_ms: int | None = None


async def _no_llm(ctx: dict[str, Any], timeout_s: float) -> str | None:
    return None


def _default_generate() -> Generate:
    try:
        from backend.llm import generate_alert
    except ImportError as error:  # e.g. a missing dependency on a fresh laptop
        log.error("backend.llm unavailable (%s): alerts use templates only", error)
        return _no_llm
    return generate_alert


def _missing_catalog(kind: str, variant: int) -> Any:
    raise KeyError(kind)


def _default_fallback() -> Fallback:
    try:
        from backend.llm.templates import english_fallback
    except ImportError as error:
        log.error("template catalog unavailable (%s): alerts use built-in lines", error)
        return _missing_catalog
    return english_fallback


class AlertWriter:
    def __init__(
        self,
        timeout_s: float,
        builtin: dict[str, dict[str, tuple[str, str]]],
        generate: Generate | None = None,
        fallback: Fallback | None = None,
    ) -> None:
        self.timeout_s = timeout_s
        self.builtin = builtin
        self.generate = generate or _default_generate()
        self.fallback = fallback or _default_fallback()
        self._next_variant: dict[str, int] = defaultdict(int)

    async def write(
        self,
        kind: str,
        tier: str,
        ctx: dict[str, Any],
        max_words: int | None = None,
        variant: int | None = None,
    ) -> Wording:
        """max_words: a longer LLM line falls back to the template (RR-06: 20 for road rage).
        variant: a fixed template variant instead of the rotation (the firm road-rage line)."""
        if tier != "T3":
            started = time.monotonic()
            text = await self._ask_llm(ctx)
            llm_ms = round((time.monotonic() - started) * 1000)
            if text and max_words is not None and len(text.split()) > max_words:
                log.info("%s: LLM line has %d words (max %d)", kind, len(text.split()), max_words)
                text = None
            if text:
                return Wording(text, "en", "llm", llm_ms=llm_ms)
            log.info("%s: no usable LLM line in %d ms, using a template", kind, llm_ms)
        else:
            llm_ms = None
        return self._template(kind, llm_ms, variant)

    async def _ask_llm(self, ctx: dict[str, Any]) -> str | None:
        try:
            # generate_alert keeps its own budget; this outer limit is the safety net.
            async with asyncio.timeout(self.timeout_s + 0.1):
                text = await self.generate(ctx, self.timeout_s)
        except Exception as error:  # noqa: BLE001 - the safety path never fails on the LLM
            log.warning("generate_alert failed: %s", type(error).__name__)
            return None
        return text.strip() if isinstance(text, str) and text.strip() else None

    def _template(self, kind: str, llm_ms: int | None, fixed: int | None = None) -> Wording:
        """Next catalog variant for this type (1, 2, 3, 1, ...), else the built-in line."""
        if fixed is not None:
            variant = fixed
        else:
            variant = self._next_variant[kind] % VARIANTS + 1
            self._next_variant[kind] += 1
        for attempt in (variant, 1):  # 1 again if the catalog has fewer variants
            try:
                entry = self.fallback(kind, attempt)
                return Wording(entry.text, "en", "template", entry.audio_key, llm_ms)
            except ValueError:
                continue  # variant out of range
            except (KeyError, OSError) as error:  # type missing, or catalog unreadable
                log.warning("no catalog template for %s (%s)", kind, error)
                break
        text, _short = self.builtin[kind]["en"]
        return Wording(text, "en", "builtin", llm_ms=llm_ms)
