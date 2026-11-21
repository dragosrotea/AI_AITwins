"""Bounded async text requests using the repository's local .env."""

import asyncio
import math
from pathlib import Path

import httpx
from dotenv import dotenv_values

ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
PROVIDERS = {
    "openai": ("https://api.openai.com/v1/chat/completions", "OPENAI_API_KEY"),
    "groq": ("https://api.groq.com/openai/v1/chat/completions", "GROQ_API_KEY"),
}


async def request_text(
    prompt: str, timeout_s: float = 2.0, *, system_prompt: str | None = None
) -> str | None:
    """Return provider text or None; caller cancellation still propagates."""
    try:
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            return None
        async with asyncio.timeout(timeout_s):
            config = await asyncio.to_thread(
                dotenv_values, ENV_PATH, interpolate=False, verbose=False
            )
            provider = (config.get("LLM_PROVIDER") or "").strip().lower()
            model = (config.get("LLM_MODEL") or "").strip()
            endpoint = PROVIDERS.get(provider)
            if endpoint is None or not model:
                return None
            url, key_name = endpoint
            key = (config.get(key_name) or "").strip()
            if not key:
                return None

            messages = [{"role": "user", "content": prompt}]
            if system_prompt is not None:
                messages.insert(0, {"role": "system", "content": system_prompt})
            async with httpx.AsyncClient(timeout=None) as client:
                response = await client.post(
                    url,
                    headers={"Authorization": f"Bearer {key}"},
                    json={
                        "model": model,
                        "messages": messages,
                    },
                )
                response.raise_for_status()
                text = response.json()["choices"][0]["message"]["content"]
                return (text.strip() or None) if isinstance(text, str) else None
    except Exception:  # noqa: BLE001 - provider failures fall back without leaking details
        return None
