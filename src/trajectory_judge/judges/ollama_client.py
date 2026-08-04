"""Minimal Ollama client for structured, seeded, reproducible generation.

Three settings here are not cosmetic:

``format``
    A JSON schema, so verdicts are parsed rather than scraped out of prose. Property order in
    the schema is also generation order, which is why every schema puts ``reasoning`` first —
    the model states its case before it commits to a verdict.
``num_ctx``
    Set explicitly. Ollama's default context is small enough to silently truncate a rendered
    trajectory, and a judge scoring the half of the trajectory it happened to see is a bug that
    looks like a finding.
``seed`` and ``temperature``
    Pinned at 0 for single-shot judges so a rerun reproduces the table. Self-consistency
    deliberately unpins them; that is the only place sampling varies.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any

import httpx

DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_NUM_CTX = 8192
DEFAULT_TIMEOUT_S = 600.0


@dataclass(frozen=True)
class Generation:
    """One model response plus what it cost to get it."""

    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    error: str | None = None

    def parse(self) -> dict[str, Any] | None:
        try:
            value = json.loads(self.text)
        except json.JSONDecodeError:
            return None
        return value if isinstance(value, dict) else None


def generate(
    model: str,
    prompt: str,
    schema: dict[str, Any],
    *,
    temperature: float = 0.0,
    seed: int = 7,
    host: str = DEFAULT_HOST,
    num_ctx: int = DEFAULT_NUM_CTX,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> Generation:
    """Call Ollama once. Never raises — transport failures come back as ``error``."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": schema,
        "options": {"temperature": temperature, "seed": seed, "num_ctx": num_ctx},
    }
    started = time.perf_counter()
    try:
        response = httpx.post(f"{host}/api/generate", json=payload, timeout=timeout_s)
        response.raise_for_status()
        body = response.json()
    except (httpx.HTTPError, json.JSONDecodeError) as exc:
        return Generation(
            "", 0, 0, time.perf_counter() - started, error=f"{type(exc).__name__}: {exc}"
        )

    return Generation(
        text=str(body.get("response", "")),
        prompt_tokens=int(body.get("prompt_eval_count", 0)),
        completion_tokens=int(body.get("eval_count", 0)),
        latency_s=time.perf_counter() - started,
    )


def is_available(host: str = DEFAULT_HOST) -> bool:
    """Whether an Ollama server is reachable. Used to skip model tests, never to hide errors."""
    try:
        return httpx.get(f"{host}/api/tags", timeout=2.0).status_code == 200
    except httpx.HTTPError:
        return False
