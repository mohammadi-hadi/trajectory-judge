"""Doubles for the LLM path.

A uniquely named module rather than the conftest: `tests/` and `tests/serve/` both land on
sys.path, so a plain `import conftest` resolves to whichever was imported first.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from trajectory_judge.judges.ollama_client import Generation


def verdict_json(
    *,
    faulty: bool = True,
    reasoning: str = "step 1 refunded without checking eligibility",
    failure_step: int | None = 1,
    failure_type: str = "skipped_precondition",
    confidence: float = 0.9,
) -> str:
    """A schema-valid response body, as Ollama would return it."""
    body: dict[str, Any] = {
        "reasoning": reasoning,
        "faulty": faulty,
        "failure_type": failure_type,
        "confidence": confidence,
    }
    if failure_step is not None:
        body["failure_step"] = failure_step
    return json.dumps(body)


class FakeGenerate:
    """A stand-in for ``generate`` that records what it was asked and returns what you set."""

    def __init__(self) -> None:
        self.text: str | Callable[[int], str] = verdict_json()
        self.error: str | None = None
        self.prompt_tokens = 11
        self.completion_tokens = 22
        self.latency_s = 0.125
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self, model: str, prompt: str, schema: dict[str, Any], **kwargs: Any
    ) -> Generation:
        self.calls.append({"model": model, "prompt": prompt, "schema": schema, **kwargs})
        text = self.text(len(self.calls) - 1) if callable(self.text) else self.text
        if self.error is not None:
            return Generation("", 0, 0, self.latency_s, error=self.error)
        return Generation(text, self.prompt_tokens, self.completion_tokens, self.latency_s)
