"""Shared fixtures.

The important one is ``fake_generate``. Every LLM judge reaches Ollama through
``trajectory_judge.judges.llm.generate``, and ``llm.py`` imports that symbol directly, so a
patch has to target the name *in llm*: patching ``ollama_client.generate`` binds nothing.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from typing import Any

import pytest

from trajectory_judge.judges.ollama_client import Generation
from trajectory_judge.trace import Observation, Step, ToolCall, Trajectory


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


@pytest.fixture
def fake_generate(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeGenerate]:
    fake = FakeGenerate()
    monkeypatch.setattr("trajectory_judge.judges.llm.generate", fake)
    yield fake


@pytest.fixture
def trajectory() -> Trajectory:
    """A two-step trajectory. Content does not matter; only that steps exist and are indexed."""
    return Trajectory(
        trajectory_id="T-1",
        instance_id="INS-1",
        goal="refund order ORD-1",
        steps=[
            Step(
                index=i,
                thought=f"thought {i}",
                call=ToolCall(tool="lookup_order", args={"order_id": "ORD-1"}),
                observation=Observation(ok=True, data={"order_id": "ORD-1"}),
            )
            for i in range(2)
        ],
        final_answer="refunded 10.00 EUR",
    )
