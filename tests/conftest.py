"""Shared fixtures.

The important one is ``fake_generate``. Every LLM judge reaches Ollama through
``trajectory_judge.judges.llm.generate``, and ``llm.py`` imports that symbol directly, so a
patch has to target the name *in llm*: patching ``ollama_client.generate`` binds nothing.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from helpers_llm import FakeGenerate

from trajectory_judge.trace import Observation, Step, ToolCall, Trajectory


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
