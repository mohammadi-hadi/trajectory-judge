"""A failed model call must never be stored as a verdict.

An LLM judge never raises: a call that gets no answer comes back as a "clean" verdict at
chance with the transport error attached. Stored, it would be read back as judged, so a server
that dies halfway through an overnight run would leave every remaining trajectory recorded as
clean and the resume logic would never revisit them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from helpers_llm import FakeGenerate
from typer.testing import CliRunner

from trajectory_judge import store
from trajectory_judge.cli import app
from trajectory_judge.judges.ollama_client import Generation

runner = CliRunner()


def run_outcome(out: Path) -> Any:
    return runner.invoke(app, ["run", "--n", "8", "--judges", "outcome", "--out", str(out)])


def test_a_dead_server_stops_the_run_and_stores_nothing(
    fake_generate: FakeGenerate, tmp_path: Path
) -> None:
    fake_generate.error = "ConnectError: [Errno 61] Connection refused"
    result = run_outcome(tmp_path)
    assert result.exit_code == 2
    assert len(fake_generate.calls) == 3
    assert store.read_verdicts(tmp_path) == []


def test_a_single_failed_call_is_left_for_the_rerun(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = FakeGenerate()
    calls = {"n": 0}

    def flaky(model: str, prompt: str, schema: dict[str, Any], **kwargs: Any) -> Generation:
        calls["n"] += 1
        if calls["n"] == 2:
            return Generation("", 0, 0, 0.1, error="ReadTimeout: timed out")
        return fake(model, prompt, schema, **kwargs)

    monkeypatch.setattr("trajectory_judge.judges.llm.generate", flaky)
    first = run_outcome(tmp_path)
    assert first.exit_code == 3
    assert len(store.read_verdicts(tmp_path)) == 7

    second = run_outcome(tmp_path)
    assert second.exit_code == 0, second.output
    verdicts = store.read_verdicts(tmp_path)
    assert len(verdicts) == 8
    assert all(v.error is None for v in verdicts)


def test_an_unusable_answer_is_stored_but_a_streak_of_them_stops_the_run(
    fake_generate: FakeGenerate, tmp_path: Path
) -> None:
    fake_generate.text = "not a verdict"
    result = run_outcome(tmp_path)
    assert result.exit_code == 2
    verdicts = store.read_verdicts(tmp_path)
    assert len(verdicts) == 3
    assert {v.error for v in verdicts} == {"unparseable response"}
