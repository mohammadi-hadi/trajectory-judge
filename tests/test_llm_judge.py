"""The LLM judging path, exercised without a model.

This path had no coverage: every committed number depends on it, and the only thing standing
between a schema change and a silently wrong results table was the models themselves being
consistent. The judges are contractually non-raising, so most of what matters here is that bad
input degrades to a *recorded* failure rather than an exception.
"""

from __future__ import annotations

import httpx
import pytest
from helpers_llm import FakeGenerate, verdict_json

from trajectory_judge.judges.llm import OutcomeJudge, StepRubricJudge
from trajectory_judge.judges.ollama_client import generate
from trajectory_judge.judges.self_consistency import SelfConsistencyJudge
from trajectory_judge.trace import FailureType, Trajectory


def test_schema_valid_response_becomes_a_verdict(
    fake_generate: FakeGenerate, trajectory: Trajectory
) -> None:
    verdict = StepRubricJudge("m").judge(trajectory, None)  # type: ignore[arg-type]
    assert verdict.faulty is True
    assert verdict.failure_step == 1
    assert verdict.failure_type is FailureType.SKIPPED_PRECONDITION
    assert verdict.confidence == pytest.approx(0.9)
    assert verdict.rationale.startswith("step 1 refunded")
    assert verdict.error is None
    assert (verdict.prompt_tokens, verdict.completion_tokens) == (11, 22)
    assert verdict.latency_s == pytest.approx(0.125)


@pytest.mark.parametrize(
    ("given", "expected"),
    [(2.0, 1.0), (-1.0, 0.5), (0.4, 0.5), (0.75, 0.75), ("high", 0.5), (None, 0.5)],
)
def test_confidence_is_clamped_into_the_reportable_range(
    fake_generate: FakeGenerate, trajectory: Trajectory, given: object, expected: float
) -> None:
    fake_generate.text = verdict_json(confidence=given)  # type: ignore[arg-type]
    verdict = StepRubricJudge("m").judge(trajectory, None)  # type: ignore[arg-type]
    assert verdict.confidence == pytest.approx(expected)


@pytest.mark.parametrize("step", [-1, 2, 99, "one", None])
def test_a_failure_step_outside_the_trajectory_is_dropped(
    fake_generate: FakeGenerate, trajectory: Trajectory, step: object
) -> None:
    fake_generate.text = verdict_json(failure_step=step)  # type: ignore[arg-type]
    verdict = StepRubricJudge("m").judge(trajectory, None)  # type: ignore[arg-type]
    assert verdict.failure_step is None


def test_an_unknown_failure_type_is_dropped_rather_than_guessed(
    fake_generate: FakeGenerate, trajectory: Trajectory
) -> None:
    fake_generate.text = verdict_json(failure_type="vibes")
    verdict = StepRubricJudge("m").judge(trajectory, None)  # type: ignore[arg-type]
    assert verdict.faulty is True
    assert verdict.failure_type is None


def test_a_clean_verdict_never_carries_a_failure_type_or_step(
    fake_generate: FakeGenerate, trajectory: Trajectory
) -> None:
    fake_generate.text = verdict_json(faulty=False, failure_type="none", failure_step=1)
    verdict = StepRubricJudge("m").judge(trajectory, None)  # type: ignore[arg-type]
    assert (verdict.faulty, verdict.failure_type, verdict.failure_step) == (False, None, None)


def test_unparseable_text_votes_clean_at_chance_and_says_so(
    fake_generate: FakeGenerate, trajectory: Trajectory
) -> None:
    fake_generate.text = "I think the agent did fine, honestly."
    verdict = StepRubricJudge("m").judge(trajectory, None)  # type: ignore[arg-type]
    assert verdict.error == "unparseable response"
    assert verdict.faulty is False
    assert verdict.confidence == pytest.approx(0.5)


def test_a_transport_error_propagates_into_the_verdict_and_does_not_raise(
    fake_generate: FakeGenerate, trajectory: Trajectory
) -> None:
    fake_generate.error = "ConnectError: [Errno 61] Connection refused"
    verdict = StepRubricJudge("m").judge(trajectory, None)  # type: ignore[arg-type]
    assert verdict.error is not None
    assert verdict.error.startswith("ConnectError: ")
    assert verdict.faulty is False
    assert verdict.confidence == pytest.approx(0.5)


def test_the_outcome_judge_never_localises_a_step(
    fake_generate: FakeGenerate, trajectory: Trajectory
) -> None:
    fake_generate.text = verdict_json(failure_step=1)
    verdict = OutcomeJudge("m").judge(trajectory, None)  # type: ignore[arg-type]
    assert verdict.faulty is True
    assert verdict.failure_step is None
    assert "[step 0]" not in fake_generate.calls[0]["prompt"]


def test_the_step_judge_is_shown_the_steps(
    fake_generate: FakeGenerate, trajectory: Trajectory
) -> None:
    StepRubricJudge("m").judge(trajectory, None)  # type: ignore[arg-type]
    assert "[step 0]" in fake_generate.calls[0]["prompt"]


def test_ground_truth_never_reaches_the_prompt(
    fake_generate: FakeGenerate, trajectory: Trajectory
) -> None:
    trajectory.label.failure_type = FailureType.WRONG_TOOL
    trajectory.label.faulty = True
    for judge in (StepRubricJudge("m"), OutcomeJudge("m")):
        judge.judge(trajectory, None)  # type: ignore[arg-type]
    for call in fake_generate.calls:
        # The taxonomy legitimately names every failure type, so only the rendered
        # trajectory is checked: that is the part that could leak this episode's answer.
        rendered = call["prompt"].split("--- BEGIN ---")[1].split("--- END ---")[0]
        assert "wrong_tool" not in rendered
        assert "faulty" not in rendered
        assert "label" not in rendered.lower()


def test_self_consistency_votes_and_sums_its_cost(
    fake_generate: FakeGenerate, trajectory: Trajectory
) -> None:
    # two faulty, one clean -> faulty at 2/3
    fake_generate.text = lambda i: verdict_json(faulty=i < 2, failure_type="wrong_tool")
    verdict = SelfConsistencyJudge("m", k=3).judge(trajectory, None)  # type: ignore[arg-type]
    assert verdict.faulty is True
    assert verdict.confidence == pytest.approx(2 / 3)
    assert verdict.failure_type is FailureType.WRONG_TOOL
    assert verdict.latency_s == pytest.approx(3 * 0.125)
    assert verdict.prompt_tokens == 33
    assert len(fake_generate.calls) == 3
    assert [c["seed"] for c in fake_generate.calls] == [7, 8, 9]


def test_self_consistency_breaks_ties_toward_clean(
    fake_generate: FakeGenerate, trajectory: Trajectory
) -> None:
    fake_generate.text = lambda i: verdict_json(faulty=i == 0)
    verdict = SelfConsistencyJudge("m", k=2).judge(trajectory, None)  # type: ignore[arg-type]
    assert verdict.faulty is False


def test_self_consistency_reports_a_member_failure(
    fake_generate: FakeGenerate, trajectory: Trajectory
) -> None:
    fake_generate.error = "ReadTimeout: timed out"
    verdict = SelfConsistencyJudge("m", k=3).judge(trajectory, None)  # type: ignore[arg-type]
    assert verdict.error is not None and verdict.error.startswith("ReadTimeout: ")


def test_generate_names_the_exception_class_in_its_error() -> None:
    """The service maps errors to status codes by this prefix, so the format is a contract."""
    result = generate("m", "p", {}, host="http://127.0.0.1:9", timeout_s=1.0)
    assert result.error is not None
    assert result.error.startswith("ConnectError: ")
    assert result.text == ""
    assert isinstance(httpx.ConnectError("x"), httpx.HTTPError)
