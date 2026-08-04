"""Fault injection has to be trustworthy before any judge is measured against it.

Two things are pinned here. First, that each mutation produces the fault it claims and nothing
else. Second, the checker's per-type coverage — the numbers the README quotes — so the claim
that a rule engine cannot see a wrong tool choice or an invented promise is a failing test if
it ever stops being true.
"""

from __future__ import annotations

import pytest

from trajectory_judge.agents.oracle import run_oracle
from trajectory_judge.env.checker import check
from trajectory_judge.env.world import generate_instances
from trajectory_judge.mutate import mutate
from trajectory_judge.trace import FailureType

INSTANCES = generate_instances(120, seed=7)
CLEAN = {ins.instance_id: run_oracle(ins) for ins in INSTANCES}

#: (fraction flagged by the checker, fraction the checker also types correctly).
#: Anything at 0.0 is only reachable by a judge that reads the trajectory.
EXPECTED_CHECKER_COVERAGE: dict[FailureType, tuple[float, float]] = {
    FailureType.WRONG_TOOL: (0.0, 0.0),
    FailureType.HALLUCINATED_ARGUMENT: (1.0, 1.0),
    FailureType.SKIPPED_PRECONDITION: (1.0, 1.0),
    FailureType.IGNORED_OBSERVATION: (1.0, 1.0),
    FailureType.PREMATURE_STOP: (1.0, 1.0),
    FailureType.UNSUPPORTED_CLAIM: (0.0, 0.0),
}

#: Faults that leave the customer-visible outcome correct, and are therefore invisible to
#: outcome-only evaluation.
ALWAYS_SILENT = (
    FailureType.WRONG_TOOL,
    FailureType.HALLUCINATED_ARGUMENT,
    FailureType.UNSUPPORTED_CLAIM,
)
NEVER_SILENT = (FailureType.IGNORED_OBSERVATION, FailureType.PREMATURE_STOP)


def _mutants(failure_type: FailureType) -> list[tuple[object, object]]:
    out = []
    for ins in INSTANCES:
        mutant = mutate(ins, CLEAN[ins.instance_id], failure_type)
        if mutant is not None:
            out.append((ins, mutant))
    return out


@pytest.mark.parametrize("failure_type", list(FailureType), ids=lambda f: f.value)
def test_every_failure_type_has_hosts(failure_type: FailureType) -> None:
    assert len(_mutants(failure_type)) >= 20


@pytest.mark.parametrize("failure_type", list(FailureType), ids=lambda f: f.value)
def test_labels_are_well_formed(failure_type: FailureType) -> None:
    for _, mutant in _mutants(failure_type):
        assert mutant.label.faulty is True  # type: ignore[attr-defined]
        assert mutant.label.failure_type is failure_type  # type: ignore[attr-defined]
        step = mutant.label.failure_step  # type: ignore[attr-defined]
        assert step is not None and 0 <= step < len(mutant.steps)  # type: ignore[attr-defined]


@pytest.mark.parametrize("failure_type", ALWAYS_SILENT, ids=lambda f: f.value)
def test_silent_faults_leave_the_outcome_correct(failure_type: FailureType) -> None:
    mutants = _mutants(failure_type)
    assert all(m.label.silent for _, m in mutants)  # type: ignore[attr-defined]


@pytest.mark.parametrize("failure_type", NEVER_SILENT, ids=lambda f: f.value)
def test_loud_faults_change_the_outcome(failure_type: FailureType) -> None:
    mutants = _mutants(failure_type)
    assert not any(m.label.outcome_correct for _, m in mutants)  # type: ignore[attr-defined]


def test_skipping_the_check_is_silent_only_at_full_price() -> None:
    """With a restocking fee, skipping the check also gets the amount wrong, so it shows."""
    by_difficulty = {
        ins.difficulty: mutant.label.silent  # type: ignore[attr-defined]
        for ins, mutant in _mutants(FailureType.SKIPPED_PRECONDITION)
    }
    assert by_difficulty["happy"] is True
    assert by_difficulty["restocking"] is False


@pytest.mark.parametrize(
    ("failure_type", "expected"),
    list(EXPECTED_CHECKER_COVERAGE.items()),
    ids=lambda v: v.value if isinstance(v, FailureType) else "",
)
def test_checker_coverage_is_what_the_readme_claims(
    failure_type: FailureType, expected: tuple[float, float]
) -> None:
    mutants = _mutants(failure_type)
    violations = [check(m, ins) for ins, m in mutants]  # type: ignore[arg-type]
    flagged = sum(1 for v in violations if v) / len(mutants)
    typed = sum(1 for v in violations if v and v[0].failure_type is failure_type) / len(mutants)
    assert (flagged, typed) == expected


def test_mutation_is_deterministic() -> None:
    ins = INSTANCES[0]
    a = mutate(ins, CLEAN[ins.instance_id], FailureType.UNSUPPORTED_CLAIM)
    b = mutate(ins, CLEAN[ins.instance_id], FailureType.UNSUPPORTED_CLAIM)
    assert a is not None and b is not None
    assert a.model_dump_json() == b.model_dump_json()


@pytest.mark.parametrize("failure_type", list(FailureType), ids=lambda f: f.value)
def test_a_mutant_always_differs_from_its_parent(failure_type: FailureType) -> None:
    for ins, mutant in _mutants(failure_type):
        clean = CLEAN[ins.instance_id]  # type: ignore[attr-defined]
        assert (mutant.final_answer, [s.call for s in mutant.steps]) != (  # type: ignore[attr-defined]
            clean.final_answer,
            [s.call for s in clean.steps],
        )


def test_replayed_observations_are_consistent_with_the_calls() -> None:
    """A hallucinated SKU must actually fail in the world, not just look wrong on paper."""
    for ins, mutant in _mutants(FailureType.HALLUCINATED_ARGUMENT):
        step = mutant.steps[mutant.label.failure_step]  # type: ignore[attr-defined]
        assert step.call.tool == "get_policy"
        assert step.observation.ok is False
        assert ins.policy["sku"] not in str(step.call.args)  # type: ignore[attr-defined]
