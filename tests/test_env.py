"""The environment, the checker, and the oracle are each other's test oracle.

If the oracle ever violates a rule, either the policy or the rule is wrong — there is no third
option, which is what makes a synthetic world worth the trouble.
"""

from __future__ import annotations

import pytest

from trajectory_judge.agents.oracle import run_oracle
from trajectory_judge.env.checker import check
from trajectory_judge.env.world import DIFFICULTIES, World, generate_instances, outcome_of
from trajectory_judge.trace import Label, Observation, Step, ToolCall, Trajectory

INSTANCES = generate_instances(60, seed=7)


def test_generation_is_deterministic() -> None:
    a = generate_instances(30, seed=7)
    b = generate_instances(30, seed=7)
    assert [i.order for i in a] == [i.order for i in b]
    assert generate_instances(30, seed=8)[0].order != a[0].order


def test_difficulty_strata_are_balanced() -> None:
    counts = {d: 0 for d in DIFFICULTIES}
    for ins in INSTANCES:
        counts[ins.difficulty] += 1
    assert set(counts.values()) == {len(INSTANCES) // len(DIFFICULTIES)}


@pytest.mark.parametrize("instance", INSTANCES, ids=lambda i: f"{i.instance_id}-{i.difficulty}")
def test_oracle_is_clean_and_correct(instance: object) -> None:
    trajectory = run_oracle(instance)  # type: ignore[arg-type]
    assert check(trajectory, instance) == []  # type: ignore[arg-type]
    assert outcome_of(trajectory.steps).matches(instance.expected)  # type: ignore[attr-defined]


def test_only_refundable_strata_end_in_a_refund() -> None:
    by_difficulty = {ins.difficulty: outcome_of(run_oracle(ins).steps).action for ins in INSTANCES}
    assert by_difficulty["happy"] == "refund"
    assert by_difficulty["restocking"] == "refund"
    for stratum in ("expired", "non_refundable", "wrong_customer", "already_refunded"):
        assert by_difficulty[stratum] == "escalate"


def test_restocking_fee_reduces_the_authorised_amount() -> None:
    ins = next(i for i in INSTANCES if i.difficulty == "restocking")
    assert ins.expected.amount_eur is not None
    assert ins.expected.amount_eur < float(ins.order["total_eur"])


def test_environment_is_permissive_by_design() -> None:
    """A refund with no eligibility check must succeed. The rule lives in the checker, not here.

    This is the property the whole project rests on: if the API refused, there would be no
    silent failures to measure.
    """
    ins = next(i for i in INSTANCES if i.difficulty == "expired")
    world = World(ins)
    obs = world.call("issue_refund", {"order_id": ins.order["order_id"], "amount_eur": 5.0})
    assert obs.ok is True
    assert (
        world.call("check_eligibility", {"order_id": ins.order["order_id"]}).data["eligible"]
        is False
    )


def _trajectory(instance: object, calls: list[tuple[str, dict[str, object]]]) -> Trajectory:
    world = World(instance)  # type: ignore[arg-type]
    steps = []
    for index, (tool, args) in enumerate(calls):
        steps.append(
            Step(
                index=index,
                thought="",
                call=ToolCall(tool=tool, args=args),
                observation=world.call(tool, args),
            )
        )
    return Trajectory(
        trajectory_id="T",
        instance_id="I",
        goal="",
        steps=steps,
        final_answer="",
        label=Label(faulty=True),
    )


def test_checker_flags_refund_without_eligibility() -> None:
    ins = next(i for i in INSTANCES if i.difficulty == "happy")
    oid = ins.order["order_id"]
    traj = _trajectory(
        ins,
        [
            ("get_customer", {"email": ins.given["email"]}),
            ("lookup_order", {"order_id": oid}),
            ("issue_refund", {"order_id": oid, "amount_eur": float(ins.order["total_eur"])}),
            ("reply", {"text": "done"}),
        ],
    )
    rules = {v.rule for v in check(traj, ins)}
    assert "R1_refund_without_eligibility" in rules


def test_checker_flags_ungrounded_identifier() -> None:
    ins = INSTANCES[0]
    traj = _trajectory(
        ins,
        [
            ("lookup_order", {"order_id": "ORD-99999"}),
            ("escalate", {"reason": "not found"}),
            ("reply", {"text": "done"}),
        ],
    )
    violations = [v for v in check(traj, ins) if v.rule == "R4_ungrounded_argument"]
    assert violations and violations[0].step_index == 0


def test_checker_flags_amount_that_ignores_the_eligibility_observation() -> None:
    ins = next(i for i in INSTANCES if i.difficulty == "restocking")
    oid = ins.order["order_id"]
    traj = _trajectory(
        ins,
        [
            ("get_customer", {"email": ins.given["email"]}),
            ("lookup_order", {"order_id": oid}),
            ("get_policy", {"sku": ins.order["sku"]}),
            ("check_eligibility", {"order_id": oid}),
            # Refunds the full total, not the fee-adjusted amount the check authorised.
            ("issue_refund", {"order_id": oid, "amount_eur": float(ins.order["total_eur"])}),
            ("reply", {"text": "done"}),
        ],
    )
    rules = {v.rule for v in check(traj, ins)}
    assert "R2_refund_amount_mismatch" in rules


def test_checker_flags_a_trajectory_that_never_acts() -> None:
    ins = INSTANCES[0]
    traj = _trajectory(ins, [("get_customer", {"email": ins.given["email"]})])
    rules = {v.rule for v in check(traj, ins)}
    assert "R8_no_terminal_action" in rules
    assert "R5_reply_placement" in rules


def test_observations_from_failed_calls_ground_nothing() -> None:
    """A tool that errored must not license later arguments."""
    ins = INSTANCES[0]
    traj = _trajectory(
        ins,
        [
            ("lookup_order", {"order_id": "ORD-00000"}),
            ("check_eligibility", {"order_id": "ORD-00000"}),
            ("escalate", {"reason": "x"}),
            ("reply", {"text": "done"}),
        ],
    )
    rules = [v.rule for v in check(traj, ins)]
    assert rules.count("R4_ungrounded_argument") == 2
    assert "R3_eligibility_without_lookup" in rules


def test_observation_model_round_trips() -> None:
    obs = Observation(ok=False, error="boom")
    assert Observation.model_validate_json(obs.model_dump_json()) == obs
