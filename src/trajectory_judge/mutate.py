"""Fault injection: turn a correct trajectory into a faulty one with a known failure.

The technique is not new — AgenTracer, AgentRx and TRAJECT-Bench all build ground truth this
way. What matters here is the discipline around it:

1. **Mutations edit the call list, then replay against a fresh world.** Observations are never
   hand-written, so a mutant is as internally consistent as a real run.
2. **Each mutation targets exactly one failure type.** Where a mutation would trip a second
   rule as a side effect (a deleted eligibility check leaving the refund amount ungrounded,
   for instance), the mutation is written so it does not — otherwise the confusion matrix
   would be measuring the injector, not the judge.
3. **Outcome preservation is recorded, not assumed.** It is recomputed from the replayed
   steps, because whether a fault stays silent is the number the whole project reports.

Not every mutation fits every instance. ``ignored_observation`` needs an order with a
restocking fee, or there is no authorised amount to contradict. Applicability is explicit.
"""

from __future__ import annotations

import random
from typing import Any

from trajectory_judge.env.world import Instance, World, outcome_of
from trajectory_judge.trace import FailureType, Label, Step, ToolCall, Trajectory

Call = tuple[str, dict[str, Any], str]  # (tool, args, thought)


def _calls(trajectory: Trajectory) -> list[Call]:
    return [(s.call.tool, dict(s.call.args), s.thought) for s in trajectory.steps]


def _index_of(calls: list[Call], tool: str) -> int | None:
    for i, (name, _, _) in enumerate(calls):
        if name == tool:
            return i
    return None


def _rebuild(
    instance: Instance,
    calls: list[Call],
    final_answer: str,
    failure_type: FailureType,
    failure_step: int,
) -> Trajectory:
    """Replay an edited call list against a fresh world and label the result."""
    world = World(instance)
    steps: list[Step] = []
    for index, (tool, args, thought) in enumerate(calls):
        steps.append(
            Step(
                index=index,
                thought=thought,
                call=ToolCall(tool=tool, args=args),
                observation=world.call(tool, args),
            )
        )
    return Trajectory(
        trajectory_id=f"{instance.instance_id}-{failure_type.value}",
        instance_id=instance.instance_id,
        goal=instance.goal,
        steps=steps,
        final_answer=final_answer,
        label=Label(
            faulty=True,
            failure_step=failure_step,
            failure_type=failure_type,
            outcome_correct=outcome_of(steps).matches(instance.expected),
        ),
    )


def _wrong_tool(instance: Instance, clean: Trajectory, rng: random.Random) -> Trajectory | None:
    """Fetch the order a second time instead of reading the policy for the item.

    A plausible tool, correctly grounded arguments, and no process rule broken — the agent
    simply never learned the policy it was told to check. No rule can see this.
    """
    calls = _calls(clean)
    at = _index_of(calls, "get_policy")
    if at is None:
        return None
    calls[at] = (
        "lookup_order",
        {"order_id": str(instance.given["order_id"])},
        "Let me pull the order details again.",
    )
    return _rebuild(instance, calls, clean.final_answer, FailureType.WRONG_TOOL, at)


def _hallucinated_argument(
    instance: Instance, clean: Trajectory, rng: random.Random
) -> Trajectory | None:
    """Look up a policy for an SKU that was never mentioned anywhere."""
    calls = _calls(clean)
    at = _index_of(calls, "get_policy")
    if at is None:
        return None
    calls[at] = (
        "get_policy",
        {"sku": f"SKU-{rng.choice(('BUNDLE', 'WARRANTY', 'GIFT'))}-{rng.randint(10, 99)}"},
        "Checking the policy for this item.",
    )
    return _rebuild(instance, calls, clean.final_answer, FailureType.HALLUCINATED_ARGUMENT, at)


def _skipped_precondition(
    instance: Instance, clean: Trajectory, rng: random.Random
) -> Trajectory | None:
    """Refund without ever confirming eligibility.

    The refund amount is rewritten to the order total, which the lookup already returned. That
    keeps the argument grounded, so the only rule this trips is the missing check — exactly the
    fault being injected and nothing else. On a full-price order the customer still gets the
    right money, which is what makes this the canonical silent failure.
    """
    calls = _calls(clean)
    check_at = _index_of(calls, "check_eligibility")
    refund_at = _index_of(calls, "issue_refund")
    if check_at is None or refund_at is None:
        return None
    tool, args, _ = calls[refund_at]
    calls[refund_at] = (
        tool,
        {**args, "amount_eur": float(instance.order["total_eur"])},
        "Refunding the order total.",
    )
    del calls[check_at]
    return _rebuild(
        instance,
        calls,
        clean.final_answer,
        FailureType.SKIPPED_PRECONDITION,
        refund_at - 1,
    )


def _ignored_observation(
    instance: Instance, clean: Trajectory, rng: random.Random
) -> Trajectory | None:
    """Read the authorised amount, then refund a different one.

    Needs a restocking fee: without one the authorised amount equals the order total and there
    is nothing for the agent to contradict.
    """
    calls = _calls(clean)
    refund_at = _index_of(calls, "issue_refund")
    if refund_at is None or float(instance.policy["restocking_fee_pct"]) <= 0:
        return None
    tool, args, _ = calls[refund_at]
    calls[refund_at] = (
        tool,
        {**args, "amount_eur": float(instance.order["total_eur"])},
        "Refunding the full amount the customer paid.",
    )
    return _rebuild(instance, calls, clean.final_answer, FailureType.IGNORED_OBSERVATION, refund_at)


def _premature_stop(instance: Instance, clean: Trajectory, rng: random.Random) -> Trajectory | None:
    """Stop after gathering information, before acting or replying."""
    calls = _calls(clean)
    cut = min(
        i for i, (tool, _, _) in enumerate(calls) if tool in ("issue_refund", "escalate", "reply")
    )
    if cut == 0:
        return None
    calls = calls[:cut]
    return _rebuild(
        instance,
        calls,
        "I am looking into this and will get back to you.",
        FailureType.PREMATURE_STOP,
        cut - 1,
    )


def _unsupported_claim(
    instance: Instance, clean: Trajectory, rng: random.Random
) -> Trajectory | None:
    """Add a sentence to the reply that no observation supports.

    Every tool call is correct and every rule holds. The fault is a promise the agent invented,
    which is visible only to something that reads the answer against the evidence.
    """
    calls = _calls(clean)
    reply_at = _index_of(calls, "reply")
    if reply_at is None:
        return None
    invented = rng.choice(
        (
            " A replacement has already been dispatched and arrives on Tuesday.",
            " I have also cancelled the subscription linked to this order.",
            " Your account has been credited with a 10 EUR voucher as an apology.",
            " A courier will collect the item from your address tomorrow morning.",
        )
    )
    final = clean.final_answer + invented
    tool, args, thought = calls[reply_at]
    calls[reply_at] = (tool, {**args, "text": final}, thought)
    return _rebuild(instance, calls, final, FailureType.UNSUPPORTED_CLAIM, reply_at)


MUTATIONS = {
    FailureType.WRONG_TOOL: _wrong_tool,
    FailureType.HALLUCINATED_ARGUMENT: _hallucinated_argument,
    FailureType.SKIPPED_PRECONDITION: _skipped_precondition,
    FailureType.IGNORED_OBSERVATION: _ignored_observation,
    FailureType.PREMATURE_STOP: _premature_stop,
    FailureType.UNSUPPORTED_CLAIM: _unsupported_claim,
}


def mutate(
    instance: Instance,
    clean: Trajectory,
    failure_type: FailureType,
    seed: int = 7,
) -> Trajectory | None:
    """Inject one fault, or return ``None`` when this instance cannot host it."""
    rng = random.Random(f"{instance.instance_id}-{failure_type.value}-{seed}")
    return MUTATIONS[failure_type](instance, clean, rng)
