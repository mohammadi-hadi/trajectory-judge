"""A real tool-calling agent playing the same environment.

Its purpose is a sanity check, not more training data: do the injected faults look like the
faults a model actually commits? If a model never skips the eligibility check and never invents
an argument, then a benchmark built out of those injections is measuring something imaginary.

**Organic runs are labelled heuristically and are reported separately.** The checker can only
see four of the six failure types, so a run it passes is not known to be clean the way an oracle
trajectory is. Mixing these into the judge comparison would quietly replace ground truth with
one of the systems under test.

One implementation note: the action schema uses flat, always-present fields rather than a
free-form ``args`` object. Constrained decoding handles a fixed set of typed fields reliably;
open-ended objects are where small models produce valid JSON with invented keys.
"""

from __future__ import annotations

from typing import Any

from trajectory_judge.env.checker import check
from trajectory_judge.env.world import TOOLS, Instance, World, outcome_of
from trajectory_judge.judges.llm import PROCEDURE
from trajectory_judge.judges.ollama_client import DEFAULT_HOST, generate
from trajectory_judge.trace import Label, Observation, Step, ToolCall, Trajectory

MAX_STEPS = 10

ACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "tool": {"type": "string", "enum": list(TOOLS)},
        "email": {"type": "string"},
        "order_id": {"type": "string"},
        "sku": {"type": "string"},
        "amount_eur": {"type": "number"},
        "text": {"type": "string"},
    },
    "required": ["reasoning", "tool", "email", "order_id", "sku", "amount_eur", "text"],
}

#: Which flat fields each tool actually consumes. Everything else in the response is ignored.
_ARGS_FOR: dict[str, tuple[str, ...]] = {
    "get_customer": ("email",),
    "lookup_order": ("order_id",),
    "get_policy": ("sku",),
    "check_eligibility": ("order_id",),
    "issue_refund": ("order_id", "amount_eur"),
    "escalate": ("text",),
    "reply": ("text",),
}

TOOL_REFERENCE = """\
get_customer(email) -> customer_id, name, phone_last4
lookup_order(order_id) -> customer_id, sku, total_eur, days_since_purchase, status
get_policy(sku) -> refundable, refund_window_days, restocking_fee_pct
check_eligibility(order_id) -> eligible, max_refund_eur, reason
issue_refund(order_id, amount_eur) -> refunded_eur
escalate(text) -> opens a ticket for a human
reply(text) -> sends the final message and ends the episode"""


def _prompt(instance: Instance, steps: list[Step]) -> str:
    history = "\n".join(
        f"[{s.index}] {s.call.tool}({s.call.args}) -> "
        + (str(s.observation.data) if s.observation.ok else f"ERROR {s.observation.error}")
        for s in steps
    )
    return f"""You are a customer-support agent for an online retailer.

{PROCEDURE}

Tools:
{TOOL_REFERENCE}

GOAL: {instance.goal}

Steps so far:
{history or "(none yet)"}

Choose the next single tool call. Fill only the fields that tool needs; leave the others as an
empty string or 0. When the case is resolved, call reply with the message for the customer.
"""


def run_llm_agent(
    instance: Instance,
    model: str = "qwen2.5:14b",
    *,
    max_steps: int = MAX_STEPS,
    seed: int = 7,
    host: str = DEFAULT_HOST,
) -> Trajectory:
    """Play one instance with a model in the loop. Never raises; a stalled episode just ends."""
    world = World(instance)
    steps: list[Step] = []
    final_answer = ""

    for index in range(max_steps):
        result = generate(
            model, _prompt(instance, steps), ACTION_SCHEMA, temperature=0.0, seed=seed, host=host
        )
        action = result.parse()
        if action is None:
            steps.append(
                Step(
                    index=index,
                    thought="(unparseable action)",
                    call=ToolCall(tool="none", args={}),
                    observation=Observation(ok=False, error=result.error or "unparseable"),
                )
            )
            break

        tool = str(action.get("tool", ""))
        args = {name: action[name] for name in _ARGS_FOR.get(tool, ()) if name in action}
        if tool in ("escalate", "reply"):
            args = {"reason" if tool == "escalate" else "text": str(action.get("text", ""))}

        steps.append(
            Step(
                index=index,
                thought=str(action.get("reasoning", ""))[:400],
                call=ToolCall(tool=tool, args=args),
                observation=world.call(tool, args),
            )
        )
        if tool == "reply":
            final_answer = str(action.get("text", ""))
            break

    violations = check(
        Trajectory(
            trajectory_id="probe",
            instance_id="probe",
            goal="",
            steps=steps,
            final_answer=final_answer,
        ),
        instance,
    )
    outcome_correct = outcome_of(steps).matches(instance.expected)
    first = violations[0] if violations else None

    return Trajectory(
        trajectory_id=f"{instance.instance_id}-agent",
        instance_id=instance.instance_id,
        goal=instance.goal,
        steps=steps,
        final_answer=final_answer,
        # Heuristic, not ground truth: the checker is blind to two of the six failure types.
        label=Label(
            faulty=bool(violations) or not outcome_correct,
            failure_step=first.step_index if first else None,
            failure_type=first.failure_type if first else None,
            outcome_correct=outcome_correct,
        ),
    )
