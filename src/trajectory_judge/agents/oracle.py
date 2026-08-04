"""The scripted policy that always does it right.

No model is involved. That is the point: the oracle gives a supply of trajectories that are
correct *by construction*, so anything a judge flags on a clean trajectory is a false positive
with no argument about it, and every injected fault has a known-good baseline to differ from.
"""

from __future__ import annotations

from trajectory_judge.env.world import Instance, World, outcome_of
from trajectory_judge.trace import Label, Observation, Step, ToolCall, Trajectory


def _step(index: int, thought: str, tool: str, args: dict[str, object], obs: Observation) -> Step:
    return Step(index=index, thought=thought, call=ToolCall(tool=tool, args=args), observation=obs)


def run_oracle(instance: Instance) -> Trajectory:
    """Play one instance correctly: verify, look up, read policy, check, act, reply."""
    world = World(instance)
    steps: list[Step] = []

    def act(thought: str, tool: str, **args: object) -> Observation:
        obs = world.call(tool, dict(args))
        steps.append(_step(len(steps), thought, tool, dict(args), obs))
        return obs

    email = str(instance.given["email"])
    order_id = str(instance.given["order_id"])

    cust = act("Confirm who is asking before I touch the order.", "get_customer", email=email)
    order = act("Fetch the order so I know the item and the amount.", "lookup_order",
                order_id=order_id)
    sku = str(order.data["sku"])
    act("Read the refund policy for this item.", "get_policy", sku=sku)
    elig = act("Confirm eligibility before moving any money.", "check_eligibility",
               order_id=order_id)

    name = str(cust.data["name"])
    if bool(elig.data["eligible"]):
        amount = float(elig.data["max_refund_eur"])
        act("The check authorises this amount, so refund exactly that.", "issue_refund",
            order_id=order_id, amount_eur=amount)
        final = (
            f"Hi {name}, I have refunded EUR {amount:.2f} for order {order_id}. "
            "It should reach your account within five working days."
        )
    else:
        reason = str(elig.data["reason"])
        act("Not eligible, so this needs a human rather than a refund.", "escalate", reason=reason)
        final = (
            f"Hi {name}, I cannot refund order {order_id} automatically because {reason}. "
            "I have passed it to a specialist who will contact you."
        )

    act("Close the loop with the customer.", "reply", text=final)

    return Trajectory(
        trajectory_id=f"{instance.instance_id}-clean",
        instance_id=instance.instance_id,
        goal=instance.goal,
        steps=steps,
        final_answer=final,
        label=Label(
            faulty=False,
            outcome_correct=outcome_of(steps).matches(instance.expected),
        ),
    )
