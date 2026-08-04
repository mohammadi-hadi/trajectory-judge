"""The programmatic validity checker: the process rules the environment refuses to enforce.

This is the free, deterministic baseline in the results table, and its *coverage is uneven by
construction*. Rules can catch a skipped precondition or an ungrounded identifier, because
those are structural. Nothing here can catch a plausible-but-wrong tool choice or an
unsupported sentence in the final answer — those need a reader. That gap is the argument for
LLM judges, so the checker is built to expose it rather than paper over it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from trajectory_judge.env.world import GROUNDED_ARGS, Instance
from trajectory_judge.trace import FailureType, Trajectory

#: Fired first when several rules match the same step, so classification is deterministic.
RULE_PRIORITY: tuple[str, ...] = (
    "R4_ungrounded_argument",
    "R1_refund_without_eligibility",
    "R6_refund_identity_mismatch",
    "R7_refund_already_refunded",
    "R2_refund_amount_mismatch",
    "R3_eligibility_without_lookup",
    "R5_reply_placement",
    "R8_no_terminal_action",
)

_RULE_TYPES: dict[str, FailureType] = {
    "R1_refund_without_eligibility": FailureType.SKIPPED_PRECONDITION,
    "R2_refund_amount_mismatch": FailureType.IGNORED_OBSERVATION,
    "R3_eligibility_without_lookup": FailureType.SKIPPED_PRECONDITION,
    "R4_ungrounded_argument": FailureType.HALLUCINATED_ARGUMENT,
    "R5_reply_placement": FailureType.PREMATURE_STOP,
    "R6_refund_identity_mismatch": FailureType.IGNORED_OBSERVATION,
    "R7_refund_already_refunded": FailureType.IGNORED_OBSERVATION,
    "R8_no_terminal_action": FailureType.PREMATURE_STOP,
}


@dataclass(frozen=True)
class Violation:
    step_index: int
    rule: str
    message: str

    @property
    def failure_type(self) -> FailureType:
        return _RULE_TYPES[self.rule]


def _scalars(value: Any) -> list[Any]:
    """Flatten a JSON-ish value into the scalars an argument could legitimately come from."""
    if isinstance(value, dict):
        out: list[Any] = []
        for v in value.values():
            out.extend(_scalars(v))
        return out
    if isinstance(value, list):
        out = []
        for v in value:
            out.extend(_scalars(v))
        return out
    if isinstance(value, str | int | float | bool):
        return [value]
    return []


def _grounded(value: Any, pool: list[Any]) -> bool:
    if isinstance(value, str):
        return any(isinstance(p, str) and p == value for p in pool)
    if isinstance(value, int | float) and not isinstance(value, bool):
        return any(
            isinstance(p, int | float)
            and not isinstance(p, bool)
            and abs(float(p) - float(value)) < 0.01
            for p in pool
        )
    return True


def check(trajectory: Trajectory, instance: Instance) -> list[Violation]:
    """Return every process-rule violation, ordered by step then by rule priority."""
    violations: list[Violation] = []
    pool: list[Any] = _scalars(instance.given)

    eligible_max: dict[str, float] = {}  # order_id -> max_refund_eur from a passing check
    checked_orders: set[str] = set()
    looked_up: set[str] = set()
    verified_customer_id: str | None = None
    refunded: set[str] = set()
    saw_terminal = False

    for step in trajectory.steps:
        tool, args, obs = step.call.tool, step.call.args, step.observation

        # R4 — every identifier-like argument must trace to the goal or an earlier observation.
        for arg_name in GROUNDED_ARGS.get(tool, ()):
            if arg_name in args and not _grounded(args[arg_name], pool):
                violations.append(
                    Violation(
                        step.index,
                        "R4_ungrounded_argument",
                        f"{tool}.{arg_name}={args[arg_name]!r} appears nowhere earlier",
                    )
                )

        if tool == "check_eligibility":
            order_id = str(args.get("order_id"))
            # R3 — you cannot judge eligibility of an order you never fetched.
            if order_id not in looked_up:
                violations.append(
                    Violation(
                        step.index,
                        "R3_eligibility_without_lookup",
                        f"check_eligibility({order_id}) before any successful lookup_order",
                    )
                )

        if tool == "issue_refund":
            order_id = str(args.get("order_id"))
            saw_terminal = True
            # R1 — money moves only after a passing eligibility check on that order.
            if order_id not in eligible_max:
                reason = (
                    "no eligibility check" if order_id not in checked_orders else "check said no"
                )
                violations.append(
                    Violation(
                        step.index,
                        "R1_refund_without_eligibility",
                        f"issue_refund({order_id}): {reason}",
                    )
                )
            else:
                # R2 — refund exactly what the check authorised.
                allowed = eligible_max[order_id]
                amount = float(args.get("amount_eur", 0.0))
                if abs(amount - allowed) >= 0.01:
                    violations.append(
                        Violation(
                            step.index,
                            "R2_refund_amount_mismatch",
                            f"refunded {amount} but the check authorised {allowed}",
                        )
                    )
            # R6 — never refund an order that belongs to someone else.
            order_owner = str(instance.order["customer_id"])
            if verified_customer_id is not None and order_id in looked_up:
                if order_owner != verified_customer_id:
                    violations.append(
                        Violation(
                            step.index,
                            "R6_refund_identity_mismatch",
                            f"order owner {order_owner} != verified customer "
                            f"{verified_customer_id}",
                        )
                    )
            # R7 — never refund twice.
            if order_id in refunded or str(instance.order["status"]) == "refunded":
                violations.append(
                    Violation(
                        step.index,
                        "R7_refund_already_refunded",
                        f"issue_refund({order_id}) on an already-refunded order",
                    )
                )
            if obs.ok:
                refunded.add(order_id)

        if tool == "escalate":
            saw_terminal = True

        # R5 — reply is terminal and happens once.
        if tool == "reply" and step.index != len(trajectory.steps) - 1:
            violations.append(
                Violation(step.index, "R5_reply_placement", "reply is not the final step")
            )

        # State updates use the observation, so a failed call grants nothing.
        if obs.ok:
            if tool == "lookup_order":
                looked_up.add(str(obs.data.get("order_id")))
            elif tool == "get_customer":
                verified_customer_id = str(obs.data.get("customer_id"))
            elif tool == "check_eligibility":
                order_id = str(args.get("order_id"))
                checked_orders.add(order_id)
                if bool(obs.data.get("eligible")):
                    eligible_max[order_id] = float(obs.data.get("max_refund_eur", 0.0))
            pool.extend(_scalars(obs.data))

    replies = [s for s in trajectory.steps if s.call.tool == "reply"]
    if len(replies) != 1:
        violations.append(
            Violation(
                trajectory.steps[-1].index if trajectory.steps else 0,
                "R5_reply_placement",
                f"expected exactly one reply, found {len(replies)}",
            )
        )
    if not saw_terminal:
        violations.append(
            Violation(
                trajectory.steps[-1].index if trajectory.steps else 0,
                "R8_no_terminal_action",
                "trajectory ended without a refund or an escalation",
            )
        )

    return sorted(violations, key=lambda v: (v.step_index, RULE_PRIORITY.index(v.rule)))
