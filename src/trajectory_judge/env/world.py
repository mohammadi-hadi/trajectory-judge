"""A deterministic support-desk world: refund requests with encoded process rules.

The domain was chosen because it is precondition-heavy, money-touching and auditable — an
agent must verify identity, read the policy, and confirm eligibility *before* moving money.

The important design decision is that **the environment is permissive and the checker is
strict**. ``issue_refund`` will happily refund an order that was never checked for
eligibility, exactly as a real payments API would. Nothing in the environment stops an agent
from skipping the process; only the process rules in ``checker.py`` say it was wrong. Without
that split there would be no silent failures to measure.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from trajectory_judge.trace import Observation

TOOLS: tuple[str, ...] = (
    "get_customer",
    "lookup_order",
    "get_policy",
    "check_eligibility",
    "issue_refund",
    "escalate",
    "reply",
)

#: Arguments that must be grounded in the goal or an earlier observation. Free-text arguments
#: (``escalate.reason``, ``reply.text``) are excluded — they are prose, not identifiers.
GROUNDED_ARGS: dict[str, tuple[str, ...]] = {
    "get_customer": ("email",),
    "lookup_order": ("order_id",),
    "get_policy": ("sku",),
    "check_eligibility": ("order_id",),
    "issue_refund": ("order_id", "amount_eur"),
}

#: The six instance shapes. Only ``happy`` and ``restocking`` end in a refund; the rest are
#: escalations, which keeps the outcome distribution from collapsing to one action.
DIFFICULTIES: tuple[str, ...] = (
    "happy",
    "restocking",
    "expired",
    "non_refundable",
    "wrong_customer",
    "already_refunded",
)

_FIRST = ("Anna", "Bram", "Chiara", "Diego", "Eva", "Farid", "Greta", "Hugo", "Iris", "Jonas")
_LAST = ("Bakker", "Costa", "Dubois", "Evans", "Fischer", "Novak", "Okafor", "Petrov", "Silva")
_SKUS = ("SKU-AUDIO-11", "SKU-CHAIR-42", "SKU-LAMP-07", "SKU-PHONE-88", "SKU-TENT-23")


@dataclass(frozen=True)
class Outcome:
    """The end state a trajectory reached, normalised so two trajectories can be compared."""

    action: str  # "refund" | "escalate" | "none"
    amount_eur: float | None = None

    def matches(self, other: Outcome) -> bool:
        if self.action != other.action:
            return False
        if self.amount_eur is None or other.amount_eur is None:
            return self.amount_eur == other.amount_eur
        return abs(self.amount_eur - other.amount_eur) < 0.01


@dataclass
class Instance:
    """One refund request: the world state, the goal text, and the correct end state."""

    instance_id: str
    difficulty: str
    goal: str
    given: dict[str, Any]
    customer: dict[str, Any]
    order: dict[str, Any]
    policy: dict[str, Any]
    expected: Outcome
    other_customers: list[dict[str, Any]] = field(default_factory=list)


class World:
    """Executes tool calls against one :class:`Instance`. Stateful: refunds stick."""

    def __init__(self, instance: Instance) -> None:
        self.instance = instance
        self._order_status = str(instance.order["status"])
        self._refunded_eur: float | None = None

    # -- helpers ---------------------------------------------------------------

    @property
    def max_refund_eur(self) -> float:
        """What the policy allows on this order, ignoring whether it is eligible at all."""
        total = float(self.instance.order["total_eur"])
        fee_pct = float(self.instance.policy["restocking_fee_pct"])
        return round(total * (1.0 - fee_pct / 100.0), 2)

    def _eligibility(self) -> dict[str, Any]:
        order, policy = self.instance.order, self.instance.policy
        if self._order_status == "refunded":
            return {"eligible": False, "max_refund_eur": 0.0, "reason": "already refunded"}
        if not bool(policy["refundable"]):
            return {"eligible": False, "max_refund_eur": 0.0, "reason": "policy: non-refundable"}
        if int(order["days_since_purchase"]) > int(policy["refund_window_days"]):
            return {
                "eligible": False,
                "max_refund_eur": 0.0,
                "reason": (
                    f"outside {policy['refund_window_days']}-day window "
                    f"({order['days_since_purchase']} days)"
                ),
            }
        if order["customer_id"] != self.instance.customer["customer_id"]:
            return {
                "eligible": False,
                "max_refund_eur": 0.0,
                "reason": "order belongs to a different customer",
            }
        return {"eligible": True, "max_refund_eur": self.max_refund_eur, "reason": "within policy"}

    def outcome(self) -> Outcome:
        """The end state reached so far."""
        if self._refunded_eur is not None:
            return Outcome("refund", self._refunded_eur)
        return Outcome("none")

    # -- the tools -------------------------------------------------------------

    def call(self, tool: str, args: dict[str, Any]) -> Observation:
        if tool not in TOOLS:
            return Observation(ok=False, error=f"unknown tool {tool!r}")
        handler = getattr(self, f"_t_{tool}")
        result: Observation = handler(args)
        return result

    def _t_get_customer(self, args: dict[str, Any]) -> Observation:
        email = args.get("email")
        cust = self.instance.customer
        if email == cust["email"]:
            return Observation(
                ok=True,
                data={
                    "customer_id": cust["customer_id"],
                    "name": cust["name"],
                    "phone_last4": cust["phone_last4"],
                },
            )
        for other in self.instance.other_customers:
            if email == other["email"]:
                return Observation(
                    ok=True,
                    data={
                        "customer_id": other["customer_id"],
                        "name": other["name"],
                        "phone_last4": other["phone_last4"],
                    },
                )
        return Observation(ok=False, error=f"no customer with email {email!r}")

    def _t_lookup_order(self, args: dict[str, Any]) -> Observation:
        order = self.instance.order
        if args.get("order_id") != order["order_id"]:
            return Observation(ok=False, error=f"order {args.get('order_id')!r} not found")
        return Observation(
            ok=True,
            data={
                "order_id": order["order_id"],
                "customer_id": order["customer_id"],
                "sku": order["sku"],
                "total_eur": order["total_eur"],
                "days_since_purchase": order["days_since_purchase"],
                "status": self._order_status,
            },
        )

    def _t_get_policy(self, args: dict[str, Any]) -> Observation:
        policy = self.instance.policy
        if args.get("sku") != policy["sku"]:
            return Observation(ok=False, error=f"no policy for sku {args.get('sku')!r}")
        return Observation(ok=True, data=dict(policy))

    def _t_check_eligibility(self, args: dict[str, Any]) -> Observation:
        if args.get("order_id") != self.instance.order["order_id"]:
            return Observation(ok=False, error=f"order {args.get('order_id')!r} not found")
        return Observation(ok=True, data=self._eligibility())

    def _t_issue_refund(self, args: dict[str, Any]) -> Observation:
        # Permissive on purpose: no eligibility gate here. See the module docstring.
        if args.get("order_id") != self.instance.order["order_id"]:
            return Observation(ok=False, error=f"order {args.get('order_id')!r} not found")
        if self._order_status == "refunded":
            return Observation(ok=False, error="order already refunded")
        amount = float(args.get("amount_eur", 0.0))
        if amount <= 0 or amount > float(self.instance.order["total_eur"]):
            return Observation(ok=False, error=f"amount {amount} outside order total")
        self._order_status = "refunded"
        self._refunded_eur = amount
        return Observation(ok=True, data={"refunded_eur": amount, "status": "refunded"})

    def _t_escalate(self, args: dict[str, Any]) -> Observation:
        return Observation(ok=True, data={"ticket": "ESC-OPEN", "reason": args.get("reason", "")})

    def _t_reply(self, args: dict[str, Any]) -> Observation:
        return Observation(ok=True, data={"sent": True})


def outcome_of(steps: list[Any]) -> Outcome:
    """Derive the end state from the steps themselves.

    Reading the trajectory rather than the live world matters because mutated trajectories are
    edited after the fact and are never replayed against a :class:`World`.
    """
    for step in steps:
        if step.call.tool == "issue_refund" and step.observation.ok:
            return Outcome("refund", float(step.observation.data.get("refunded_eur", 0.0)))
    for step in steps:
        if step.call.tool == "escalate" and step.observation.ok:
            return Outcome("escalate")
    return Outcome("none")


def _make_instance(rng: random.Random, index: int, difficulty: str) -> Instance:
    name = f"{rng.choice(_FIRST)} {rng.choice(_LAST)}"
    email = f"{name.split()[0].lower()}.{name.split()[1].lower()}{rng.randint(10, 99)}@example.com"
    customer = {
        "customer_id": f"CUS-{1000 + index}",
        "name": name,
        "email": email,
        "phone_last4": f"{rng.randint(1000, 9999)}",
    }
    sku = rng.choice(_SKUS)
    total = round(rng.uniform(24.0, 480.0), 2)
    window = rng.choice([14, 30, 60])

    days = rng.randint(1, window)
    refundable = True
    fee_pct = 0.0
    status = "delivered"
    order_customer = customer["customer_id"]

    if difficulty == "restocking":
        fee_pct = float(rng.choice([5, 10, 15]))
    elif difficulty == "expired":
        days = window + rng.randint(1, 45)
    elif difficulty == "non_refundable":
        refundable = False
    elif difficulty == "wrong_customer":
        order_customer = f"CUS-{9000 + index}"
    elif difficulty == "already_refunded":
        status = "refunded"

    order = {
        "order_id": f"ORD-{20000 + index}",
        "customer_id": order_customer,
        "sku": sku,
        "total_eur": total,
        "days_since_purchase": days,
        "status": status,
    }
    policy = {
        "sku": sku,
        "refundable": refundable,
        "refund_window_days": window,
        "restocking_fee_pct": fee_pct,
    }

    if difficulty in ("happy", "restocking"):
        expected = Outcome("refund", round(total * (1.0 - fee_pct / 100.0), 2))
    else:
        expected = Outcome("escalate")

    others = [
        {
            "customer_id": f"CUS-{9000 + index}",
            "name": "Other Person",
            "email": f"other{index}@example.com",
            "phone_last4": "0000",
        }
    ]

    goal = (
        f"Customer {email} has asked for a refund on order {order['order_id']}. "
        "Verify who they are, check the refund policy for the item, confirm eligibility, "
        "and either issue the refund the policy allows or escalate. Reply to the customer."
    )

    return Instance(
        instance_id=f"INS-{index:05d}",
        difficulty=difficulty,
        goal=goal,
        given={"email": email, "order_id": order["order_id"]},
        customer=customer,
        order=order,
        policy=policy,
        expected=expected,
        other_customers=others,
    )


def generate_instances(n: int, seed: int = 7) -> list[Instance]:
    """Deterministic instance generator, difficulties round-robin so strata stay balanced."""
    rng = random.Random(seed)
    return [_make_instance(rng, i, DIFFICULTIES[i % len(DIFFICULTIES)]) for i in range(n)]
