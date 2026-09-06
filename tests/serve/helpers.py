"""Request builders for the service tests.

A separate module rather than the conftest: both `tests/` and `tests/serve/` land on
sys.path, so `import conftest` there would be ambiguous.
"""

from __future__ import annotations

from typing import Any


def payload(judge: str = "step", **over: Any) -> dict[str, Any]:
    """A minimal valid request body."""
    body: dict[str, Any] = {
        "trajectory": {
            "trajectory_id": "T-1",
            "goal": "refund order ORD-1",
            "steps": [],
            "final_answer": "refunded",
        },
        "judge": judge,
    }
    if judge in {"programmatic", "mock"}:
        body["context"] = {
            "given": {"order_id": "ORD-1"},
            "order": {"customer_id": "C-1", "status": "delivered"},
        }
    body.update(over)
    return body
