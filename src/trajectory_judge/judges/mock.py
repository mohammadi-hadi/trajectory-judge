"""A deterministic stand-in so the pipeline can be exercised with no model and no network.

This is **not** a model and its numbers mean nothing about judging. It takes the rule engine's
verdict and corrupts it at a fixed rate, seeded by trajectory id, purely so that CI and
``make demo`` produce a non-degenerate report — a confusion matrix with off-diagonal mass and a
calibration curve with more than one point. Every results table in the README comes from real
models; the mock exists so that a fresh clone runs end to end in under a minute.
"""

from __future__ import annotations

import hashlib

from trajectory_judge.env.world import Instance
from trajectory_judge.judges.base import Judge
from trajectory_judge.judges.programmatic import ProgrammaticJudge
from trajectory_judge.trace import FailureType, Trajectory, Verdict

_TYPES = list(FailureType)


def _unit(trajectory_id: str, salt: str) -> float:
    """A stable pseudo-random number in [0, 1) — same input, same value, forever."""
    digest = hashlib.sha256(f"{trajectory_id}:{salt}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


class MockJudge(Judge):
    judge_id = "mock"

    def __init__(self, noise: float = 0.25) -> None:
        self.noise = noise
        self._inner = ProgrammaticJudge()

    def judge(self, trajectory: Trajectory, instance: Instance) -> Verdict:
        verdict = self._inner.judge(trajectory, instance)
        verdict.judge_id = self.judge_id

        if _unit(trajectory.trajectory_id, "flip") < self.noise:
            verdict.faulty = not verdict.faulty
            if verdict.faulty:
                at = int(_unit(trajectory.trajectory_id, "step") * max(len(trajectory.steps), 1))
                verdict.failure_step = at
                verdict.failure_type = _TYPES[
                    int(_unit(trajectory.trajectory_id, "type") * len(_TYPES))
                ]
            else:
                verdict.failure_step = None
                verdict.failure_type = None

        verdict.confidence = round(0.5 + 0.49 * _unit(trajectory.trajectory_id, "conf"), 3)
        verdict.rationale = "mock judge: deterministic stand-in, not a model"
        return verdict
