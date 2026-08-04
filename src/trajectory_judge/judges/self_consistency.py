"""Sample the step-rubric judge k times and vote.

The expectation was that vote share would be a better confidence signal than a number a single
judge asserts about itself. On this benchmark it is not: at k=3 the ensemble matches the
single-pass judge on detection, loses slightly on silent recall and type attribution, and is
*worse* calibrated (ECE 0.084 against 0.033), because three votes can only ever express two
confidence levels and neither lands near the observed accuracy. It costs three times as much.

The row stays in the results table because a negative result that cost 3x is worth publishing,
and because it settles a question a single pass cannot: the failures it misses are missed by
all three samples, so they are systematic, not sampling noise.

Ties go to "clean". With an odd k there are none, but a conservative default matters more than
it looks: on a task where the base rate of faults is high, breaking ties toward "faulty" would
flatter the ensemble for free.
"""

from __future__ import annotations

from collections import Counter

from trajectory_judge.env.world import Instance
from trajectory_judge.judges.base import Judge
from trajectory_judge.judges.llm import StepRubricJudge
from trajectory_judge.judges.ollama_client import DEFAULT_HOST
from trajectory_judge.trace import Trajectory, Verdict


class SelfConsistencyJudge(Judge):
    def __init__(
        self,
        model: str = "qwen2.5:14b",
        *,
        k: int = 3,
        temperature: float = 0.7,
        base_seed: int = 7,
        host: str = DEFAULT_HOST,
    ) -> None:
        self.k = k
        self.judge_id = f"selfcons{k}:{model}"
        self._members = [
            StepRubricJudge(model, temperature=temperature, seed=base_seed + i, host=host)
            for i in range(k)
        ]

    def judge(self, trajectory: Trajectory, instance: Instance) -> Verdict:
        votes = [member.judge(trajectory, instance) for member in self._members]
        faulty_votes = [v for v in votes if v.faulty]
        faulty = len(faulty_votes) * 2 > self.k

        agreeing = faulty_votes if faulty else [v for v in votes if not v.faulty]
        confidence = len(agreeing) / self.k

        verdict = Verdict(
            trajectory_id=trajectory.trajectory_id,
            judge_id=self.judge_id,
            faulty=faulty,
            confidence=confidence,
            rationale=f"{len(faulty_votes)}/{self.k} votes for faulty",
            latency_s=sum(v.latency_s for v in votes),
            prompt_tokens=sum(v.prompt_tokens for v in votes),
            completion_tokens=sum(v.completion_tokens for v in votes),
            error=next((v.error for v in votes if v.error), None),
        )
        if faulty:
            verdict.failure_step = _mode([v.failure_step for v in faulty_votes])
            verdict.failure_type = _mode([v.failure_type for v in faulty_votes])
        return verdict


def _mode[T](values: list[T | None]) -> T | None:
    """Most common non-null value; ties broken by first appearance, so it stays deterministic."""
    present = [v for v in values if v is not None]
    if not present:
        return None
    counts = Counter(present)
    best = max(counts.values())
    for value in present:
        if counts[value] == best:
            return value
    return None
