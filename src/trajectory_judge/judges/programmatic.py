"""The rule engine as a judge: free, instant, and blind in two places.

Its confidence is asymmetric on purpose. When a rule fires, the verdict is close to certain —
the violation is structural. When no rule fires, "clean" is a weak claim, because two of the
six failure types are outside what rules can express at all. Reporting that asymmetry honestly
is what makes its calibration number meaningful rather than decorative.
"""

from __future__ import annotations

import time

from trajectory_judge.env.checker import check
from trajectory_judge.env.world import Instance
from trajectory_judge.judges.base import Judge
from trajectory_judge.trace import Trajectory, Verdict

#: A fired rule is structural evidence; silence is only weak evidence of innocence.
CONFIDENCE_WHEN_FLAGGED = 0.95
CONFIDENCE_WHEN_CLEAN = 0.60


class ProgrammaticJudge(Judge):
    judge_id = "programmatic"

    def judge(self, trajectory: Trajectory, instance: Instance) -> Verdict:
        started = time.perf_counter()
        violations = check(trajectory, instance)
        elapsed = time.perf_counter() - started

        if not violations:
            return Verdict(
                trajectory_id=trajectory.trajectory_id,
                judge_id=self.judge_id,
                faulty=False,
                confidence=CONFIDENCE_WHEN_CLEAN,
                rationale="no process rule fired; two failure types are outside rule coverage",
                latency_s=elapsed,
            )

        first = violations[0]
        return Verdict(
            trajectory_id=trajectory.trajectory_id,
            judge_id=self.judge_id,
            faulty=True,
            failure_step=first.step_index,
            failure_type=first.failure_type,
            confidence=CONFIDENCE_WHEN_FLAGGED,
            rationale="; ".join(f"{v.rule}: {v.message}" for v in violations[:3]),
            latency_s=elapsed,
        )
