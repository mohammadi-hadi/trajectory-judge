"""The judge interface.

Every judge answers the same four questions about a trajectory — is it faulty, at which step,
of which kind, and how sure are you — so their answers are directly comparable no matter how
they arrive at them.

One rule holds for all of them: a judge may read ``instance`` for world facts, but never
``instance.expected``. The rule engine reads the order and the policy because that is what a
rule engine does; the LLM judges are handed nothing but ``trajectory.render()``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from trajectory_judge.env.world import Instance
from trajectory_judge.trace import Trajectory, Verdict


class Judge(ABC):
    """Base class for anything that returns a :class:`Verdict`."""

    #: Stable identifier used as the key in results files and report rows.
    judge_id: str

    @abstractmethod
    def judge(self, trajectory: Trajectory, instance: Instance) -> Verdict:
        """Return a verdict for one trajectory. Must never raise: report errors in the verdict."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}(judge_id={self.judge_id!r})"
