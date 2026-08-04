"""The judges under comparison."""

from trajectory_judge.judges.base import Judge
from trajectory_judge.judges.mock import MockJudge
from trajectory_judge.judges.programmatic import ProgrammaticJudge

__all__ = ["Judge", "MockJudge", "ProgrammaticJudge"]
