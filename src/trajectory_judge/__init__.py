"""Measuring what LLM judges miss when an agent reaches the right answer the wrong way.

The ``trajectory-judge`` console script drives the full pipeline. This module re-exports
the data model and the judges, which is enough to score trajectories of your own: build
``Trajectory`` objects, hand them to a judge, collect ``Verdict`` objects. The synthetic
environment and the fault injector stay in their submodules (``trajectory_judge.env``,
``trajectory_judge.mutate``) because they are the experiment, not the API.
"""

from trajectory_judge.judges import (
    Judge,
    LlmJudge,
    MockJudge,
    OutcomeJudge,
    ProgrammaticJudge,
    SelfConsistencyJudge,
    StepRubricJudge,
)
from trajectory_judge.trace import (
    FailureType,
    Label,
    Observation,
    Step,
    ToolCall,
    Trajectory,
    Verdict,
)

__version__ = "0.1.0"

__all__ = [
    "FailureType",
    "Judge",
    "Label",
    "LlmJudge",
    "MockJudge",
    "Observation",
    "OutcomeJudge",
    "ProgrammaticJudge",
    "SelfConsistencyJudge",
    "Step",
    "StepRubricJudge",
    "ToolCall",
    "Trajectory",
    "Verdict",
]
