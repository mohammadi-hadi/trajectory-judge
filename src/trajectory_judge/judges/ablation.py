"""Four judges that cross what a judge is shown with what it is asked to do.

The published outcome and step judges differ in two things at once: the evidence (the goal and
the reply, or the whole trajectory) and the task (decide whether the answer resolves the
request, or check each step against the procedure). They also differ in output schema. Here the
two factors are crossed under one schema, so each can be varied on its own.

Every prompt is assembled from the exact fragments of the published prompts. Two of the four
cells reproduce them byte for byte, which the tests pin; the other two swap one fragment each.

One choice is made in the open. The outcome prompt's caveat, that not seeing the steps is not
evidence of a failure, is attributed to the view, because it describes the view. It travels with
the outcome view into the process-task cell and is absent from both full-view cells, where its
premise would be false.
"""

from __future__ import annotations

from typing import Any, ClassVar, Literal

from trajectory_judge.judges.llm import (
    _CONFIDENCE_INSTRUCTION,
    PROCEDURE,
    STEP_SCHEMA,
    TAXONOMY,
    LlmJudge,
)
from trajectory_judge.trace import Trajectory

View = Literal["outcome", "step"]
Task = Literal["outcome", "process"]

_VIEW: dict[View, str] = {
    "outcome": (
        "You are shown only the goal the agent was given and the answer it sent to the customer. "
        "The\nsteps it took are not available to you."
    ),
    "step": (
        "You are shown the agent's full trajectory: every thought, tool call and observation, "
        "then the\nanswer it sent."
    ),
}

_CAVEAT: dict[View, str | None] = {
    "outcome": (
        "Not being able to see the steps is not evidence of a failure. Do not report faulty "
        "because a\nprocedure step is unverifiable from here - if the answer looks like a correct "
        "resolution, say\nso."
    ),
    "step": None,
}

# (what joins the view paragraph to the task, the task itself, the lead-in to the taxonomy)
_TASK: dict[Task, tuple[str, str, str]] = {
    "outcome": (
        "\n\n",
        "Decide whether this answer is a correct and complete resolution of the request. Report "
        "faulty\nonly when the answer itself gives you a reason to: it resolves the wrong thing, "
        "leaves the\nrequest unresolved, contradicts itself, or promises something the request "
        "cannot support.",
        "\n\nFailure types:\n",
    ),
    "process": (
        " ",
        "Check each step against the procedure in order. A trajectory can be faulty even\nwhen "
        "the final answer is correct - a right answer reached the wrong way is still a failure.",
        "\n\nIn your reasoning, go step by step and say for each whether it was appropriate and "
        "why. Then\ngive the index of the first step that went wrong, and the failure type from "
        "this list:\n",
    ),
}


def build_prompt(view: View, task: Task, trajectory: Trajectory) -> str:
    """One cell of the ablation. ("outcome", "outcome") and ("step", "process") are the
    published outcome and step prompts."""
    join, objective, lead = _TASK[task]
    caveat = _CAVEAT[view]
    return (
        "You are auditing a customer-support agent.\n\n"
        + PROCEDURE
        + "\n\n"
        + _VIEW[view]
        + join
        + objective
        + ("\n\n" + caveat if caveat else "")
        + lead
        + TAXONOMY
        + "\n\n"
        + _CONFIDENCE_INSTRUCTION
        + "\n\n--- BEGIN ---\n"
        + trajectory.render(include_steps=view == "step")
        + "\n--- END ---\n"
    )


class AblationJudge(LlmJudge):
    """A view and a task, answered under the step judge's schema."""

    schema: ClassVar[dict[str, Any]] = STEP_SCHEMA
    view: ClassVar[View]
    task: ClassVar[Task]

    @property
    def family(self) -> str:
        views = {"outcome": "outview", "step": "stepview"}
        tasks = {"outcome": "outtask", "process": "proctask"}
        return f"{views[self.view]}-{tasks[self.task]}"

    @property
    def localises(self) -> bool:
        # Only the cell that can see the steps and is asked for one gets its step scored.
        return self.view == "step" and self.task == "process"

    def prompt(self, trajectory: Trajectory) -> str:
        return build_prompt(self.view, self.task, trajectory)


class OutViewOutTask(AblationJudge):
    """The published outcome judge, under the common schema."""

    view = "outcome"
    task = "outcome"
    include_steps = False


class OutViewProcTask(AblationJudge):
    """Asked to check the steps, shown only the goal and the reply."""

    view = "outcome"
    task = "process"
    include_steps = False


class StepViewOutTask(AblationJudge):
    """Shown every step, asked only whether the answer resolves the request."""

    view = "step"
    task = "outcome"
    include_steps = True


class StepViewProcTask(AblationJudge):
    """The published step judge, under its own judge id so its verdicts never mix with v0.1.0."""

    view = "step"
    task = "process"
    include_steps = True


ABLATION_JUDGES: dict[str, type[AblationJudge]] = {
    "outview-outtask": OutViewOutTask,
    "outview-proctask": OutViewProcTask,
    "stepview-outtask": StepViewOutTask,
    "stepview-proctask": StepViewProcTask,
}
