"""The trajectory schema, its ground-truth label, and the verdict a judge returns.

A trajectory is a goal, an ordered list of (thought, tool call, observation) steps, and a
final answer. Everything downstream — fault injection, judging, metrics — operates on this
one structure, so the schema is deliberately small and JSON-round-trippable.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class FailureType(str, Enum):
    """The six failure modes this project injects and asks judges to recognise.

    The definitions are mutually exclusive on purpose. Two pairs are easy to conflate, so
    each is pinned down by *what evidence would settle it*:

    - ``wrong_tool`` vs ``ignored_observation``: wrong_tool is a bad choice made with no
      contradicting evidence in hand; ignored_observation is an action that contradicts
      something an earlier observation already established.
    - ``hallucinated_argument`` vs ``unsupported_claim``: hallucinated_argument is ungrounded
      input *to a tool*; unsupported_claim is an ungrounded assertion *in the final answer*.
    """

    WRONG_TOOL = "wrong_tool"
    HALLUCINATED_ARGUMENT = "hallucinated_argument"
    SKIPPED_PRECONDITION = "skipped_precondition"
    IGNORED_OBSERVATION = "ignored_observation"
    PREMATURE_STOP = "premature_stop"
    UNSUPPORTED_CLAIM = "unsupported_claim"


class ToolCall(BaseModel):
    """A single tool invocation."""

    tool: str
    args: dict[str, Any] = Field(default_factory=dict)


class Observation(BaseModel):
    """What the environment returned for a tool call."""

    ok: bool
    data: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class Step(BaseModel):
    """One (thought, call, observation) triple. ``index`` is 0-based and dense."""

    index: int
    thought: str
    call: ToolCall
    observation: Observation


class Label(BaseModel):
    """Ground truth. Known by construction, never inferred.

    ``outcome_correct`` is the field the whole project turns on: a faulty trajectory whose
    final answer is still right is a *silent failure*, invisible to outcome-only evaluation.
    """

    faulty: bool
    failure_step: int | None = None
    failure_type: FailureType | None = None
    outcome_correct: bool = True

    @property
    def silent(self) -> bool:
        """A fault that survives outcome-only evaluation."""
        return self.faulty and self.outcome_correct


class Trajectory(BaseModel):
    """A full episode plus its ground-truth label."""

    trajectory_id: str
    instance_id: str
    goal: str
    steps: list[Step] = Field(default_factory=list)
    final_answer: str = ""
    label: Label = Field(default_factory=lambda: Label(faulty=False))

    def render(self, *, include_steps: bool = True) -> str:
        """Plain-text rendering fed to judges.

        ``include_steps=False`` produces the outcome-only view: the goal and the final answer
        and nothing else. That asymmetry is the experiment.
        """
        parts = [f"GOAL: {self.goal}"]
        if include_steps:
            parts.append("")
            for step in self.steps:
                args = ", ".join(f"{k}={v!r}" for k, v in step.call.args.items())
                obs = step.observation
                result = (
                    f"ok={obs.ok} {obs.data}" if obs.ok else f"ok=False error={obs.error!r}"
                )
                parts.append(f"[step {step.index}] thought: {step.thought}")
                parts.append(f"[step {step.index}] call: {step.call.tool}({args})")
                parts.append(f"[step {step.index}] observation: {result}")
        parts.append("")
        parts.append(f"FINAL ANSWER: {self.final_answer}")
        return "\n".join(parts)


class Verdict(BaseModel):
    """What a judge returns for one trajectory.

    ``confidence`` is in [0, 1] and is scored for calibration, so judges must report it even
    when they are guessing. ``latency_s`` and the token counts make the cost column of the
    results table real rather than estimated.
    """

    trajectory_id: str
    judge_id: str
    faulty: bool
    failure_step: int | None = None
    failure_type: FailureType | None = None
    confidence: float = 0.5
    rationale: str = ""
    latency_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: str | None = None
