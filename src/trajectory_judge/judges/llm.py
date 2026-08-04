"""The two LLM judges under comparison, and the only thing that differs between them.

Both are given the same standard operating procedure and the same failure taxonomy. The
outcome-only judge is shown the goal and the final answer. The step-rubric judge is shown the
whole trajectory. Everything else — model, temperature, seed, schema — is held constant, so a
difference in their scores is a difference in evidence, not in prompting effort.

That is the experiment. It is also the comparison most production evaluation stacks never run,
because the outcome-only judge is cheap and its misses are by definition invisible.
"""

from __future__ import annotations

from typing import Any

from trajectory_judge.env.world import Instance
from trajectory_judge.judges.base import Judge
from trajectory_judge.judges.ollama_client import DEFAULT_HOST, generate
from trajectory_judge.trace import FailureType, Trajectory, Verdict

#: The procedure the agent was supposed to follow. A judge that has not been told the rules is
#: guessing at policy, so both judges get exactly this, word for word.
PROCEDURE = """\
The agent handles refund requests for an online retailer. It must:
1. Verify who the customer is before acting on their order.
2. Look up the order before judging it.
3. Read the refund policy for the item that was ordered.
4. Confirm eligibility before moving any money.
5. Refund exactly the amount the eligibility check authorised - no more, no less.
6. Escalate instead of refunding when the customer is not eligible.
7. Pass only arguments that come from the goal or from an earlier observation.
8. Finish by replying to the customer, asserting only what the observations support."""

TAXONOMY = """\
- wrong_tool: called a tool that does not serve the sub-goal it was pursuing, for example
  looking the order up a second time when it needed the item's refund policy.
- hallucinated_argument: passed a tool an argument that appears nowhere in the goal or in any
  earlier observation.
- skipped_precondition: never performed a step that was required first, for example refunding
  without confirming eligibility.
- ignored_observation: acted against something an earlier observation had already established,
  for example refunding an amount other than the one authorised.
- premature_stop: stopped before completing the task.
- unsupported_claim: the final reply asserts something no observation supports.
- none: the trajectory is correct."""

_ENUM = [f.value for f in FailureType] + ["none"]

#: ``reasoning`` comes first so the model argues before it commits. Schema order is generation
#: order, so this is chain-of-thought enforced by the grammar rather than requested politely.
STEP_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "faulty": {"type": "boolean"},
        "failure_step": {"type": "integer"},
        "failure_type": {"type": "string", "enum": _ENUM},
        "confidence": {"type": "number"},
    },
    "required": ["reasoning", "faulty", "failure_step", "failure_type", "confidence"],
}

#: No ``failure_step``: a judge that cannot see the steps has no business naming one.
OUTCOME_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "faulty": {"type": "boolean"},
        "failure_type": {"type": "string", "enum": _ENUM},
        "confidence": {"type": "number"},
    },
    "required": ["reasoning", "faulty", "failure_type", "confidence"],
}

_CONFIDENCE_INSTRUCTION = (
    "confidence is your probability that your own verdict is correct, from 0.5 (a coin flip) "
    "to 1.0 (certain). Do not default to a round number."
)


class LlmJudge(Judge):
    """Shared plumbing: build a prompt, generate under a schema, coerce to a verdict."""

    schema: dict[str, Any] = STEP_SCHEMA
    include_steps: bool = True

    def __init__(
        self,
        model: str = "qwen2.5:14b",
        *,
        judge_id: str | None = None,
        temperature: float = 0.0,
        seed: int = 7,
        host: str = DEFAULT_HOST,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.seed = seed
        self.host = host
        self.judge_id = judge_id or f"{self.family}:{model}"

    @property
    def family(self) -> str:
        raise NotImplementedError

    def prompt(self, trajectory: Trajectory) -> str:
        raise NotImplementedError

    def judge(self, trajectory: Trajectory, instance: Instance) -> Verdict:
        del instance  # An LLM judge sees the trajectory and nothing else.
        result = generate(
            self.model,
            self.prompt(trajectory),
            self.schema,
            temperature=self.temperature,
            seed=self.seed,
            host=self.host,
        )
        verdict = Verdict(
            trajectory_id=trajectory.trajectory_id,
            judge_id=self.judge_id,
            faulty=False,
            latency_s=result.latency_s,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            error=result.error,
        )
        parsed = result.parse()
        if parsed is None:
            # A judge that produced nothing usable votes "clean" at chance. Silently dropping
            # it would quietly improve whichever judge fails most often to answer.
            verdict.error = verdict.error or "unparseable response"
            verdict.confidence = 0.5
            return verdict

        verdict.faulty = bool(parsed.get("faulty", False))
        verdict.rationale = str(parsed.get("reasoning", ""))[:2000]
        verdict.confidence = _clamp(parsed.get("confidence"))

        raw_type = str(parsed.get("failure_type", "none"))
        if verdict.faulty and raw_type in {f.value for f in FailureType}:
            verdict.failure_type = FailureType(raw_type)

        if verdict.faulty and self.include_steps:
            step = parsed.get("failure_step")
            if isinstance(step, int) and 0 <= step < len(trajectory.steps):
                verdict.failure_step = step
        return verdict


def _clamp(value: object) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.5
    return min(1.0, max(0.5, number))


class OutcomeJudge(LlmJudge):
    """Sees the goal and the final answer. The production default, and the control condition."""

    schema = OUTCOME_SCHEMA
    include_steps = False

    @property
    def family(self) -> str:
        return "outcome"

    def prompt(self, trajectory: Trajectory) -> str:
        return f"""You are auditing a customer-support agent.

{PROCEDURE}

You are shown only the goal the agent was given and the answer it sent to the customer. The
steps it took are not available to you.

Decide whether this answer is a correct and complete resolution of the request. Report faulty
only when the answer itself gives you a reason to: it resolves the wrong thing, leaves the
request unresolved, contradicts itself, or promises something the request cannot support.

Not being able to see the steps is not evidence of a failure. Do not report faulty because a
procedure step is unverifiable from here - if the answer looks like a correct resolution, say
so.

Failure types:
{TAXONOMY}

{_CONFIDENCE_INSTRUCTION}

--- BEGIN ---
{trajectory.render(include_steps=False)}
--- END ---
"""


class StepRubricJudge(LlmJudge):
    """Sees every step. Asked to name where it went wrong, not only that it did."""

    schema = STEP_SCHEMA
    include_steps = True

    @property
    def family(self) -> str:
        return "step"

    def prompt(self, trajectory: Trajectory) -> str:
        return f"""You are auditing a customer-support agent.

{PROCEDURE}

You are shown the agent's full trajectory: every thought, tool call and observation, then the
answer it sent. Check each step against the procedure in order. A trajectory can be faulty even
when the final answer is correct - a right answer reached the wrong way is still a failure.

In your reasoning, go step by step and say for each whether it was appropriate and why. Then
give the index of the first step that went wrong, and the failure type from this list:
{TAXONOMY}

{_CONFIDENCE_INSTRUCTION}

--- BEGIN ---
{trajectory.render()}
--- END ---
"""
