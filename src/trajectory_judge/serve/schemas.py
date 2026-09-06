"""The wire models.

Two omissions are the point of this module, and both are enforced by the schema rather than by
discipline:

``TrajectoryIn`` has no ``label``
    ``Trajectory.label`` is ground truth. Nothing on the judging path reads it today, but an
    eval API that can physically be told the answer is one refactor away from leaking it.

``JudgeContext`` has no ``expected``
    Same argument for the world state. The rule engine reads ``given``, ``order.customer_id``
    and ``order.status`` and nothing else, so those are all the service accepts.

``extra="forbid"`` turns a posted ``label`` into a 422 rather than a silent drop.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from trajectory_judge.env.world import Instance, Outcome
from trajectory_judge.trace import FailureType, Step, Trajectory

JudgeName = Literal["programmatic", "mock", "outcome", "step", "selfcons"]

Scalar = str | int | float | bool


class OrderFacts(BaseModel):
    """The order fields the rule engine reads.

    Typed and required on purpose. ``env.checker`` indexes ``instance.order["customer_id"]``
    and ``["status"]`` directly, so a loose dict here would turn a missing key into a KeyError
    inside a judge whose contract says it must never raise, and that into a 500.
    """

    model_config = ConfigDict(extra="forbid")

    customer_id: str = Field(min_length=1, max_length=200)
    status: str = Field(default="delivered", max_length=200)


class JudgeContext(BaseModel):
    """World facts a rule judge may read. There is deliberately no ``expected`` field."""

    model_config = ConfigDict(extra="forbid")

    given: dict[str, Any] = Field(default_factory=dict)
    order: OrderFacts

    def to_instance(self, instance_id: str) -> Instance:
        """Build the Instance the rule engine wants.

        ``expected`` is a sentinel: ``check()`` never reads it, and the wire model has no way
        to set it. Everything else the rule path does not touch is left empty.
        """
        return Instance(
            instance_id=instance_id,
            difficulty="unspecified",
            goal="",
            given=self.given,
            customer={},
            order=self.order.model_dump(),
            policy={},
            expected=Outcome(action="none"),
        )


class TrajectoryIn(BaseModel):
    """A trajectory to judge. Carries no ground-truth label, by construction."""

    model_config = ConfigDict(extra="forbid")

    trajectory_id: str = Field(default_factory=lambda: uuid4().hex, max_length=200)
    instance_id: str = Field(default="unspecified", max_length=200)
    goal: str = Field(min_length=1, max_length=8_000)
    steps: list[Step] = Field(default_factory=list)
    final_answer: str = Field(default="", max_length=16_000)

    def to_trajectory(self) -> Trajectory:
        return Trajectory(
            trajectory_id=self.trajectory_id,
            instance_id=self.instance_id,
            goal=self.goal,
            steps=self.steps,
            final_answer=self.final_answer,
        )


class JudgeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trajectory: TrajectoryIn
    judge: JudgeName
    model: str | None = Field(default=None, max_length=200)
    context: JudgeContext | None = None
    seed: int = 7
    include_rationale: bool = True


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[JudgeRequest] = Field(min_length=1)
    max_concurrency: Annotated[int, Field(ge=1, le=32)] = 4


class VerdictOut(BaseModel):
    trajectory_id: str
    judge_id: str
    faulty: bool
    failure_step: int | None = None
    failure_type: FailureType | None = None
    confidence: float
    rationale: str = ""


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    upstream_calls: int


class Timing(BaseModel):
    """Where the wall clock went.

    ``model_s`` is what the provider spent and ``overhead_s`` is what this service added. They
    are reported separately because a p99 that is 99% model time says nothing about the server,
    and a reader with curl should be able to check that claim rather than take it on trust.
    """

    total_s: float
    model_s: float
    queue_wait_s: float
    overhead_s: float


class JudgeResponse(BaseModel):
    request_id: str
    verdict: VerdictOut
    usage: Usage
    timing: Timing


class BatchItemResult(BaseModel):
    index: int
    status: int
    verdict: VerdictOut | None = None
    usage: Usage | None = None
    timing: Timing | None = None
    error: ErrorBody | None = None


class BatchSummary(BaseModel):
    ok: int
    failed: int
    total_s: float
    model_s_sum: float


class BatchResponse(BaseModel):
    request_id: str
    results: list[BatchItemResult]
    summary: BatchSummary


class ErrorBody(BaseModel):
    code: str
    message: str
    request_id: str
    judge: str | None = None
    retryable: bool = False
    details: Any = None


class ErrorResponse(BaseModel):
    error: ErrorBody


class JudgeInfo(BaseModel):
    """What a client needs to call a judge without reading the source."""

    id: JudgeName
    default_judge_id: str
    requires_context: bool
    requires_model: bool
    sees_steps: bool
    upstream_calls: int
    for_benchmarking: bool
    description: str


class HealthResponse(BaseModel):
    status: str
    version: str
    git_sha: str


class Check(BaseModel):
    ok: bool
    latency_ms: float | None = None
    error: str | None = None


class ReadyResponse(BaseModel):
    ready: bool
    checks: dict[str, Check]


class BackendInfo(BaseModel):
    available: bool
    error: str | None = None


class ModelsResponse(BaseModel):
    models: list[str]
    backend: BackendInfo


BatchItemResult.model_rebuild()
