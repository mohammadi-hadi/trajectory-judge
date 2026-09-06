"""What the app holds while it is running: one client, one semaphore, the judges.

The concurrency choice is the interesting one. Endpoints are ``async``, and the blocking judge
call is pushed to a worker thread through ``anyio.to_thread.run_sync``, behind an explicit
semaphore. A plain ``def`` endpoint would get the same offload for free, but the queue would
then be anyio's default thread limiter: invisible, unmeasurable, and with no way to answer 429.
Rewriting the judges async would perform better and would fork the exact call path that
produced every committed number, to save a thread on a unit of work that takes ten seconds.

One consequence of that design has to be handled explicitly: anyio's default thread limiter is
40 and it is global, so a ``max_concurrency`` above it would admit requests that then block
invisibly inside ``to_thread``. Startup raises the limiter to match.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

import anyio
import anyio.to_thread
import httpx

from trajectory_judge.judges.base import Judge
from trajectory_judge.judges.llm import OutcomeJudge, StepRubricJudge
from trajectory_judge.judges.mock import MockJudge
from trajectory_judge.judges.ollama_client import is_available
from trajectory_judge.judges.programmatic import ProgrammaticJudge
from trajectory_judge.judges.self_consistency import SelfConsistencyJudge
from trajectory_judge.serve.config import LOCAL_JUDGES, Settings
from trajectory_judge.serve.schemas import JudgeInfo, JudgeName

#: Headroom over max_concurrency for readiness probes and /v1/models, which also block.
THREAD_HEADROOM = 8

DESCRIPTIONS: dict[str, str] = {
    "programmatic": (
        "Rule engine over the refund procedure. Free and instant, blind to two failure types."
    ),
    "mock": (
        "Deterministic stand-in, seeded by trajectory id. For benchmarking the service only: "
        "its verdicts mean nothing about judging."
    ),
    "outcome": "Sees the goal and the final answer only. The production default.",
    "step": "Sees every step and names where it went wrong.",
    "selfcons": "The step judge sampled k times, majority vote. Costs k model calls.",
}


def judge_info(name: JudgeName, model: str, k: int = 3) -> JudgeInfo:
    """What ``/v1/judges`` reports, so a client can plan without reading the source."""
    needs_model = name not in LOCAL_JUDGES
    default_id = {
        "programmatic": "programmatic",
        "mock": "mock",
        "outcome": f"outcome:{model}",
        "step": f"step:{model}",
        "selfcons": f"selfcons{k}:{model}",
    }[name]
    return JudgeInfo(
        id=name,
        default_judge_id=default_id,
        requires_context=name in LOCAL_JUDGES,
        requires_model=needs_model,
        sees_steps=name != "outcome",
        upstream_calls=k if name == "selfcons" else (1 if needs_model else 0),
        for_benchmarking=name == "mock",
        description=DESCRIPTIONS[name],
    )


@dataclass
class Readiness:
    """A cached backend probe, so a probe storm cannot become a load source."""

    ttl_s: float
    _ok: bool = False
    _latency_ms: float | None = None
    _error: str | None = None
    _checked_at: float = field(default=-1e9)

    def get(self, host: str, client: httpx.Client) -> tuple[bool, float | None, str | None]:
        now = time.monotonic()
        if now - self._checked_at < self.ttl_s:
            return self._ok, self._latency_ms, self._error
        started = time.perf_counter()
        ok = is_available(host, client=client)
        self._latency_ms = (time.perf_counter() - started) * 1000
        self._ok = ok
        self._error = None if ok else "no response from /api/tags"
        self._checked_at = now
        return self._ok, self._latency_ms, self._error


class Runtime:
    """Everything the handlers need, built once in the lifespan."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        limit = settings.max_concurrency
        self.client = httpx.Client(
            limits=httpx.Limits(max_connections=limit + 4, max_keepalive_connections=limit),
            timeout=httpx.Timeout(
                connect=settings.connect_timeout_s,
                read=settings.upstream_timeout_s,
                write=5.0,
                pool=5.0,
            ),
        )
        self.semaphore = asyncio.Semaphore(limit)
        self.waiting = 0
        self.readiness = Readiness(ttl_s=settings.readiness_ttl_s)

    def raise_thread_limit(self) -> None:
        """Keep anyio's global thread limiter above our own, or the semaphore becomes a lie."""
        limiter = anyio.to_thread.current_default_thread_limiter()
        wanted = self.settings.max_concurrency + THREAD_HEADROOM
        if limiter.total_tokens < wanted:
            limiter.total_tokens = wanted

    def build(self, name: JudgeName, model: str, seed: int) -> Judge:
        """Construct a judge. Cheap: judges hold configuration, never a loaded model.

        ``host``, ``client`` and ``timeout_s`` are passed explicitly every time. The library
        binds ``DEFAULT_HOST`` at import and freezes it into constructor defaults, so relying
        on those defaults would ignore runtime configuration.
        """
        settings = self.settings
        common = {
            "host": settings.ollama_host,
            "client": self.client,
            "timeout_s": settings.upstream_timeout_s,
        }
        if name == "programmatic":
            return ProgrammaticJudge()
        if name == "mock":
            return MockJudge()
        if name == "outcome":
            return OutcomeJudge(model, seed=seed, **common)  # type: ignore[arg-type]
        if name == "step":
            return StepRubricJudge(model, seed=seed, **common)  # type: ignore[arg-type]
        return SelfConsistencyJudge(model, base_seed=seed, **common)  # type: ignore[arg-type]

    def close(self) -> None:
        self.client.close()
