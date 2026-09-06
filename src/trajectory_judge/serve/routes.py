"""The endpoints.

The shape of a judged request: acquire a slot (or 429), run the blocking judge in a worker
thread, then look at ``verdict.error`` and decide whether this was a 200. Timing is split three
ways on the way out — queue wait, model time, and what is left, which is this service's own
cost. That last number is the only one a reader should attribute to the server.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import anyio
import anyio.to_thread
from fastapi import APIRouter, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from trajectory_judge import __version__
from trajectory_judge.judges.base import Judge
from trajectory_judge.serve import errors
from trajectory_judge.serve.config import LOCAL_JUDGES
from trajectory_judge.serve.errors import ServiceError
from trajectory_judge.serve.logging import request_id_var
from trajectory_judge.serve.runtime import Runtime, judge_info
from trajectory_judge.serve.schemas import (
    BackendInfo,
    BatchItemResult,
    BatchRequest,
    BatchResponse,
    BatchSummary,
    Check,
    ErrorBody,
    HealthResponse,
    JudgeInfo,
    JudgeRequest,
    JudgeResponse,
    ModelsResponse,
    ReadyResponse,
    Timing,
    Usage,
    VerdictOut,
)
from trajectory_judge.trace import Verdict

log = logging.getLogger("trajectory_judge.serve")
router = APIRouter()

#: Batch items are judged through the same path, so they share the endpoint label.
ENDPOINT_JUDGE = "/v1/judge"


def _runtime(request: Request) -> Runtime:
    runtime: Runtime = request.app.state.runtime
    return runtime


@router.get("/healthz", response_model=HealthResponse, tags=["ops"])
async def healthz(request: Request) -> HealthResponse:
    """Liveness. Checks nothing downstream: the process being up is the whole claim."""
    settings = _runtime(request).settings
    return HealthResponse(status="ok", version=__version__, git_sha=settings.git_sha)


@router.get("/readyz", response_model=ReadyResponse, tags=["ops"])
async def readyz(request: Request, response: Response) -> ReadyResponse:
    """Readiness. Probes the backend only if a judge that needs one is enabled."""
    runtime = _runtime(request)
    if not runtime.settings.needs_model:
        return ReadyResponse(ready=True, checks={})
    ok, latency_ms, error = await anyio.to_thread.run_sync(
        lambda: runtime.readiness.get(runtime.settings.ollama_host, runtime.client)
    )
    if not ok:
        response.status_code = 503
    return ReadyResponse(
        ready=ok, checks={"ollama": Check(ok=ok, latency_ms=latency_ms, error=error)}
    )


@router.get("/v1/judges", response_model=list[JudgeInfo], tags=["catalogue"])
async def judges(request: Request) -> list[JudgeInfo]:
    settings = _runtime(request).settings
    return [
        judge_info(name, settings.default_model)
        for name in ("programmatic", "mock", "outcome", "step", "selfcons")
        if name in settings.enabled_judges
    ]


@router.get("/v1/models", response_model=ModelsResponse, tags=["catalogue"])
async def models(request: Request) -> ModelsResponse:
    """What the backend reports. 200 even when it is down: /readyz is the endpoint that fails."""
    runtime = _runtime(request)
    host = runtime.settings.ollama_host

    def fetch() -> tuple[list[str], str | None]:
        try:
            reply = runtime.client.get(f"{host}/api/tags", timeout=5.0)
            reply.raise_for_status()
            body: dict[str, Any] = reply.json()
        except Exception as exc:
            return [], f"{type(exc).__name__}: {exc}"
        names = [str(m.get("name", "")) for m in body.get("models", [])]
        return [n for n in names if n], None

    found, error = await anyio.to_thread.run_sync(fetch)
    return ModelsResponse(models=found, backend=BackendInfo(available=error is None, error=error))


@router.get("/metrics", include_in_schema=False)
async def metrics(request: Request) -> Response:
    registry = _runtime(request).app_metrics.registry  # type: ignore[attr-defined]
    return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)


def _check_request(runtime: Runtime, payload: JudgeRequest) -> None:
    """Everything that can be refused before a slot is taken."""
    settings = runtime.settings
    if payload.judge not in settings.enabled_judges:
        raise ServiceError(
            errors.Failure(
                422, errors.INVALID_REQUEST, f"the {payload.judge!r} judge is not enabled here"
            ),
            judge=payload.judge,
        )
    if len(payload.trajectory.steps) > settings.max_steps:
        raise ServiceError(errors.payload_too_large("steps", settings.max_steps), payload.judge)
    if payload.judge in LOCAL_JUDGES and payload.context is None:
        raise ServiceError(errors.context_required(payload.judge), payload.judge)


async def run_one(runtime: Runtime, payload: JudgeRequest, request_id: str) -> JudgeResponse:
    """Judge one trajectory, accounting for where the time went."""
    _check_request(runtime, payload)
    settings = runtime.settings
    model = payload.model or settings.default_model
    info = judge_info(payload.judge, model)
    metrics = runtime.app_metrics  # type: ignore[attr-defined]

    trajectory = payload.trajectory.to_trajectory()
    instance = (
        payload.context.to_instance(payload.trajectory.instance_id)
        if payload.context is not None
        else None
    )

    started = time.perf_counter()
    runtime.waiting += 1
    metrics.queue_depth.set(runtime.waiting)
    try:
        with anyio.fail_after(settings.queue_timeout_s):
            await runtime.semaphore.acquire()
    except TimeoutError:
        metrics.rejected.labels(reason="queue_full").inc()
        raise ServiceError(errors.overloaded(settings.queue_timeout_s), payload.judge) from None
    finally:
        runtime.waiting -= 1
        metrics.queue_depth.set(runtime.waiting)

    queue_wait = time.perf_counter() - started
    metrics.queue_wait.labels(judge=payload.judge).observe(queue_wait)
    metrics.inflight.labels(judge=payload.judge).inc()
    try:
        judge: Judge = runtime.build(payload.judge, model, payload.seed)
        deadline = settings.upstream_timeout_s * max(info.upstream_calls, 1) + 5.0
        try:
            with anyio.fail_after(deadline):
                verdict: Verdict = await anyio.to_thread.run_sync(
                    lambda: judge.judge(trajectory, instance)  # type: ignore[arg-type]
                )
        except TimeoutError:
            metrics.upstream_errors.labels(judge=payload.judge, model=model, kind="deadline").inc()
            raise ServiceError(
                errors.Failure(504, errors.UPSTREAM_TIMEOUT, f"no verdict within {deadline:g}s"),
                payload.judge,
            ) from None
    finally:
        runtime.semaphore.release()
        metrics.inflight.labels(judge=payload.judge).dec()

    total = time.perf_counter() - started

    if verdict.error:
        failure = errors.classify(verdict.error, settings.upstream_timeout_s)
        metrics.upstream_errors.labels(judge=payload.judge, model=model, kind=failure.code).inc()
        raise ServiceError(failure, payload.judge)

    overhead = max(0.0, total - verdict.latency_s - queue_wait)
    metrics.upstream.labels(judge=payload.judge, model=model).observe(verdict.latency_s)
    metrics.duration.labels(endpoint=ENDPOINT_JUDGE, judge=payload.judge).observe(total)
    # The headline number of the load test, exported by the service itself so the claim in
    # the README can be checked from outside with a scrape.
    metrics.overhead.labels(endpoint=ENDPOINT_JUDGE, judge=payload.judge).observe(overhead)
    metrics.requests.labels(
        endpoint=ENDPOINT_JUDGE, judge=payload.judge, status="200", outcome="ok"
    ).inc()
    token_counts = (("prompt", verdict.prompt_tokens), ("completion", verdict.completion_tokens))
    for kind, count in token_counts:
        if count:
            metrics.tokens.labels(judge=payload.judge, model=model, kind=kind).inc(count)

    return JudgeResponse(
        request_id=request_id,
        verdict=VerdictOut(
            trajectory_id=verdict.trajectory_id,
            judge_id=verdict.judge_id,
            faulty=verdict.faulty,
            failure_step=verdict.failure_step,
            failure_type=verdict.failure_type,
            confidence=verdict.confidence,
            rationale=verdict.rationale if payload.include_rationale else "",
        ),
        usage=Usage(
            prompt_tokens=verdict.prompt_tokens,
            completion_tokens=verdict.completion_tokens,
            upstream_calls=info.upstream_calls,
        ),
        timing=Timing(
            total_s=total,
            model_s=verdict.latency_s,
            queue_wait_s=queue_wait,
            overhead_s=overhead,
        ),
    )


@router.post("/v1/judge", response_model=JudgeResponse, tags=["judge"])
async def judge_one(request: Request, payload: JudgeRequest) -> JudgeResponse:
    return await run_one(_runtime(request), payload, request_id_var.get())


@router.post("/v1/judge/batch", response_model=BatchResponse, tags=["judge"])
async def judge_batch(request: Request, payload: BatchRequest) -> BatchResponse:
    """200 with per-item statuses, never 207.

    A batch where one item's backend timed out is not a failed HTTP request: the caller got
    exactly what it asked for. 207 is a WebDAV code that most retry middleware treats as an
    error, and a retry would re-run the items that already succeeded.
    """
    runtime = _runtime(request)
    request_id = request_id_var.get()
    if len(payload.items) > runtime.settings.max_batch:
        raise ServiceError(errors.payload_too_large("items", runtime.settings.max_batch))

    # Every item still passes the global semaphore, so a batch cannot starve single requests.
    gate = asyncio.Semaphore(min(payload.max_concurrency, runtime.settings.max_concurrency))
    started = time.perf_counter()

    async def one(index: int, item: JudgeRequest) -> BatchItemResult:
        async with gate:
            try:
                done = await run_one(runtime, item, request_id)
            except ServiceError as exc:
                return BatchItemResult(
                    index=index,
                    status=exc.failure.status,
                    error=ErrorBody(
                        code=exc.failure.code,
                        message=exc.failure.message,
                        request_id=request_id,
                        judge=exc.judge,
                        retryable=exc.failure.retryable,
                    ),
                )
            return BatchItemResult(
                index=index,
                status=200,
                verdict=done.verdict,
                usage=done.usage,
                timing=done.timing,
            )

    results = await asyncio.gather(*(one(i, item) for i, item in enumerate(payload.items)))
    ok = sum(1 for r in results if r.status == 200)
    return BatchResponse(
        request_id=request_id,
        results=list(results),
        summary=BatchSummary(
            ok=ok,
            failed=len(results) - ok,
            total_s=time.perf_counter() - started,
            model_s_sum=sum(r.timing.model_s for r in results if r.timing is not None),
        ),
    )
