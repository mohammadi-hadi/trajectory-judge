"""The application factory.

``create_app(settings)`` and not a module-level ``app``. A module-level application reads its
environment at import time, which is exactly the bug the library already has in
``ollama_client.DEFAULT_HOST``: set the variable after the import and nothing happens. There is
a test that constructs an app after changing the environment and asserts the judge picked it up.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import uuid4

import anyio
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from trajectory_judge import __version__
from trajectory_judge.serve import logging as jsonlog
from trajectory_judge.serve.config import Settings, from_env
from trajectory_judge.serve.errors import INVALID_REQUEST, ServiceError
from trajectory_judge.serve.metrics import Metrics
from trajectory_judge.serve.routes import router
from trajectory_judge.serve.runtime import Runtime
from trajectory_judge.serve.schemas import ErrorBody, ErrorResponse

log = logging.getLogger("trajectory_judge.serve")

#: An inbound request id is echoed only if it is safe to put in a log line.
_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def _error_response(status: int, body: ErrorBody, retry_after: int | None = None) -> JSONResponse:
    headers = {"X-Request-ID": body.request_id}
    if retry_after is not None:
        headers["Retry-After"] = str(retry_after)
    return JSONResponse(
        status_code=status, content=ErrorResponse(error=body).model_dump(), headers=headers
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved = settings or from_env()
    metrics = Metrics()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        runtime = Runtime(resolved)
        runtime.app_metrics = metrics  # type: ignore[attr-defined]
        runtime.raise_thread_limit()
        app.state.runtime = runtime
        log.info(
            "service starting",
            extra={
                "judges": sorted(resolved.enabled_judges),
                "max_concurrency": resolved.max_concurrency,
                "ollama_host": resolved.ollama_host,
                "version": __version__,
            },
        )
        try:
            yield
        finally:
            # Drain: wait for in-flight judging rather than cutting it off. A judge call can
            # be 30 seconds, and killing one wastes work that is nearly done.
            deadline = time.monotonic() + resolved.shutdown_grace_s
            while runtime.semaphore._value < resolved.max_concurrency:
                if time.monotonic() > deadline:
                    outstanding = resolved.max_concurrency - runtime.semaphore._value
                    log.warning("shutdown grace expired", extra={"abandoned": outstanding})
                    break
                await anyio.sleep(0.05)
            runtime.close()
            log.info("service stopped")

    app = FastAPI(
        title="trajectory-judge",
        version=__version__,
        summary="Judge agent trajectories over HTTP.",
        lifespan=lifespan,
    )
    app.include_router(router)

    @app.middleware("http")
    async def context(request: Request, call_next):  # type: ignore[no-untyped-def]
        inbound = request.headers.get("X-Request-ID", "")
        request_id = inbound if _ID.match(inbound) else uuid4().hex
        token = jsonlog.request_id_var.set(request_id)
        started = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            jsonlog.request_id_var.reset(token)
        response.headers["X-Request-ID"] = request_id

        route = request.scope.get("route")
        endpoint = getattr(route, "path", request.url.path)
        if endpoint != "/metrics":
            elapsed = time.perf_counter() - started
            metrics.duration.labels(endpoint=endpoint, judge="-").observe(elapsed)
            log.info(
                "request",
                extra={
                    "method": request.method,
                    "path": endpoint,
                    "status": response.status_code,
                    "duration_ms": round(elapsed * 1000, 3),
                },
            )
        return response

    @app.exception_handler(ServiceError)
    async def on_service_error(request: Request, exc: ServiceError) -> JSONResponse:
        failure = exc.failure
        request_id = jsonlog.request_id_var.get()
        log.warning(
            "request failed",
            extra={"code": failure.code, "status": failure.status, "judge": exc.judge},
        )
        metrics.requests.labels(
            endpoint=request.url.path,
            judge=exc.judge or "-",
            status=str(failure.status),
            outcome=failure.code,
        ).inc()
        return _error_response(
            failure.status,
            ErrorBody(
                code=failure.code,
                message=failure.message,
                request_id=request_id,
                judge=exc.judge,
                retryable=failure.retryable,
                details=exc.details,
            ),
            failure.retry_after,
        )

    @app.exception_handler(RequestValidationError)
    async def on_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        """Same envelope as every other error, so a client parses one shape."""
        return _error_response(
            422,
            ErrorBody(
                code=INVALID_REQUEST,
                message="the request does not match the schema",
                request_id=jsonlog.request_id_var.get(),
                details=exc.errors(),
            ),
        )

    return app


def run() -> None:  # pragma: no cover - exercised by the container smoke test
    """Entry point used by the CLI. One worker: scale with replicas, not with processes."""
    import uvicorn

    settings = from_env()
    jsonlog.configure(settings.log_level)
    uvicorn.run(
        create_app(settings),
        host="0.0.0.0",
        port=8000,
        workers=1,
        log_config=None,
        timeout_graceful_shutdown=int(settings.shutdown_grace_s),
    )
