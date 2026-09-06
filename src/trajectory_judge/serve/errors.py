"""Turning a judge's recorded failure into an HTTP status.

``Judge.judge`` never raises: when Ollama is unreachable it returns a perfectly valid
``Verdict`` with ``error`` set, ``faulty=False`` and ``confidence=0.5``. That is the right
choice for a benchmark, where a judge that produced nothing should still occupy a row. It is
the wrong thing to return as 200, because the service would then report full availability
while judging nothing at all. So the library keeps its behaviour and the service reinterprets
it here.

The mapping keys off the exception class name, because ``ollama_client.generate`` formats its
errors as ``f"{type(exc).__name__}: {exc}"``. That format is now a contract between two
modules and there is a test pinning it.
"""

from __future__ import annotations

from dataclasses import dataclass

CONTEXT_REQUIRED = "context_required"
INVALID_REQUEST = "invalid_request"
OVERLOADED = "overloaded"
PAYLOAD_TOO_LARGE = "payload_too_large"
UPSTREAM_ERROR = "upstream_error"
UPSTREAM_INVALID_RESPONSE = "upstream_invalid_response"
UPSTREAM_TIMEOUT = "upstream_timeout"
UPSTREAM_UNAVAILABLE = "upstream_unavailable"


@dataclass(frozen=True)
class Failure:
    """How one failure should be reported."""

    status: int
    code: str
    message: str
    retryable: bool = False
    retry_after: int | None = None


class ServiceError(Exception):
    """An error the handler already knows how to report."""

    def __init__(self, failure: Failure, judge: str | None = None, details: object = None) -> None:
        super().__init__(failure.message)
        self.failure = failure
        self.judge = judge
        self.details = details


#: Connect failures are the one retryable class. A read timeout is not retried: the model is
#: probably still working on the request, so a retry doubles load on the actual bottleneck.
_CONNECT = ("ConnectError", "ConnectTimeout")
_TIMEOUT = ("ReadTimeout", "WriteTimeout", "PoolTimeout", "TimeoutException")


def classify(error: str, upstream_timeout_s: float) -> Failure:
    """Map a ``Generation.error`` string onto a status code."""
    name = error.split(":", 1)[0].strip()

    if name in _CONNECT:
        return Failure(
            503,
            UPSTREAM_UNAVAILABLE,
            "the model backend is not reachable",
            retryable=True,
            retry_after=5,
        )
    if name in _TIMEOUT:
        return Failure(
            504, UPSTREAM_TIMEOUT, f"the model did not answer within {upstream_timeout_s:g}s"
        )
    if error == "unparseable response":
        return Failure(
            502, UPSTREAM_INVALID_RESPONSE, "the model returned something that is not a verdict"
        )
    return Failure(502, UPSTREAM_ERROR, f"the model backend failed: {error}")


def overloaded(queue_timeout_s: float) -> Failure:
    return Failure(
        429,
        OVERLOADED,
        f"no capacity within {queue_timeout_s:g}s; retry",
        retryable=True,
        retry_after=1,
    )


def context_required(judge: str) -> Failure:
    return Failure(
        422,
        CONTEXT_REQUIRED,
        f"the {judge!r} judge reads world state, so `context` is required for it",
    )


def payload_too_large(what: str, limit: int) -> Failure:
    return Failure(413, PAYLOAD_TOO_LARGE, f"{what} exceeds the limit of {limit}")
