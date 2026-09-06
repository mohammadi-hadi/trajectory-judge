"""Logs are JSON and carry no customer text; metrics count what the README claims."""

from __future__ import annotations

import json
import logging

import pytest
from fastapi.testclient import TestClient
from helpers import payload  # type: ignore[import-not-found]

from trajectory_judge.serve.logging import JsonFormatter, request_id_var

pytestmark = pytest.mark.serve

SECRET_GOAL = "refund order ORD-SECRET for Wilhelmina Bakker"
SECRET_ANSWER = "I have refunded 41.55 EUR to your account ending 9931"


def _emit(record_args: dict[str, object]) -> dict[str, object]:
    record = logging.LogRecord(
        name="t", level=logging.INFO, pathname="p", lineno=1, msg="m", args=(), exc_info=None
    )
    for key, value in record_args.items():
        setattr(record, key, value)
    return json.loads(JsonFormatter().format(record))


def test_every_record_is_one_json_object_with_the_required_keys() -> None:
    token = request_id_var.set("rid-1")
    try:
        body = _emit({"status": 200})
    finally:
        request_id_var.reset(token)
    assert body["level"] == "info"
    assert body["msg"] == "m"
    assert body["request_id"] == "rid-1"
    assert body["status"] == 200
    assert body["ts"].endswith("Z")


def test_customer_text_never_reaches_the_log(
    client: TestClient, backend, caplog: pytest.LogCaptureFixture
) -> None:  # type: ignore[no-untyped-def]
    body = payload()
    body["trajectory"]["goal"] = SECRET_GOAL
    body["trajectory"]["final_answer"] = SECRET_ANSWER
    with caplog.at_level(logging.DEBUG):
        assert client.post("/v1/judge", json=body).status_code == 200

    written = "\n".join(JsonFormatter().format(record) for record in caplog.records)
    assert SECRET_GOAL not in written
    assert SECRET_ANSWER not in written
    assert "Wilhelmina" not in written
    assert "9931" not in written
    # the rationale is the model's text about that content, and is equally out
    assert "skipped the eligibility check" not in written


def test_metrics_count_the_request_and_resolve_a_slow_p99(client: TestClient, backend) -> None:  # type: ignore[no-untyped-def]
    client.post("/v1/judge", json=payload())
    body = client.get("/metrics").text

    assert "tj_service_overhead_seconds_count" in body
    assert 'tj_upstream_duration_seconds_bucket{judge="step"' in body
    assert 'tj_tokens_total{judge="step"' in body
    # The default Prometheus histogram stops at 10s, which would put every step judge
    # (10.4s mean) and self-consistency run (30.2s) into +Inf.
    assert 'le="120.0"' in body


def test_a_judged_request_is_counted_once_per_duration_metric(client: TestClient, backend) -> None:  # type: ignore[no-untyped-def]
    """The edge histogram and the per-judge histogram must not both cover /v1/judge.

    They used to share ``tj_request_duration_seconds``, the middleware writing ``judge="-"`` and
    the handler writing the real name, so summing the rate across the judge label counted every
    request twice.
    """
    client.post("/v1/judge", json=payload())
    lines = client.get("/metrics").text.splitlines()

    def total(metric: str) -> float:
        return sum(
            float(line.rsplit(" ", 1)[1]) for line in lines if line.startswith(f"{metric}_count{{")
        )

    assert total("tj_request_duration_seconds") == 1.0
    assert total("tj_http_request_duration_seconds") == 1.0
    assert 'judge="-"' not in "\n".join(lines)


def test_a_rejected_request_is_counted_as_rejected(client: TestClient, backend) -> None:  # type: ignore[no-untyped-def]
    backend.error = "ConnectError: refused"
    client.post("/v1/judge", json=payload())
    body = client.get("/metrics").text
    assert 'tj_upstream_errors_total{judge="step"' in body
    assert 'kind="upstream_unavailable"' in body


def test_metrics_are_not_measured_by_their_own_middleware(client: TestClient, backend) -> None:  # type: ignore[no-untyped-def]
    client.get("/metrics")
    body = client.get("/metrics").text
    assert 'endpoint="/metrics"' not in body
