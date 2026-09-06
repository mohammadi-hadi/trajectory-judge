"""Backpressure, batching, and what the service says when it is full."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient
from helpers import payload  # type: ignore[import-not-found]

from trajectory_judge.serve.app import create_app
from trajectory_judge.serve.config import Settings

pytestmark = [pytest.mark.serve, pytest.mark.slow]


def test_the_semaphore_actually_bounds_concurrent_model_calls(backend) -> None:  # type: ignore[no-untyped-def]
    limited = Settings(enabled_judges=frozenset({"step"}), max_concurrency=2, queue_timeout_s=10.0)
    backend.latency_s = 0.15
    with TestClient(create_app(limited)) as client, ThreadPoolExecutor(max_workers=8) as pool:
        replies = list(pool.map(lambda _: client.post("/v1/judge", json=payload()), range(8)))
    assert [r.status_code for r in replies] == [200] * 8
    assert backend.calls == 8
    assert backend.peak <= 2, f"ran {backend.peak} model calls at once with a limit of 2"


def test_a_full_queue_answers_429_rather_than_waiting_forever(backend) -> None:  # type: ignore[no-untyped-def]
    full = Settings(enabled_judges=frozenset({"step"}), max_concurrency=1, queue_timeout_s=0.05)
    backend.latency_s = 0.4
    with TestClient(create_app(full)) as client, ThreadPoolExecutor(max_workers=4) as pool:
        replies = list(pool.map(lambda _: client.post("/v1/judge", json=payload()), range(4)))
    codes = sorted(r.status_code for r in replies)
    assert 429 in codes, f"expected at least one rejection, got {codes}"
    rejected = next(r for r in replies if r.status_code == 429)
    assert rejected.json()["error"]["code"] == "overloaded"
    assert rejected.json()["error"]["retryable"] is True
    assert rejected.headers["Retry-After"] == "1"


def test_a_slow_tenth_does_not_poison_the_rest(backend) -> None:  # type: ignore[no-untyped-def]
    """The degraded case: failures arrive as failures and the successes stay fast."""
    settings = Settings(enabled_judges=frozenset({"step"}), max_concurrency=4, queue_timeout_s=10.0)
    with TestClient(create_app(settings)) as client:
        good = client.post("/v1/judge", json=payload())
        backend.error = "ReadTimeout: timed out"
        bad = client.post("/v1/judge", json=payload())
        backend.error = None
        after = client.post("/v1/judge", json=payload())
    assert (good.status_code, bad.status_code, after.status_code) == (200, 504, 200)
    assert after.json()["timing"]["overhead_s"] < 1.0


def test_a_batch_returns_200_with_per_item_statuses(client: TestClient, backend) -> None:  # type: ignore[no-untyped-def]
    reply = client.post(
        "/v1/judge/batch",
        json={"items": [payload(), payload("programmatic"), payload()], "max_concurrency": 2},
    )
    assert reply.status_code == 200
    body = reply.json()
    assert [r["status"] for r in body["results"]] == [200, 200, 200]
    assert body["summary"]["ok"] == 3
    assert body["summary"]["failed"] == 0
    assert [r["index"] for r in body["results"]] == [0, 1, 2]


def test_a_batch_reports_one_bad_item_without_failing_the_request(
    client: TestClient, backend
) -> None:  # type: ignore[no-untyped-def]
    bad = payload("programmatic")
    del bad["context"]
    reply = client.post("/v1/judge/batch", json={"items": [payload(), bad]})
    assert reply.status_code == 200
    body = reply.json()
    assert body["summary"] == {
        "ok": 1,
        "failed": 1,
        **{k: body["summary"][k] for k in ("total_s", "model_s_sum")},
    }
    failed = body["results"][1]
    assert failed["status"] == 422
    assert failed["error"]["code"] == "context_required"
    assert failed["verdict"] is None


def test_an_oversized_batch_is_refused(client: TestClient, backend) -> None:  # type: ignore[no-untyped-def]
    reply = client.post("/v1/judge/batch", json={"items": [payload()] * 9})
    assert reply.status_code == 413


def test_batch_concurrency_never_exceeds_the_service_limit(backend) -> None:  # type: ignore[no-untyped-def]
    settings = Settings(
        enabled_judges=frozenset({"step"}), max_concurrency=2, queue_timeout_s=10.0, max_batch=8
    )
    backend.latency_s = 0.1
    with TestClient(create_app(settings)) as client:
        reply = client.post(
            "/v1/judge/batch", json={"items": [payload()] * 6, "max_concurrency": 32}
        )
    assert reply.status_code == 200
    assert backend.peak <= 2


def test_the_client_is_closed_when_the_app_shuts_down(settings) -> None:  # type: ignore[no-untyped-def]
    app = create_app(settings)
    with TestClient(app) as client:
        client.get("/healthz")
        runtime = app.state.runtime
        assert runtime.client.is_closed is False
    assert runtime.client.is_closed is True


def test_shutdown_waits_for_work_in_flight(backend) -> None:  # type: ignore[no-untyped-def]
    settings = Settings(enabled_judges=frozenset({"step"}), max_concurrency=2, shutdown_grace_s=5.0)
    backend.latency_s = 0.3
    app = create_app(settings)
    with TestClient(app) as client, ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(client.post, "/v1/judge", json=payload()) for _ in range(2)]
        time.sleep(0.05)
        results = [f.result() for f in futures]
    assert [r.status_code for r in results] == [200, 200]
    assert app.state.runtime.client.is_closed is True
