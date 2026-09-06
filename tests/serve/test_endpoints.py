"""Endpoint behaviour, schema hygiene and the failure mapping."""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient
from helpers import payload  # type: ignore[import-not-found]

from trajectory_judge.judges.programmatic import ProgrammaticJudge
from trajectory_judge.serve.app import create_app
from trajectory_judge.serve.config import Settings
from trajectory_judge.serve.schemas import JudgeContext, TrajectoryIn

pytestmark = pytest.mark.serve


def test_healthz_is_up_without_checking_anything_downstream(client: TestClient) -> None:
    body = client.get("/healthz").json()
    assert body["status"] == "ok"
    assert body["git_sha"] == "unknown"


def test_readyz_skips_the_probe_when_no_judge_needs_a_model() -> None:
    local_only = Settings(enabled_judges=frozenset({"programmatic", "mock"}))
    with TestClient(create_app(local_only)) as c:
        reply = c.get("/readyz")
    assert reply.status_code == 200
    assert reply.json() == {"ready": True, "checks": {}}


def test_readyz_is_503_when_the_backend_is_unreachable() -> None:
    unreachable = Settings(
        enabled_judges=frozenset({"step"}), ollama_host="http://127.0.0.1:9", readiness_ttl_s=0.0
    )
    with TestClient(create_app(unreachable)) as c:
        reply = c.get("/readyz")
    assert reply.status_code == 503
    assert reply.json()["ready"] is False
    assert reply.json()["checks"]["ollama"]["ok"] is False


def test_the_judge_catalogue_declares_what_each_judge_needs(client: TestClient) -> None:
    by_id = {j["id"]: j for j in client.get("/v1/judges").json()}
    assert by_id["programmatic"]["requires_context"] is True
    assert by_id["programmatic"]["requires_model"] is False
    assert by_id["step"]["requires_context"] is False
    assert by_id["step"]["upstream_calls"] == 1
    assert by_id["selfcons"]["upstream_calls"] == 3
    assert by_id["mock"]["for_benchmarking"] is True
    assert by_id["outcome"]["sees_steps"] is False


def test_models_answers_200_even_when_the_backend_is_down() -> None:
    down = Settings(enabled_judges=frozenset({"step"}), ollama_host="http://127.0.0.1:9")
    with TestClient(create_app(down)) as c:
        reply = c.get("/v1/models")
    assert reply.status_code == 200
    assert reply.json()["models"] == []
    assert reply.json()["backend"]["available"] is False


def test_a_judged_request_reports_where_the_time_went(client: TestClient, backend) -> None:  # type: ignore[no-untyped-def]
    backend.latency_s = 0.05
    reply = client.post("/v1/judge", json=payload("step"))
    assert reply.status_code == 200
    body = reply.json()
    assert body["verdict"]["faulty"] is True
    assert body["verdict"]["judge_id"].startswith("step:")
    assert body["usage"] == {"prompt_tokens": 7, "completion_tokens": 13, "upstream_calls": 1}
    timing = body["timing"]
    assert timing["model_s"] == pytest.approx(0.05, abs=0.02)
    assert timing["overhead_s"] >= 0.0
    assert timing["overhead_s"] < timing["total_s"]
    assert reply.headers["X-Request-ID"] == body["request_id"]


def test_an_inbound_request_id_is_echoed(client: TestClient, backend) -> None:  # type: ignore[no-untyped-def]
    reply = client.post("/v1/judge", json=payload(), headers={"X-Request-ID": "abc-123"})
    assert reply.headers["X-Request-ID"] == "abc-123"


def test_a_hostile_request_id_is_replaced(client: TestClient, backend) -> None:  # type: ignore[no-untyped-def]
    reply = client.post("/v1/judge", json=payload(), headers={"X-Request-ID": "a b\nc"})
    assert reply.headers["X-Request-ID"] != "a b\nc"


def test_include_rationale_false_drops_the_text(client: TestClient, backend) -> None:  # type: ignore[no-untyped-def]
    reply = client.post("/v1/judge", json=payload(include_rationale=False))
    assert reply.json()["verdict"]["rationale"] == ""


def test_the_rule_judge_needs_context(client: TestClient) -> None:
    body = payload("programmatic")
    del body["context"]
    reply = client.post("/v1/judge", json=body)
    assert reply.status_code == 422
    assert reply.json()["error"]["code"] == "context_required"


def test_a_partial_order_is_a_422_not_a_500(client: TestClient) -> None:
    """`env.checker` indexes order["customer_id"] directly; the schema is what stops a 500."""
    body = payload("programmatic")
    body["context"]["order"] = {"status": "delivered"}
    reply = client.post("/v1/judge", json=body)
    assert reply.status_code == 422
    assert reply.json()["error"]["code"] == "invalid_request"


def test_the_endpoint_agrees_with_calling_the_judge_directly(client: TestClient) -> None:
    body = payload("programmatic")
    served = client.post("/v1/judge", json=body).json()["verdict"]

    trajectory = TrajectoryIn(**body["trajectory"]).to_trajectory()
    instance = JudgeContext(**body["context"]).to_instance(trajectory.instance_id)
    direct = ProgrammaticJudge().judge(trajectory, instance)

    assert served["faulty"] == direct.faulty
    assert served["failure_step"] == direct.failure_step
    assert served["failure_type"] == (direct.failure_type.value if direct.failure_type else None)
    assert served["confidence"] == pytest.approx(direct.confidence)


def test_ground_truth_cannot_be_posted(client: TestClient) -> None:
    body = payload()
    body["trajectory"]["label"] = {"faulty": True}
    reply = client.post("/v1/judge", json=body)
    assert reply.status_code == 422
    assert reply.json()["error"]["code"] == "invalid_request"


def test_the_schema_has_no_place_for_ground_truth(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()["components"]["schemas"]
    assert "label" not in schema["TrajectoryIn"]["properties"]
    assert "expected" not in schema["JudgeContext"]["properties"]


def test_too_many_steps_is_refused(client: TestClient) -> None:
    body = payload()
    body["trajectory"]["steps"] = [
        {
            "index": i,
            "thought": "t",
            "call": {"tool": "lookup_order", "args": {}},
            "observation": {"ok": True, "data": {}},
        }
        for i in range(11)
    ]
    reply = client.post("/v1/judge", json=body)
    assert reply.status_code == 413
    assert reply.json()["error"]["code"] == "payload_too_large"


def test_a_disabled_judge_is_refused() -> None:
    limited = Settings(enabled_judges=frozenset({"programmatic"}))
    with TestClient(create_app(limited)) as c:
        assert c.post("/v1/judge", json=payload("step")).status_code == 422


@pytest.mark.parametrize(
    ("error", "status", "code"),
    [
        ("ConnectError: [Errno 61] Connection refused", 503, "upstream_unavailable"),
        ("ConnectTimeout: timed out", 503, "upstream_unavailable"),
        ("ReadTimeout: timed out", 504, "upstream_timeout"),
        ("PoolTimeout: no connection", 504, "upstream_timeout"),
        ("HTTPStatusError: 404 model not found", 502, "upstream_error"),
    ],
)
def test_backend_failures_map_to_status_codes(
    client: TestClient, backend, error: str, status: int, code: str
) -> None:  # type: ignore[no-untyped-def]
    backend.error = error
    reply = client.post("/v1/judge", json=payload())
    assert reply.status_code == status
    body = reply.json()["error"]
    assert body["code"] == code
    assert body["judge"] == "step"
    assert body["request_id"]
    if status == 503:
        assert body["retryable"] is True
        assert reply.headers["Retry-After"] == "5"
    else:
        assert body["retryable"] is False


def test_unparseable_output_is_a_502_not_a_200(client: TestClient, backend) -> None:  # type: ignore[no-untyped-def]
    """A verdict the model never really gave must not be served as a success."""
    backend.text = "the agent did fine"
    reply = client.post("/v1/judge", json=payload())
    assert reply.status_code == 502
    assert reply.json()["error"]["code"] == "upstream_invalid_response"


def test_self_consistency_with_a_failed_member_is_an_error_not_a_partial_verdict(
    client: TestClient, backend
) -> None:  # type: ignore[no-untyped-def]
    backend.error = "ReadTimeout: timed out"
    reply = client.post("/v1/judge", json=payload("selfcons"))
    assert reply.status_code == 504


def test_settings_are_read_when_the_app_is_built_not_when_it_is_imported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The library freezes OLLAMA_HOST at import; the service must not repeat that."""
    monkeypatch.setenv("OLLAMA_HOST", "http://elsewhere:1234")
    monkeypatch.setenv("TJ_ENABLED_JUDGES", "step")
    app = create_app()
    with TestClient(app):
        runtime = app.state.runtime
        assert runtime.settings.ollama_host == "http://elsewhere:1234"
        assert runtime.build("step", "m", 7).host == "http://elsewhere:1234"
    assert os.environ["OLLAMA_HOST"] == "http://elsewhere:1234"
