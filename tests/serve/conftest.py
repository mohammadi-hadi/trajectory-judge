"""Fixtures for the service tests.

``importorskip`` at module scope skips this whole directory when the serve extra is absent,
which is what keeps the CI `package` job green: it installs the bare wheel and runs the suite.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from typing import Any

import pytest

pytest.importorskip("fastapi", reason="needs the serve extra")

from fastapi.testclient import TestClient

from trajectory_judge.judges.ollama_client import Generation
from trajectory_judge.serve.app import create_app
from trajectory_judge.serve.config import Settings


class FakeBackend:
    """A stand-in for the model call, with a lock-protected concurrency high-water mark.

    The lock is not decoration. The double runs in worker threads, and an unlocked
    ``current += 1; peak = max(peak, current)`` loses updates, so the concurrency test would
    pass even with a broken semaphore.
    """

    def __init__(self) -> None:
        self.latency_s = 0.0
        self.error: str | None = None
        self.text = json.dumps(
            {
                "reasoning": "step 0 skipped the eligibility check",
                "faulty": True,
                "failure_step": 0,
                "failure_type": "skipped_precondition",
                "confidence": 0.8,
            }
        )
        self.calls = 0
        self.current = 0
        self.peak = 0
        self._lock = threading.Lock()

    def __call__(
        self, model: str, prompt: str, schema: dict[str, Any], **kwargs: Any
    ) -> Generation:
        with self._lock:
            self.calls += 1
            self.current += 1
            self.peak = max(self.peak, self.current)
        try:
            if self.latency_s:
                threading.Event().wait(self.latency_s)
            if self.error is not None:
                return Generation("", 0, 0, self.latency_s, error=self.error)
            return Generation(self.text, 7, 13, self.latency_s)
        finally:
            with self._lock:
                self.current -= 1


@pytest.fixture
def backend(monkeypatch: pytest.MonkeyPatch) -> FakeBackend:
    fake = FakeBackend()
    monkeypatch.setattr("trajectory_judge.judges.llm.generate", fake)
    return fake


@pytest.fixture
def settings() -> Settings:
    return Settings(
        enabled_judges=frozenset({"programmatic", "mock", "outcome", "step", "selfcons"}),
        max_concurrency=4,
        queue_timeout_s=2.0,
        upstream_timeout_s=5.0,
        max_steps=10,
        max_batch=8,
    )


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    with TestClient(create_app(settings)) as test_client:
        yield test_client
