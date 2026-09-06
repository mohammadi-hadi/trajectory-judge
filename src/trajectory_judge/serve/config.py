"""Runtime settings, read from the environment once and then passed explicitly.

Everything is read in ``from_env`` rather than at import time. The library already has one
import-time binding of ``OLLAMA_HOST`` and it is a nuisance: changing the variable after the
module loads has no effect. The service does not repeat that.
"""

from __future__ import annotations

import os

from pydantic import BaseModel, Field

from trajectory_judge.judges.ollama_client import DEFAULT_HOST

#: Judges that need no model, and so keep working when Ollama is gone.
LOCAL_JUDGES = frozenset({"programmatic", "mock"})

DEFAULT_ENABLED = "programmatic,mock,outcome,step,selfcons"


class Settings(BaseModel):
    """Service configuration. Field names match their ``TJ_``-prefixed variables."""

    model_config = {"frozen": True}

    ollama_host: str = DEFAULT_HOST
    default_model: str = "qwen2.5:14b"
    enabled_judges: frozenset[str] = frozenset(DEFAULT_ENABLED.split(","))

    max_concurrency: int = Field(default=4, ge=1, le=64)
    queue_timeout_s: float = Field(default=5.0, ge=0.0)
    upstream_timeout_s: float = Field(default=60.0, gt=0.0)
    connect_timeout_s: float = Field(default=2.0, gt=0.0)
    shutdown_grace_s: float = Field(default=30.0, ge=0.0)
    readiness_ttl_s: float = Field(default=10.0, ge=0.0)

    max_steps: int = Field(default=100, ge=1)
    max_batch: int = Field(default=64, ge=1)

    log_level: str = "info"
    git_sha: str = "unknown"

    @property
    def needs_model(self) -> bool:
        """Whether any enabled judge calls a model, so whether readiness has anything to probe."""
        return bool(self.enabled_judges - LOCAL_JUDGES)


def _float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return default if raw is None else float(raw)


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return default if raw is None else int(raw)


def from_env() -> Settings:
    """Build settings from ``TJ_*``, with ``OLLAMA_HOST`` still authoritative for the backend."""
    return Settings(
        ollama_host=os.environ.get("OLLAMA_HOST", DEFAULT_HOST),
        default_model=os.environ.get("TJ_DEFAULT_MODEL", "qwen2.5:14b"),
        enabled_judges=frozenset(
            part.strip()
            for part in os.environ.get("TJ_ENABLED_JUDGES", DEFAULT_ENABLED).split(",")
            if part.strip()
        ),
        max_concurrency=_int("TJ_MAX_CONCURRENCY", 4),
        queue_timeout_s=_float("TJ_QUEUE_TIMEOUT_S", 5.0),
        upstream_timeout_s=_float("TJ_UPSTREAM_TIMEOUT_S", 60.0),
        connect_timeout_s=_float("TJ_CONNECT_TIMEOUT_S", 2.0),
        shutdown_grace_s=_float("TJ_SHUTDOWN_GRACE_S", 30.0),
        readiness_ttl_s=_float("TJ_READINESS_TTL_S", 10.0),
        max_steps=_int("TJ_MAX_STEPS", 100),
        max_batch=_int("TJ_MAX_BATCH", 64),
        log_level=os.environ.get("TJ_LOG_LEVEL", "info"),
        git_sha=os.environ.get("TJ_GIT_SHA", "unknown"),
    )
