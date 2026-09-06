"""JSON logs, and the request id that ties them together.

Two decisions worth stating.

The formatter is forty lines of stdlib rather than structlog, because a package with three
runtime dependencies should not gain a logging framework to emit one line per request.

The ``dictConfig`` replaces uvicorn's handlers as well as the root's. Uvicorn installs its own
``uvicorn``, ``uvicorn.error`` and ``uvicorn.access`` loggers, and without this you ship a
service whose output is half JSON and half plain text, which is worse than either.

What is never logged: the goal, the step text, the final answer, the rationale. In any real
deployment that is the customer's data. ``prompt_chars`` goes in instead.
"""

from __future__ import annotations

import contextvars
import json
import logging
import logging.config
from typing import Any

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")

#: Anything set on a record outside this set is dropped, so a stray ``extra`` cannot leak text.
_RESERVED = frozenset(
    {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "message",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "thread",
        "threadName",
        "taskName",
    }
)


class JsonFormatter(logging.Formatter):
    """One JSON object per line, with the request id pulled from the context."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S") + f".{int(record.msecs):03d}Z",
            "level": record.levelname.lower(),
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": request_id_var.get(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure(level: str = "info") -> None:
    """Install the JSON formatter everywhere, uvicorn's own loggers included."""
    handler = {"class": "logging.StreamHandler", "formatter": "json", "stream": "ext://sys.stdout"}
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {"json": {"()": f"{__name__}.JsonFormatter"}},
            "handlers": {"default": handler},
            "root": {"handlers": ["default"], "level": level.upper()},
            "loggers": {
                name: {"handlers": ["default"], "level": level.upper(), "propagate": False}
                # uvicorn.access is silenced: this service logs one richer access record of its
                # own, and two lines per request that disagree on fields is a debugging tax.
                for name in ("uvicorn", "uvicorn.error", "uvicorn.access")
            },
        }
    )
    logging.getLogger("uvicorn.access").disabled = True
