"""Prometheus instruments.

The bucket boundaries are the part that matters. Prometheus' default histogram tops out at
10 seconds; the step-rubric judge averages 10.4s and self-consistency 30.2s, so every
interesting request would land in ``+Inf`` and p99 would be unrecoverable. The service-overhead
histogram has the opposite problem and is sized for the sub-millisecond range it measures.

``tj_service_overhead_seconds`` carries the same labels as ``tj_request_duration_seconds`` so a
dashboard can put them side by side. It is the number the load test reports, exported by the
service itself, which is what makes that number checkable from outside.
"""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

REQUEST_BUCKETS = (
    0.001,
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    20.0,
    40.0,
    60.0,
    120.0,
)
OVERHEAD_BUCKETS = (0.0005, 0.001, 0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0)


class Metrics:
    """One registry per app, so tests can build an app without touching global state."""

    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "tj_requests_total",
            "Requests by endpoint, judge, status and outcome.",
            ["endpoint", "judge", "status", "outcome"],
            registry=self.registry,
        )
        self.duration = Histogram(
            "tj_request_duration_seconds",
            "Wall-clock time per request.",
            ["endpoint", "judge"],
            buckets=REQUEST_BUCKETS,
            registry=self.registry,
        )
        self.overhead = Histogram(
            "tj_service_overhead_seconds",
            "Request time minus model time and queue wait.",
            ["endpoint", "judge"],
            buckets=OVERHEAD_BUCKETS,
            registry=self.registry,
        )
        self.upstream = Histogram(
            "tj_upstream_duration_seconds",
            "Time inside the model backend.",
            ["judge", "model"],
            buckets=REQUEST_BUCKETS,
            registry=self.registry,
        )
        self.queue_wait = Histogram(
            "tj_queue_wait_seconds",
            "Time spent waiting for a concurrency slot.",
            ["judge"],
            buckets=OVERHEAD_BUCKETS,
            registry=self.registry,
        )
        self.inflight = Gauge(
            "tj_inflight", "Requests currently judging.", ["judge"], registry=self.registry
        )
        self.queue_depth = Gauge(
            "tj_queue_depth", "Requests waiting for a slot.", registry=self.registry
        )
        self.upstream_errors = Counter(
            "tj_upstream_errors_total",
            "Model backend failures by kind.",
            ["judge", "model", "kind"],
            registry=self.registry,
        )
        self.rejected = Counter(
            "tj_rejected_total",
            "Requests refused before judging.",
            ["reason"],
            registry=self.registry,
        )
        self.tokens = Counter(
            "tj_tokens_total",
            "Tokens in and out.",
            ["judge", "model", "kind"],
            registry=self.registry,
        )
