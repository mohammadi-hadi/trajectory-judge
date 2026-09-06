"""A small asyncio load driver.

Tool choice: httpx is already a runtime dependency of this project, so a driver written on it
adds nothing to install and runs anywhere the service runs. k6 would mean a Go binary in a
three-dependency Python repo; locust drags in gevent, whose monkeypatching fights anyio.

Closed loop by default: N workers each send one request and wait. The quantity reported is
"latency at concurrency N" and throughput is derived from it, never targeted, so there is no
coordinated omission to correct for. `--arrival-rate` switches to an open loop with Poisson
arrivals, which is only honest against a backend fast enough to keep up.

Percentiles are nearest-rank on the sorted sample, and raw samples are kept when there are few
enough, so every number in the output can be recomputed by hand.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import httpx

SCHEMA_VERSION = 1


def percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile, so it is reproducible by hand from the raw samples."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    rank = max(1, min(len(ordered), int(-(-q * len(ordered) // 1))))
    return ordered[rank - 1]


@dataclass
class Sample:
    status: int
    total_s: float
    model_s: float = 0.0
    queue_wait_s: float = 0.0
    overhead_s: float = 0.0


@dataclass
class ScenarioResult:
    name: str
    judge: str
    upstream: str
    concurrency: int
    arrival: str
    n: int
    p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float
    rps: float
    error_rate: float
    status_counts: dict[str, int]
    overhead_p50_ms: float
    overhead_p99_ms: float
    model_ms_p50: float
    queue_p99_ms: float
    scraped_overhead_p99_ms: float | None = None
    samples_ms: list[float] = field(default_factory=list)


def body(judge: str, steps: int = 6) -> dict[str, Any]:
    trajectory: dict[str, Any] = {
        "trajectory_id": "bench",
        "goal": "refund order ORD-1 for the customer",
        "steps": [
            {
                "index": i,
                "thought": f"considering step {i}",
                "call": {"tool": "lookup_order", "args": {"order_id": "ORD-1"}},
                "observation": {"ok": True, "data": {"order_id": "ORD-1", "total_eur": 41.55}},
            }
            for i in range(steps)
        ],
        "final_answer": "I have refunded the order.",
    }
    payload: dict[str, Any] = {"trajectory": trajectory, "judge": judge}
    if judge in {"programmatic", "mock"}:
        payload["context"] = {
            "given": {"order_id": "ORD-1"},
            "order": {"customer_id": "C-1", "status": "delivered"},
        }
    return payload


async def _one(client: httpx.AsyncClient, url: str, payload: dict[str, Any]) -> Sample:
    started = time.perf_counter()
    try:
        reply = await client.post(url, json=payload)
    except httpx.HTTPError:
        return Sample(status=0, total_s=time.perf_counter() - started)
    elapsed = time.perf_counter() - started
    if reply.status_code != 200:
        return Sample(status=reply.status_code, total_s=elapsed)
    timing = reply.json()["timing"]
    return Sample(
        status=200,
        total_s=elapsed,
        model_s=timing["model_s"],
        queue_wait_s=timing["queue_wait_s"],
        overhead_s=timing["overhead_s"],
    )


async def closed_loop(
    base: str, judge: str, concurrency: int, requests: int, timeout_s: float
) -> list[Sample]:
    payload = body(judge)
    url = f"{base}/v1/judge"
    counter = iter(range(requests))
    samples: list[Sample] = []
    limits = httpx.Limits(max_connections=concurrency + 4, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(limits=limits, timeout=timeout_s) as client:

        async def worker() -> None:
            for _ in counter:
                samples.append(await _one(client, url, payload))

        await asyncio.gather(*(worker() for _ in range(concurrency)))
    return samples


async def open_loop(
    base: str, judge: str, rate: float, seconds: float, timeout_s: float
) -> list[Sample]:
    payload = body(judge)
    url = f"{base}/v1/judge"
    rng = random.Random(7)
    samples: list[Sample] = []
    tasks: list[asyncio.Task[Sample]] = []
    async with httpx.AsyncClient(timeout=timeout_s) as client:
        deadline = time.perf_counter() + seconds
        while time.perf_counter() < deadline:
            tasks.append(asyncio.create_task(_one(client, url, payload)))
            await asyncio.sleep(rng.expovariate(rate))
        samples = list(await asyncio.gather(*tasks))
    return samples


async def scrape_overhead_p99(base: str) -> float | None:
    """Read the service's own overhead histogram, as an independent estimate."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            text = (await client.get(f"{base}/metrics")).text
    except httpx.HTTPError:
        return None
    buckets: list[tuple[float, float]] = []
    total = 0.0
    for line in text.splitlines():
        if not line.startswith("tj_service_overhead_seconds_bucket"):
            continue
        le = line.split('le="')[1].split('"')[0]
        count = float(line.rsplit(" ", 1)[1])
        buckets.append((float(le), count))
        total = max(total, count)
    if not buckets or total == 0:
        return None
    target = 0.99 * total
    for le, count in sorted(buckets):
        if count >= target:
            return le * 1000
    return None


def summarise(
    name: str,
    judge: str,
    upstream: str,
    concurrency: int,
    arrival: str,
    samples: list[Sample],
    wall_s: float,
    scraped: float | None,
) -> ScenarioResult:
    ok = [s for s in samples if s.status == 200]
    totals = [s.total_s * 1000 for s in samples]
    counts: dict[str, int] = {}
    for s in samples:
        key = str(s.status)
        counts[key] = counts.get(key, 0) + 1
    return ScenarioResult(
        name=name,
        judge=judge,
        upstream=upstream,
        concurrency=concurrency,
        arrival=arrival,
        n=len(samples),
        p50_ms=round(percentile(totals, 0.50), 3),
        p90_ms=round(percentile(totals, 0.90), 3),
        p95_ms=round(percentile(totals, 0.95), 3),
        p99_ms=round(percentile(totals, 0.99), 3),
        max_ms=round(max(totals), 3) if totals else float("nan"),
        rps=round(len(samples) / wall_s, 2) if wall_s else 0.0,
        error_rate=round(1 - len(ok) / len(samples), 4) if samples else 0.0,
        status_counts=counts,
        overhead_p50_ms=round(percentile([s.overhead_s * 1000 for s in ok], 0.50), 4),
        overhead_p99_ms=round(percentile([s.overhead_s * 1000 for s in ok], 0.99), 4),
        model_ms_p50=round(percentile([s.model_s * 1000 for s in ok], 0.50), 3),
        queue_p99_ms=round(percentile([s.queue_wait_s * 1000 for s in ok], 0.99), 4),
        scraped_overhead_p99_ms=None if scraped is None else round(scraped, 3),
        samples_ms=[round(t, 3) for t in totals] if len(totals) <= 2000 else [],
    )


def environment(label: str) -> dict[str, Any]:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        sha = "unknown"
    return {
        "schema_version": SCHEMA_VERSION,
        "git_sha": sha,
        "label": label,
        "python": platform.python_version(),
        "machine": platform.machine(),
        "system": platform.system(),
        "cpu_count": os.cpu_count(),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


async def run_scenario(args: argparse.Namespace) -> ScenarioResult:
    started = time.perf_counter()
    if args.arrival_rate:
        samples = await open_loop(
            args.base, args.judge, args.arrival_rate, args.seconds, args.timeout
        )
        arrival = f"poisson@{args.arrival_rate}/s"
    else:
        samples = await closed_loop(
            args.base, args.judge, args.concurrency, args.requests, args.timeout
        )
        arrival = "closed"
    wall = time.perf_counter() - started
    scraped = await scrape_overhead_p99(args.base)
    return summarise(
        args.name, args.judge, args.upstream, args.concurrency, arrival, samples, wall, scraped
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--name", default="adhoc")
    parser.add_argument("--judge", default="step")
    parser.add_argument("--upstream", default="fake", help="what the service is talking to")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--arrival-rate", type=float, default=0.0)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    if args.warmup:
        asyncio.run(closed_loop(args.base, args.judge, 1, args.warmup, args.timeout))

    result = asyncio.run(run_scenario(args))
    text = json.dumps(asdict(result), indent=2)
    if args.out:
        with open(args.out, "w") as handle:
            handle.write(text + "\n")
    print(text)
    if result.error_rate > 0 and args.name not in {"degraded", "overload"}:
        print(f"error rate {result.error_rate} in a scenario that should not fail", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
