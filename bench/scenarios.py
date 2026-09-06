"""Run the overhead sweep: start a fake backend and a service, drive both, write the JSON.

Five scenarios, chosen so that each isolates one thing:

``floor``     the mock judge, no I/O at all. This is the cost of ASGI, Pydantic, the metrics
              and the logging: the framework layer and nothing else.
``rules``     the rule engine. Real work at real speed, with context validation on the path.
``pooled``    the step judge against a backend that sleeps a known 250 ms. Overhead against a
              known model time, and the saturation curve.
``degraded``  the same, with a tenth of requests hanging past the timeout. Failure handling.
``overload``  four times the semaphore's capacity offered at once. Backpressure.

``pooled`` produces two independent estimates of overhead: the client's own
``total - 250 ms``, and the service's ``tj_service_overhead_seconds`` scraped from /metrics.
Two estimates that agree is the argument for believing either.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx

HERE = Path(__file__).parent
ROOT = HERE.parent
FAKE_PORT = 11500
SERVICE_PORT = 8111
FAKE_LATENCY_MS = 250.0


def ensure_port_free(port: int) -> None:
    """Refuse to run if something already holds the port.

    A leftover process from an earlier run keeps serving while the new one fails to bind, and
    the benchmark then measures a stranger. That failure is silent and it invalidates the whole
    sweep, so it is worth one socket call.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.5)
        if probe.connect_ex(("127.0.0.1", port)) == 0:
            raise SystemExit(
                f"port {port} is already in use; kill the stale process before benchmarking"
            )


def wait_for(url: str, timeout_s: float = 30.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            if httpx.get(url, timeout=2.0).status_code < 500:
                return
        except httpx.HTTPError:
            time.sleep(0.2)
    raise SystemExit(f"nothing answered at {url} within {timeout_s:g}s")


def spawn(args: list[str], env: dict[str, str]) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        args, env={**os.environ, **env}, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )


def drive(**kwargs: Any) -> dict[str, Any]:
    args = [sys.executable, str(HERE / "driver.py"), "--base", f"http://127.0.0.1:{SERVICE_PORT}"]
    for key, value in kwargs.items():
        args += [f"--{key.replace('_', '-')}", str(value)]
    done = subprocess.run(args, capture_output=True, text=True)
    if done.returncode != 0 and "should not fail" not in done.stderr:
        raise SystemExit(f"driver failed: {done.stderr[-500:]}")
    payload: dict[str, Any] = json.loads(done.stdout)
    return payload


def run(label: str, quick: bool) -> dict[str, Any]:
    for port in (FAKE_PORT, SERVICE_PORT):
        ensure_port_free(port)
    requests = 60 if quick else 200
    results: list[dict[str, Any]] = []

    # 1. no upstream at all: floor and rules
    service = spawn(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "--factory",
            "--port",
            str(SERVICE_PORT),
            "--log-level",
            "warning",
            "trajectory_judge.serve.app:create_app",
        ],
        {"TJ_ENABLED_JUDGES": "mock,programmatic", "TJ_MAX_CONCURRENCY": "8"},
    )
    try:
        wait_for(f"http://127.0.0.1:{SERVICE_PORT}/healthz")
        for name, judge in (("floor", "mock"), ("rules", "programmatic")):
            for concurrency in (1, 8) if quick else (1, 8, 32):
                results.append(
                    drive(
                        name=name,
                        judge=judge,
                        upstream="none",
                        concurrency=concurrency,
                        requests=requests,
                    )
                )
    finally:
        service.send_signal(signal.SIGTERM)
        service.wait(timeout=30)

    # 2. against the fake backend: pooled, degraded, overload
    for name, hang_rate, concurrencies, conc_limit in (
        ("pooled", 0.0, (1, 4, 8) if quick else (1, 2, 4, 8, 16), 8),
        ("degraded", 0.1, (8,), 8),
        ("overload", 0.0, (32,), 4),
    ):
        fake = spawn(
            [
                sys.executable,
                str(HERE / "fake_ollama.py"),
                "--port",
                str(FAKE_PORT),
                "--latency-ms",
                str(FAKE_LATENCY_MS),
                "--hang-rate",
                str(hang_rate),
                "--hang-s",
                "8",
            ],
            {},
        )
        service = spawn(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "--factory",
                "--port",
                str(SERVICE_PORT),
                "--log-level",
                "warning",
                "trajectory_judge.serve.app:create_app",
            ],
            {
                "OLLAMA_HOST": f"http://127.0.0.1:{FAKE_PORT}",
                "TJ_ENABLED_JUDGES": "step",
                "TJ_MAX_CONCURRENCY": str(conc_limit),
                "TJ_UPSTREAM_TIMEOUT_S": "2",
                "TJ_QUEUE_TIMEOUT_S": "0.4" if name == "overload" else "30",
            },
        )
        try:
            wait_for(f"http://127.0.0.1:{FAKE_PORT}/api/tags")
            wait_for(f"http://127.0.0.1:{SERVICE_PORT}/healthz")
            for concurrency in concurrencies:
                results.append(
                    drive(
                        name=name,
                        judge="step",
                        upstream=f"fake {FAKE_LATENCY_MS:g}ms",
                        concurrency=concurrency,
                        requests=requests,
                        timeout=30,
                    )
                )
        finally:
            for proc in (service, fake):
                proc.send_signal(signal.SIGTERM)
                proc.wait(timeout=30)

    sys.path.insert(0, str(HERE))
    from driver import environment

    return {
        "environment": environment(label),
        "fake_latency_ms": FAKE_LATENCY_MS,
        "scenarios": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="unlabelled machine")
    parser.add_argument("--quick", action="store_true", help="fewer points, for CI")
    parser.add_argument("--out", default=str(HERE / "results" / "overhead.json"))
    args = parser.parse_args()

    payload = run(args.label, args.quick)
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.out} with {len(payload['scenarios'])} scenarios")


if __name__ == "__main__":
    main()
