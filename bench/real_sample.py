"""A small sample against a real Ollama, to show the ratio rather than characterise the model.

Deliberately 30 requests, not 3000. The expected finding is stated here in advance rather than
discovered: this service's own cost is a fraction of a percent of the total, and the p99 is a
property of the model on this machine.

The concurrency curve is the part worth having. Ollama serialises per loaded model unless
OLLAMA_NUM_PARALLEL says otherwise, so latency grows roughly linearly with concurrency while
throughput stays flat. That is the evidence behind the operational rule
`TJ_MAX_CONCURRENCY = OLLAMA_NUM_PARALLEL`.

    python bench/real_sample.py --model qwen2.5:14b --requests 30
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from driver import environment, run_scenario  # noqa: E402
from scenarios import ensure_port_free, wait_for  # noqa: E402

SERVICE_PORT = 8113


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen2.5:14b")
    parser.add_argument("--host", default=os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
    parser.add_argument("--requests", type=int, default=30)
    parser.add_argument("--label", default="unlabelled machine")
    parser.add_argument("--concurrency", default="1,2,4")
    parser.add_argument("--out", default=str(HERE / "results" / "real_sample.json"))
    args = parser.parse_args()

    ensure_port_free(SERVICE_PORT)
    service = subprocess.Popen(
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
        env={
            **os.environ,
            "OLLAMA_HOST": args.host,
            "TJ_ENABLED_JUDGES": "step",
            "TJ_DEFAULT_MODEL": args.model,
            "TJ_MAX_CONCURRENCY": "8",
            "TJ_QUEUE_TIMEOUT_S": "600",
            "TJ_UPSTREAM_TIMEOUT_S": "300",
        },
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    import asyncio

    results: list[dict[str, Any]] = []
    try:
        wait_for(f"http://127.0.0.1:{SERVICE_PORT}/healthz")
        base = f"http://127.0.0.1:{SERVICE_PORT}"
        for concurrency in [int(c) for c in args.concurrency.split(",")]:
            print(f"  concurrency {concurrency} ...", flush=True)
            namespace = argparse.Namespace(
                base=base,
                name="real",
                judge="step",
                upstream=f"ollama {args.model}",
                concurrency=concurrency,
                requests=args.requests,
                arrival_rate=0.0,
                seconds=0.0,
                timeout=600.0,
            )
            results.append(asdict(asyncio.run(run_scenario(namespace))))
    finally:
        service.send_signal(signal.SIGTERM)
        service.wait(timeout=60)

    payload = {"environment": environment(args.label), "model": args.model, "scenarios": results}
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
