"""An Ollama stand-in that sleeps a known amount and answers with a valid verdict.

The point is to measure the service, not the model. Pointing the real service at this exercises
the whole real path -- httpx, the connection pool, the thread offload, the semaphore, schema
decoding, metrics, logging -- and replaces only the model's thinking time with a constant.

That is a stronger measurement than benchmarking the mock judge, which never touches httpx at
all, and httpx plus the threadpool is where the interesting behaviour lives.

    python bench/fake_ollama.py --port 11500 --latency-ms 250 --hang-rate 0.1
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

VERDICT = json.dumps(
    {
        "reasoning": "step 2 issued a refund before the eligibility check came back",
        "faulty": True,
        "failure_step": 2,
        "failure_type": "skipped_precondition",
        "confidence": 0.82,
    }
)


def build_app(latency_s: float, hang_rate: float, hang_s: float, seed: int) -> FastAPI:
    # fastapi is imported at module level on purpose: with postponed annotations, a
    # function-local import leaves `Request` unresolvable, and FastAPI then reads the
    # parameter as a query field instead of the request object.
    app = FastAPI()
    rng = random.Random(seed)

    @app.get("/api/tags")
    async def tags() -> dict[str, list[dict[str, str]]]:
        return {"models": [{"name": "qwen2.5:14b"}, {"name": "llama3.1:8b"}]}

    @app.post("/api/generate")
    async def generate(request: Request) -> JSONResponse:
        await request.body()
        # A deliberate slow tail: the point of the degraded scenario is to show the failures
        # arrive as failures and the fast requests stay fast.
        hang = hang_rate > 0 and rng.random() < hang_rate
        await asyncio.sleep(hang_s if hang else latency_s)
        return JSONResponse(
            {
                "response": VERDICT,
                "prompt_eval_count": 850,
                "eval_count": 120,
                "done": True,
            }
        )

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=11500)
    parser.add_argument("--latency-ms", type=float, default=250.0)
    parser.add_argument("--hang-rate", type=float, default=0.0)
    parser.add_argument("--hang-s", type=float, default=120.0)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(
        build_app(args.latency_ms / 1000.0, args.hang_rate, args.hang_s, args.seed),
        host="127.0.0.1",
        port=args.port,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
