# Agent-trajectory evaluation. The mock judge needs no model at all, so this
# image runs end to end with nothing else installed.
#   docker run --rm ghcr.io/mohammadi-hadi/trajectory-judge
#   docker run --rm -e OLLAMA_HOST=http://host.docker.internal:11434 \
#     ghcr.io/mohammadi-hadi/trajectory-judge run --n 100 --judges step_rubric
# The same image serves the judges over HTTP:
#   docker run --rm -p 8000:8000 ghcr.io/mohammadi-hadi/trajectory-judge serve
#
# Single stage on purpose. Nothing here compiles: fastapi, uvicorn, pydantic, httpx, typer and
# prometheus-client all ship wheels, so a builder stage would strip no toolchain and would add
# a real failure mode in copied console-script shebangs.
FROM python:3.12-slim

LABEL org.opencontainers.image.source="https://github.com/mohammadi-hadi/trajectory-judge" \
      org.opencontainers.image.description="How much an LLM judge misses when an agent reaches the right answer the wrong way" \
      org.opencontainers.image.licenses="MIT"

# __version__ is pinned at 0.1.0 while the repo moves on, so /healthz reports the commit
# instead: a version that cannot change is not a deploy identifier.
ARG TJ_GIT_SHA=unknown
ENV TJ_GIT_SHA=${TJ_GIT_SHA} \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
# The report command renders figures, so matplotlib comes in with its extra.
RUN pip install --no-cache-dir ".[report,serve]"

RUN useradd --create-home app
USER app
WORKDIR /home/app

ENV TJ_MAX_CONCURRENCY=4 \
    TJ_UPSTREAM_TIMEOUT_S=60 \
    TJ_LOG_LEVEL=info
EXPOSE 8000

# No HEALTHCHECK here: the image has two modes, and one probing port 8000 would mark every
# demo container permanently unhealthy. The healthcheck belongs to the deployment, so it
# lives in compose.yaml.
#
# The default stays the demo the Makefile ships: 20 trajectories against the mock judge, no
# model access. Output goes to /tmp because this runs as a non-root user with no writable
# project directory of its own.
ENTRYPOINT ["trajectory-judge"]
CMD ["run", "--n", "20", "--judges", "mock", "--out", "/tmp/demo", "--seed", "7"]
