# Agent-trajectory evaluation. The mock judge needs no model at all, so this
# image runs end to end with nothing else installed.
#   docker run --rm ghcr.io/mohammadi-hadi/trajectory-judge
#   docker run --rm -e OLLAMA_HOST=http://host.docker.internal:11434 \
#     ghcr.io/mohammadi-hadi/trajectory-judge run --n 100 --judges step_rubric
FROM python:3.12-slim

LABEL org.opencontainers.image.source="https://github.com/mohammadi-hadi/trajectory-judge" \
      org.opencontainers.image.description="How much an LLM judge misses when an agent reaches the right answer the wrong way" \
      org.opencontainers.image.licenses="MIT"

WORKDIR /build
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
# The report command renders figures, so matplotlib comes in with its extra.
RUN pip install --no-cache-dir ".[report]"

RUN useradd --create-home app
USER app
WORKDIR /home/app

# The same demo the Makefile ships: 20 trajectories against the mock judge, no
# model access. Output goes to /tmp because this runs as a non-root user with
# no writable project directory of its own.
ENTRYPOINT ["trajectory-judge"]
CMD ["run", "--n", "20", "--judges", "mock", "--out", "/tmp/demo", "--seed", "7"]
