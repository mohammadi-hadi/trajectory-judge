.PHONY: install test lint build demo run report clean

MODEL ?= qwen2.5:14b
N     ?= 400
SEED  ?= 7

install:
	pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check src tests
	ruff format --check src tests
	mypy src

# Build sdist + wheel from a pristine `git archive` export, so a distribution can only ever
# contain committed files, and verify the metadata renders. CI installs the wheel into a bare
# environment and runs the suite against it, so a broken package fails before a release does.
build:
	rm -rf dist
	tmp=$$(mktemp -d) && git archive --prefix=pkg/ HEAD | tar -x -C "$$tmp" && \
		python -m build --outdir "$(CURDIR)/dist" "$$tmp/pkg" && rm -rf "$$tmp"
	twine check --strict dist/*

# End-to-end on the deterministic mock judge: no Ollama, no network, under a minute.
demo:
	trajectory-judge run --n 20 --judges mock --out results/demo --seed $(SEED)
	trajectory-judge report --raw results/demo --out results/demo

# The real run. Resumable: re-running skips (trajectory, judge) pairs already in the JSONL.
run:
	trajectory-judge run --n $(N) --model $(MODEL) --seed $(SEED) --out results/raw

# Rebuilds every table and figure from committed JSONL. Zero model calls.
report:
	trajectory-judge report --raw results/raw --out results

clean:
	rm -rf results/demo .pytest_cache .ruff_cache .mypy_cache
