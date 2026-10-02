#!/usr/bin/env bash
# The view-by-task ablation and the agent-episode judging, in order, against one Ollama session.
#
# Resumable: a rerun skips every verdict already on disk, so each step is retried after a
# failure. The engine version and model digest are logged before and after every step, and the
# queue stops if either changes, because cells judged on different engines are not comparable.
#
#   OLLAMA_HOST=http://127.0.0.1:11435 PYTHON=.venv/bin/python results/ablation/queue.sh
set -u

HOST="${OLLAMA_HOST:-http://127.0.0.1:11434}"
PYTHON="${PYTHON:-python}"
MODEL="qwen2.5:14b"
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HERE/raw"
ORGANIC="$HERE/organic"
ENGINE_LOG="$HERE/engine.txt"
RUN=("$PYTHON" -m trajectory_judge.cli run --model "$MODEL" --seed 7 --keep-responses)

engine() {
  local version digest
  version="$(curl -s "$HOST/api/version")"
  digest="$(curl -s "$HOST/api/tags" | "$PYTHON" -c '
import json, sys
models = json.load(sys.stdin).get("models", [])
print(next((m["digest"] for m in models if m["name"] == "'"$MODEL"'"), "missing"))')"
  echo "$version $digest"
}

PINNED="$(engine)"
{
  echo "$(date -u +%FT%TZ) start engine=$PINNED"
  echo "$(date -u +%FT%TZ) code=$("$PYTHON" -c 'import trajectory_judge; print(trajectory_judge.__file__)')"
} >>"$ENGINE_LOG"

step() {
  local name="$1"
  shift
  local attempt now
  for attempt in 1 2 3 4 5 6; do
    now="$(engine)"
    echo "$(date -u +%FT%TZ) $name attempt $attempt engine=$now" >>"$ENGINE_LOG"
    if [ "$now" != "$PINNED" ]; then
      echo "$(date -u +%FT%TZ) $name engine changed, stopping" >>"$ENGINE_LOG"
      exit 4
    fi
    if "${RUN[@]}" "$@"; then
      echo "$(date -u +%FT%TZ) $name done engine=$(engine)" >>"$ENGINE_LOG"
      return 0
    fi
    sleep 60
  done
  echo "$(date -u +%FT%TZ) $name gave up" >>"$ENGINE_LOG"
  exit 5
}

BENCH=(--n 400 --with-parents --first premature_stop,unsupported_claim --out "$OUT")
step "D' stepview-proctask" "${BENCH[@]}" --judges stepview-proctask
step "B outview-proctask" "${BENCH[@]}" --judges outview-proctask
step "C stepview-outtask" "${BENCH[@]}" --judges stepview-outtask
step "A' outview-outtask" "${BENCH[@]}" --judges outview-outtask
step "A outcome, own schema" "${BENCH[@]}" --judges outcome
step "agent episodes" --source "$HERE/../agent" --out "$ORGANIC" \
  --judges outview-outtask,stepview-proctask
echo "$(date -u +%FT%TZ) all done" >>"$ENGINE_LOG"
