"""Bootstrap confidence intervals for every number the paper quotes.

Reads the committed raw verdicts (data/), reproduces the published point estimates
exactly (the script aborts if any disagrees with results/tables in the code repo),
then computes 95% intervals by a stratified paired percentile bootstrap:

- resampling unit: trajectory;
- strata: the 8 design cells (clean; each failure type split by outcome survival),
  cell sizes preserved exactly in every replicate, so silent/loud/per-type margins
  stay internally consistent;
- one resample is shared by all five judges per replicate, which is what licenses
  the paired between-judge deltas;
- B = 10,000, numpy default_rng(20260829), percentile method;
- cells observed at 0/n or n/n get Clopper-Pearson exact intervals instead (the
  percentile bootstrap collapses to a point there).

The intervals quantify resampling uncertainty over the fixed 400-trajectory design,
conditional on generation seed 7 and greedy decoding. Deterministic: rerunning
produces a byte-identical ci.json.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "analysis" / "ci.json"

B = 10_000
SEED = 20260829
N_BINS = 10

JUDGES = [
    "programmatic",
    "outcome:qwen2.5:14b",
    "step:qwen2.5:14b",
    "step:llama3.1:8b",
    "selfcons3:qwen2.5:14b",
]
TYPES = [
    "wrong_tool",
    "hallucinated_argument",
    "skipped_precondition",
    "ignored_observation",
    "premature_stop",
    "unsupported_claim",
]

# Published values (results/tables/summary.md and per_type_recall.md at v0.1.0).
# The script refuses to emit ci.json unless it reproduces every one of these.
EXPECTED_SUMMARY = {
    "programmatic": (0.800, 0.429, 1.000, 0.000, 1.000, 0.667, 0.075),
    "outcome:qwen2.5:14b": (0.712, 0.451, 0.840, 0.330, None, 0.300, 0.253),
    "step:qwen2.5:14b": (0.923, 0.766, 0.984, 0.000, 0.973, 0.606, 0.033),
    "step:llama3.1:8b": (0.852, 0.983, 1.000, 1.000, 0.807, 0.311, 0.148),
    "selfcons3:qwen2.5:14b": (0.913, 0.760, 0.960, 0.010, 0.982, 0.583, 0.084),
}
EXPECTED_PER_TYPE = {
    "programmatic": (0.00, 1.00, 1.00, 1.00, 1.00, 0.00),
    "outcome:qwen2.5:14b": (0.34, 0.34, 0.74, 0.76, 1.00, 0.50),
    "step:qwen2.5:14b": (1.00, 1.00, 1.00, 1.00, 0.96, 0.18),
    "step:llama3.1:8b": (1.00, 0.94, 1.00, 1.00, 1.00, 1.00),
    "selfcons3:qwen2.5:14b": (1.00, 1.00, 1.00, 1.00, 0.90, 0.16),
}


def load() -> tuple[list[dict], dict[str, list[dict]]]:
    trajectories = [
        json.loads(line) for line in (DATA / "trajectories.jsonl").read_text().splitlines()
    ]
    by_judge: dict[str, dict[str, dict]] = {j: {} for j in JUDGES}
    for line in (DATA / "verdicts.jsonl").read_text().splitlines():
        v = json.loads(line)
        if v["judge_id"] in by_judge:
            by_judge[v["judge_id"]][v["trajectory_id"]] = v
    order = [t["trajectory_id"] for t in trajectories]
    aligned = {j: [by_judge[j][tid] for tid in order] for j in JUDGES}
    return trajectories, aligned


class JudgeArrays:
    """Per-trajectory vectors for one judge, aligned to the canonical order."""

    def __init__(self, trajectories: list[dict], verdicts: list[dict]) -> None:
        labels = [t["label"] for t in trajectories]
        self.lab_faulty = np.array([bool(x["faulty"]) for x in labels])
        self.lab_outcome = np.array([bool(x["outcome_correct"]) for x in labels])
        self.lab_type = np.array(
            [x["failure_type"] if x["failure_type"] else "" for x in labels], dtype=object
        )
        lab_step = np.array(
            [x["failure_step"] if x["failure_step"] is not None else -1 for x in labels]
        )
        self.v_faulty = np.array([bool(v["faulty"]) for v in verdicts])
        self.v_type = np.array(
            [v["failure_type"] if v["failure_type"] else "" for v in verdicts], dtype=object
        )
        v_step = np.array(
            [v["failure_step"] if v["failure_step"] is not None else -1 for v in verdicts]
        )
        self.conf = np.array([float(v["confidence"]) for v in verdicts])
        self.correct = self.lab_faulty == self.v_faulty
        self.silent = self.lab_faulty & self.lab_outcome
        self.loud = self.lab_faulty & ~self.lab_outcome
        self.clean = ~self.lab_faulty
        self.localisable = self.lab_faulty & self.v_faulty & (lab_step >= 0) & (v_step >= 0)
        self.step_hit = self.localisable & (lab_step == v_step)


def _safe(num: float, den: float) -> float:
    return num / den if den else 0.0


def metrics(a: JudgeArrays, idx: np.ndarray) -> dict[str, float]:
    lf, vf = a.lab_faulty[idx], a.v_faulty[idx]
    tp = float(np.sum(lf & vf))
    fp = float(np.sum(~lf & vf))
    fn = float(np.sum(lf & ~vf))
    precision = _safe(tp, tp + fp)
    recall = _safe(tp, tp + fn)
    f1 = _safe(2 * precision * recall, precision + recall)

    silent, loud, clean = a.silent[idx], a.loud[idx], a.clean[idx]
    out = {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "silent_recall": _safe(float(np.sum(silent & vf)), float(np.sum(silent))),
        "loud_recall": _safe(float(np.sum(loud & vf)), float(np.sum(loud))),
        "false_alarm_rate": _safe(float(np.sum(clean & vf)), float(np.sum(clean))),
        "step_exact": _safe(float(np.sum(a.step_hit[idx])), float(np.sum(a.localisable[idx]))),
        "step_scored_n": float(np.sum(a.localisable[idx])),
    }

    # Type macro-F1 over the six classes, faulty subset only; a flagged-but-untyped
    # or unflagged faulty trajectory falls in the `missed` sink (no class gets its FP).
    lt, vt = a.lab_type[idx], a.v_type[idx]
    typed = lf & (lt != "")
    pred = np.where(vf & (vt != ""), vt, "missed")
    f1s = []
    for t in TYPES:
        tp_t = float(np.sum(typed & (lt == t) & (pred == t)))
        fp_t = float(np.sum(typed & (lt != t) & (pred == t)))
        fn_t = float(np.sum(typed & (lt == t) & (pred != t)))
        p_t = _safe(tp_t, tp_t + fp_t)
        r_t = _safe(tp_t, tp_t + fn_t)
        f1s.append(_safe(2 * p_t * r_t, p_t + r_t))
    out["type_f1"] = float(np.mean(f1s))

    for t in TYPES:
        mask = typed & (lt == t)
        out[f"recall_{t}"] = _safe(float(np.sum(mask & vf)), float(np.sum(mask)))

    # ECE: 10 equal-width bins over [0,1], left-open right-closed, population-weighted,
    # against the binary verdict. Confidences are clamped to [0.5, 1] upstream.
    conf, corr = a.conf[idx], a.correct[idx].astype(float)
    bins = np.clip(np.ceil(conf * N_BINS).astype(int) - 1, 0, N_BINS - 1)
    ece = 0.0
    n = len(conf)
    for b in range(N_BINS):
        members = bins == b
        m = float(np.sum(members))
        if m:
            ece += (m / n) * abs(float(np.mean(corr[members])) - float(np.mean(conf[members])))
    out["ece"] = ece
    out["brier"] = float(np.mean((conf - corr) ** 2))
    return out


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact binomial interval via bisection on the binomial CDF (no scipy)."""

    def cdf_at_least(p: float) -> float:  # P(X >= k)
        return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k, n + 1))

    def cdf_at_most(p: float) -> float:  # P(X <= k)
        return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(0, k + 1))

    def solve(fn, target, lo, hi, increasing):
        for _ in range(200):
            mid = (lo + hi) / 2
            if (fn(mid) < target) == increasing:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    lower = 0.0 if k == 0 else solve(cdf_at_least, alpha / 2, 0.0, 1.0, True)
    upper = 1.0 if k == n else solve(cdf_at_most, alpha / 2, 0.0, 1.0, False)
    return lower, upper


def main() -> None:
    trajectories, aligned = load()
    arrays = {j: JudgeArrays(trajectories, aligned[j]) for j in JUDGES}
    n = len(trajectories)

    # The 8 design cells, sizes asserted against the committed composition.
    labels = [t["label"] for t in trajectories]
    cells: dict[str, list[int]] = {}
    for i, lab in enumerate(labels):
        if not lab["faulty"]:
            key = "clean"
        else:
            key = f"{lab['failure_type']}-{'silent' if lab['outcome_correct'] else 'loud'}"
        cells.setdefault(key, []).append(i)
    expected_cells = {
        "clean": 100,
        "wrong_tool-silent": 50,
        "hallucinated_argument-silent": 50,
        "unsupported_claim-silent": 50,
        "skipped_precondition-silent": 25,
        "skipped_precondition-loud": 25,
        "ignored_observation-loud": 50,
        "premature_stop-loud": 50,
    }
    assert {k: len(v) for k, v in cells.items()} == expected_cells, "design cells drifted"

    # Point estimates must reproduce the published tables before any interval is emitted.
    full = np.arange(n)
    points = {j: metrics(arrays[j], full) for j in JUDGES}
    for j, (f1, sil, loud, fa, step, tf1, ece) in EXPECTED_SUMMARY.items():
        got = points[j]
        checks = {
            "f1": f1, "silent_recall": sil, "loud_recall": loud,
            "false_alarm_rate": fa, "type_f1": tf1, "ece": ece,
        }
        if step is not None:
            checks["step_exact"] = step
        for metric, want in checks.items():
            assert round(got[metric], 3) == want, (j, metric, got[metric], want)
    for j, wants in EXPECTED_PER_TYPE.items():
        for t, want in zip(TYPES, wants):
            assert round(points[j][f"recall_{t}"], 2) == want, (j, t)
    print("point estimates reproduce results/tables exactly")

    rng = np.random.default_rng(SEED)
    cell_arrays = [np.array(v) for v in cells.values()]
    reps: dict[str, dict[str, list[float]]] = {j: {} for j in JUDGES}
    delta_reps: dict[str, list[float]] = {}
    deltas = {
        "outcome_loud_minus_silent": lambda m: m["outcome:qwen2.5:14b"]["loud_recall"]
        - m["outcome:qwen2.5:14b"]["silent_recall"],
        "step_minus_outcome_silent": lambda m: m["step:qwen2.5:14b"]["silent_recall"]
        - m["outcome:qwen2.5:14b"]["silent_recall"],
        "step_minus_rules_silent": lambda m: m["step:qwen2.5:14b"]["silent_recall"]
        - m["programmatic"]["silent_recall"],
        "selfcons_minus_step_f1": lambda m: m["selfcons3:qwen2.5:14b"]["f1"]
        - m["step:qwen2.5:14b"]["f1"],
        "selfcons_minus_step_silent": lambda m: m["selfcons3:qwen2.5:14b"]["silent_recall"]
        - m["step:qwen2.5:14b"]["silent_recall"],
        "selfcons_minus_step_type_f1": lambda m: m["selfcons3:qwen2.5:14b"]["type_f1"]
        - m["step:qwen2.5:14b"]["type_f1"],
        "selfcons_minus_step_ece": lambda m: m["selfcons3:qwen2.5:14b"]["ece"]
        - m["step:qwen2.5:14b"]["ece"],
        "selfcons_minus_step_step_exact": lambda m: m["selfcons3:qwen2.5:14b"]["step_exact"]
        - m["step:qwen2.5:14b"]["step_exact"],
    }
    for _ in range(B):
        idx = np.concatenate([rng.choice(c, size=len(c), replace=True) for c in cell_arrays])
        rep = {j: metrics(arrays[j], idx) for j in JUDGES}
        for j in JUDGES:
            for k, v in rep[j].items():
                reps[j].setdefault(k, []).append(v)
        for name, fn in deltas.items():
            delta_reps.setdefault(name, []).append(fn(rep))

    # Denominators for the Clopper-Pearson fallback on degenerate proportions.
    denominators = {
        "silent_recall": 175, "loud_recall": 125, "false_alarm_rate": 100,
        **{f"recall_{t}": 50 for t in TYPES},
    }

    def interval(judge: str, metric: str) -> dict:
        point = points[judge][metric]
        den = denominators.get(metric)
        if metric == "step_exact":
            den = int(points[judge]["step_scored_n"])
        if den and (point == 0.0 or point == 1.0):
            k = round(point * den)
            lo, hi = clopper_pearson(k, den)
            return {"point": point, "lo": lo, "hi": hi, "n": den, "method": "clopper-pearson"}
        dist = np.array(reps[judge][metric])
        lo, hi = np.percentile(dist, [2.5, 97.5])
        return {"point": point, "lo": float(lo), "hi": float(hi), "method": "percentile"}

    result = {
        "meta": {
            "B": B, "seed": SEED, "method": "stratified paired percentile bootstrap",
            "cells": expected_cells, "n_trajectories": n,
            "note": "one resample shared across all judges per replicate; "
            "degenerate proportions use exact Clopper-Pearson",
        },
        "judges": {
            j: {
                m: interval(j, m)
                for m in [
                    "f1", "precision", "recall", "silent_recall", "loud_recall",
                    "false_alarm_rate", "step_exact", "type_f1", "ece", "brier",
                    *[f"recall_{t}" for t in TYPES],
                ]
                if not (j == "outcome:qwen2.5:14b" and m == "step_exact")
            }
            for j in JUDGES
        },
        "deltas": {
            name: {
                "point": float(np.round(fn({j: points[j] for j in JUDGES}), 6)),
                "lo": float(np.percentile(delta_reps[name], 2.5)),
                "hi": float(np.percentile(delta_reps[name], 97.5)),
            }
            for name, fn in deltas.items()
        },
        "extras": {
            j: {"step_scored_n": int(points[j]["step_scored_n"])} for j in JUDGES
        },
    }

    def rounded(obj):
        if isinstance(obj, float):
            return round(obj, 6)
        if isinstance(obj, dict):
            return {k: rounded(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [rounded(v) for v in obj]
        return obj

    OUT.write_text(json.dumps(rounded(result), indent=2, sort_keys=True) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
