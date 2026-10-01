"""Paired analysis: every judge's verdict on a fault against its verdict on the clean parent.

Recall on injected faults says how often a judge flags them. It does not say whether the judge
can tell them from correct runs: a judge that flags three quarters of all full-refund replies
scores 76% recall on any fault hosted there, whether or not it sees the fault. Each fault in
this benchmark was derived from a clean run that is itself judged, so the fault's verdict can
be compared with its parent's. Paired discrimination is the share of faults a judge flags
minus the share of their parents it flags. Pairing with the clean run was suggested by a
reviewer of the workshop paper.

Writes analysis/paired.json. Reads only data/, never touches analysis/ci.json, and draws from
its own random streams, so every published interval stays byte-identical.

What it computes, for the published (v0.1.0) verdicts:
- which faults leave the judge-visible reply unchanged, and whether the outcome judge's
  verdict on them equals its verdict on the parent (asserted, not assumed);
- paired discrimination per judge and fault type, with exact McNemar tests on single types and
  an instance-clustered bootstrap for pooled rows and contrasts (one parent serves several
  faults, so pairs are not independent);
- the outcome judge's loud/silent split, pooled and paired, and its flag rate per scenario;
- localisation over detected faults (out-of-range steps counted as misses) and over all faults;
- detection-and-type, sensitivity, specificity, precision at deployment prevalence, Brier by
  class, and a few counts the paper quotes.

When the October ablation verdicts are present in data/, it also analyses the view-by-task
cells and the agent-episode judging, as specified in results/ablation/PREDICTIONS.md.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "analysis" / "paired.json"

B = 10_000
SEED = 20261001  # new streams; ci.json keeps 20260829

JUDGES = [
    "programmatic",
    "outcome:qwen2.5:14b",
    "step:qwen2.5:14b",
    "step:llama3.1:8b",
    "selfcons3:qwen2.5:14b",
]
OUTCOME, STEP, LLAMA, SELFCONS = JUDGES[1], JUDGES[2], JUDGES[3], JUDGES[4]
TYPES = [
    "wrong_tool",
    "hallucinated_argument",
    "skipped_precondition",
    "ignored_observation",
    "premature_stop",
    "unsupported_claim",
]
#: Fault types whose injected trajectory keeps the clean run's goal and final reply.
REPLY_SAME = {"wrong_tool", "hallucinated_argument", "skipped_precondition", "ignored_observation"}
#: Round-robin stratum of an instance, by index (env/world.py).
STRATA = ["happy", "restocking", "expired", "non_refundable", "wrong_customer", "already_refunded"]
PREVALENCE = [0.01, 0.05, 0.10, 0.25]
CP_UPPER_0_OF_100 = 0.0362  # two-sided 95% Clopper-Pearson upper bound for 0/100

#: The four sentences the injector appends for unsupported_claim (mutate.py), and stems that
#: show a rationale mentions the invented content.
TEMPLATES = {
    "replacement": (
        "A replacement has already been dispatched",
        ["replacement", "dispatch", "tuesday"],
    ),
    "subscription": ("cancelled the subscription", ["subscription", "cancel"]),
    "voucher": ("10 EUR voucher", ["voucher", "credited"]),
    "courier": ("A courier will collect", ["courier", "collect", "tomorrow"]),
}
REPLY_WORDS = re.compile(r"\b(repl(y|ies|ied|ying)|answer\w*|messages?|respon\w*)\b", re.IGNORECASE)
NUMBER = re.compile(r"\d+(?:\.\d+)?")


# ---------------------------------------------------------------------------- loading


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def stratum(instance_id: str) -> str:
    return STRATA[int(instance_id.split("-")[1]) % 6]


def by_key(verdicts: list[dict]) -> dict[tuple[str, str], dict]:
    return {(v["judge_id"], v["trajectory_id"]): v for v in verdicts}


# ---------------------------------------------------------------------------- statistics


def mcnemar_p(b10: int, b01: int) -> float | None:
    """Exact two-sided McNemar p-value; None when no pair is discordant."""
    n = b10 + b01
    if n == 0:
        return None
    k = min(b10, b01)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values."""
    ordered = sorted(pvalues.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    m = len(ordered)
    for rank, (name, p) in enumerate(ordered):
        running = max(running, min(1.0, (m - rank) * p))
        adjusted[name] = running
    return adjusted


def percentile_ci(samples) -> tuple[float, float]:
    lo, hi = np.percentile(np.array(samples), [2.5, 97.5])
    return float(lo), float(hi)


def bootstrap_p(samples) -> float:
    """Two-sided bootstrap p-value for a contrast against zero."""
    arr = np.array(samples)
    below = float(np.mean(arr <= 0))
    above = float(np.mean(arr >= 0))
    return min(1.0, 2 * min(below, above))


def ppv(sens: float, fa: float, prevalence: float) -> float:
    hit = sens * prevalence
    false = fa * (1 - prevalence)
    return hit / (hit + false) if hit + false else 0.0


# ---------------------------------------------------------------------------- pairing


class Pairs:
    """Faults whose clean parent is in the judged set, with per-judge flags for both."""

    def __init__(self, trajectories: list[dict], verdicts: dict[tuple[str, str], dict], judges):
        by_id = {t["trajectory_id"]: t for t in trajectories}
        self.rows: list[dict] = []
        for t in trajectories:
            lab = t["label"]
            if not lab["faulty"]:
                continue
            parent = by_id.get(f"{t['instance_id']}-clean")
            if parent is None:
                continue
            same = (t["goal"], t["final_answer"]) == (parent["goal"], parent["final_answer"])
            # The reply-unchanged label is a property of the fault type; check it holds.
            assert same == (lab["failure_type"] in REPLY_SAME), t["trajectory_id"]
            row = {
                "fault": t["trajectory_id"],
                "parent": parent["trajectory_id"],
                "instance": t["instance_id"],
                "type": lab["failure_type"],
                "loud": not lab["outcome_correct"],
                "reply_same": same,
            }
            for j in judges:
                row[j] = (
                    bool(verdicts[(j, t["trajectory_id"])]["faulty"]),
                    bool(verdicts[(j, parent["trajectory_id"])]["faulty"]),
                )
            self.rows.append(row)
        self.instances = sorted({r["instance"] for r in self.rows})
        position = {inst: i for i, inst in enumerate(self.instances)}
        self.instance_index = np.array([position[r["instance"]] for r in self.rows])
        self.masks = {cell: np.array([in_cell(r, cell) for r in self.rows]) for cell in CELLS}
        # +1 fault flagged and parent not, -1 the reverse, 0 when the two verdicts agree.
        self.diff = {j: np.array([int(r[j][0]) - int(r[j][1]) for r in self.rows]) for j in judges}

    def weights(self, rng: np.random.Generator) -> np.ndarray:
        """B x pairs: how often each pair is drawn when instances are resampled with
        replacement. One clean parent serves several faults, so the instance is the unit."""
        n = len(self.instances)
        counts = rng.multinomial(n, np.full(n, 1.0 / n), size=B)
        return counts[:, self.instance_index].astype(float)


def in_cell(row: dict, cell: str) -> bool:
    if cell in TYPES:
        return row["type"] == cell
    return {
        "reply_same": row["reply_same"],
        "reply_same_kept": row["reply_same"] and not row["loud"],
        "reply_same_broke": row["reply_same"] and row["loud"],
        "reply_changed": not row["reply_same"],
        "loud": row["loud"],
        "silent": not row["loud"],
        "all": True,
    }[cell]


CELLS = [
    *TYPES,
    "reply_same",
    "reply_changed",
    "loud",
    "silent",
    "all",
    "reply_same_kept",
    "reply_same_broke",
]


def paired_table(pairs: Pairs, judges: list[str], rng: np.random.Generator):
    """Point estimates, discordant counts, McNemar on single types, clustered CIs throughout."""
    w = pairs.weights(rng)
    table: dict = {}
    reps: dict[tuple[str, str], np.ndarray] = {}
    for j in judges:
        d = pairs.diff[j]
        table[j] = {}
        for cell in CELLS:
            m = pairs.masks[cell]
            den = w @ m.astype(float)
            reps[(j, cell)] = np.where(den > 0, (w @ (m * d)) / np.maximum(den, 1), 0.0)
            b10 = int(np.sum(m & (d == 1)))
            b01 = int(np.sum(m & (d == -1)))
            n = int(m.sum())
            lo, hi = percentile_ci(reps[(j, cell)])
            entry = {
                "n": n,
                "delta": (b10 - b01) / n if n else 0.0,
                "b10": b10,
                "b01": b01,
                "lo": lo,
                "hi": hi,
                # No discordant pair at all: the zero is exact, not an estimate.
                "structural": b10 + b01 == 0,
            }
            if cell in TYPES:
                entry["mcnemar_p"] = mcnemar_p(b10, b01)
            table[j][cell] = entry
    return table, reps


# ---------------------------------------------------------------------------- per-trajectory


class Arrays:
    """Per-trajectory vectors for one judge over the 400-trajectory design."""

    def __init__(self, trajectories: list[dict], verdicts: list[dict]) -> None:
        labels = [t["label"] for t in trajectories]
        self.faulty = np.array([bool(x["faulty"]) for x in labels])
        self.loud = np.array([bool(x["faulty"]) and not x["outcome_correct"] for x in labels])
        self.type = np.array([x["failure_type"] or "" for x in labels], dtype=object)
        self.step = np.array(
            [x["failure_step"] if x["failure_step"] is not None else -1 for x in labels]
        )
        self.flag = np.array([bool(v["faulty"]) for v in verdicts])
        self.vtype = np.array([v["failure_type"] or "" for v in verdicts], dtype=object)
        self.vstep = np.array(
            [v["failure_step"] if v["failure_step"] is not None else -1 for v in verdicts]
        )
        self.conf = np.array([float(v["confidence"]) for v in verdicts])
        self.correct = self.faulty == self.flag


def trajectory_metrics(a: Arrays, idx: np.ndarray) -> dict[str, float]:
    faulty, flag = a.faulty[idx], a.flag[idx]
    clean = ~faulty
    detected = faulty & flag
    valid = detected & (a.vstep[idx] >= 0)
    exact = valid & (a.vstep[idx] == a.step[idx])
    omission = a.type[idx] == "premature_stop"
    n_faulty = float(np.sum(faulty))
    sens = float(np.sum(detected)) / n_faulty
    fa = float(np.sum(clean & flag)) / float(np.sum(clean))
    conf, corr = a.conf[idx], a.correct[idx].astype(float)
    sq = (conf - corr) ** 2
    out = {
        "sens": sens,
        "spec": 1 - fa,
        "fa": fa,
        "bal_acc": (sens + 1 - fa) / 2,
        "loc_det": float(np.sum(exact)) / max(1.0, float(np.sum(detected))),
        "loc_joint": float(np.sum(exact)) / n_faulty,
        "loc_nonomit": float(np.sum(exact & ~omission))
        / max(1.0, float(np.sum(detected & ~omission))),
        "loc_joint_nonomit": float(np.sum(exact & ~omission)) / float(np.sum(faulty & ~omission)),
        "type_joint": float(np.sum(detected & (a.vtype[idx] == a.type[idx]))) / n_faulty,
        "brier": float(np.mean(sq)),
        "brier_faulty": float(np.mean(sq[faulty])),
        "brier_clean": float(np.mean(sq[clean])),
    }
    for p in PREVALENCE:
        out[f"ppv_{p}"] = ppv(sens, fa, p)
    return out


def ece(conf: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    idx = np.clip(np.ceil(conf * bins).astype(int) - 1, 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def design_cells(trajectories: list[dict]) -> list[np.ndarray]:
    cells: dict[str, list[int]] = {}
    for i, t in enumerate(trajectories):
        lab = t["label"]
        key = (
            "clean"
            if not lab["faulty"]
            else f"{lab['failure_type']}-{'silent' if lab['outcome_correct'] else 'loud'}"
        )
        cells.setdefault(key, []).append(i)
    return [np.array(v) for v in cells.values()]


# ---------------------------------------------------------------------------- main analyses


def published(trajectories: list[dict], verdicts: list[dict], agent: list[dict]) -> dict:
    vmap = by_key(verdicts)
    labels = {t["trajectory_id"]: t["label"] for t in trajectories}
    by_id = {t["trajectory_id"]: t for t in trajectories}
    rng = np.random.default_rng(np.random.SeedSequence(SEED).spawn(3)[0])
    result: dict = {}

    # Pairing, and the identity the outcome judge must satisfy on reply-unchanged pairs.
    pairs = Pairs(trajectories, vmap, JUDGES)
    reply_same_pairs = [r for r in pairs.rows if r["reply_same"]]
    identical = 0
    for r in reply_same_pairs:
        a, b = vmap[(OUTCOME, r["fault"])], vmap[(OUTCOME, r["parent"])]
        keys = ("faulty", "confidence", "failure_type", "rationale")
        identical += all(a[k] == b[k] for k in keys)
    faults = [t for t in trajectories if t["label"]["faulty"]]
    reply_same_faults = [t for t in faults if t["label"]["failure_type"] in REPLY_SAME]
    design = {
        "n_faults": len(faults),
        "n_reply_same": len(reply_same_faults),
        "n_reply_same_kept": sum(1 for t in reply_same_faults if t["label"]["outcome_correct"]),
        "n_reply_same_broke": sum(
            1 for t in reply_same_faults if not t["label"]["outcome_correct"]
        ),
        "n_pairs": len(pairs.rows),
        "n_pairs_reply_same": len(reply_same_pairs),
        "n_pairs_loud": sum(1 for r in pairs.rows if r["loud"]),
        "n_pairs_silent": sum(1 for r in pairs.rows if not r["loud"]),
        "n_late_pairs": len(faults) - len(pairs.rows),
        "n_late_parents": len(
            {t["instance_id"] for t in faults} - {r["instance"] for r in pairs.rows}
        ),
        "outcome_identical_pairs": identical,
    }
    assert design["n_pairs"] == 251 and design["n_pairs_reply_same"] == 151, design
    assert identical == 151, identical
    result["design"] = design

    table, reps = paired_table(pairs, JUDGES, rng)
    result["paired"] = table
    unsup_contrast = reps[(STEP, "unsupported_claim")] - reps[(OUTCOME, "unsupported_claim")]
    gap = reps[(OUTCOME, "loud")] - reps[(OUTCOME, "silent")]
    result["contrasts"] = {
        "unsup_step_minus_outcome": {
            "point": table[STEP]["unsupported_claim"]["delta"]
            - table[OUTCOME]["unsupported_claim"]["delta"],
            "ci": percentile_ci(unsup_contrast),
        },
        "outcome_paired_loud_minus_silent": {
            "point": table[OUTCOME]["loud"]["delta"] - table[OUTCOME]["silent"]["delta"],
            "ci": percentile_ci(gap),
        },
    }

    # The outcome judge by scenario: clean runs and reply-unchanged faults.
    out_v = {tid: vmap[(OUTCOME, tid)] for tid in labels}
    strata: dict[str, dict] = {}
    for s in STRATA:
        clean = [
            t for t in trajectories if not t["label"]["faulty"] and stratum(t["instance_id"]) == s
        ]
        same = [t for t in reply_same_faults if stratum(t["instance_id"]) == s]
        strata[s] = {
            "clean_flagged": sum(out_v[t["trajectory_id"]]["faulty"] for t in clean),
            "clean_n": len(clean),
            "reply_same_flagged": sum(out_v[t["trajectory_id"]]["faulty"] for t in same),
            "reply_same_n": len(same),
        }
    result["outcome_strata"] = strata
    loud = [t for t in faults if not t["label"]["outcome_correct"]]
    result["outcome_loud"] = {
        "flags": sum(out_v[t["trajectory_id"]]["faulty"] for t in loud),
        "flags_reply_same": sum(
            out_v[t["trajectory_id"]]["faulty"]
            for t in loud
            if t["label"]["failure_type"] in REPLY_SAME
        ),
    }
    per_type_recall = {
        ft: sum(
            out_v[t["trajectory_id"]]["faulty"] for t in faults if t["label"]["failure_type"] == ft
        )
        / sum(1 for t in faults if t["label"]["failure_type"] == ft)
        for ft in TYPES
    }
    result["outcome_per_type_recall"] = per_type_recall
    rows = {
        "reply_same_kept": [t for t in reply_same_faults if t["label"]["outcome_correct"]],
        "reply_same_broke": [t for t in reply_same_faults if not t["label"]["outcome_correct"]],
    }
    result["cell_recall"] = {
        j: {
            name: sum(vmap[(j, t["trajectory_id"])]["faulty"] for t in faults_in) / len(faults_in)
            for name, faults_in in rows.items()
        }
        for j in JUDGES
    }
    result["clean_flags"] = {
        j: sum(
            vmap[(j, t["trajectory_id"])]["faulty"]
            for t in trajectories
            if not t["label"]["faulty"]
        )
        for j in JUDGES
    }

    # Per-trajectory metrics with a design-cell bootstrap on its own stream.
    order = [t["trajectory_id"] for t in trajectories]
    arrays = {j: Arrays(trajectories, [vmap[(j, tid)] for tid in order]) for j in JUDGES}
    full = np.arange(len(trajectories))
    points = {j: trajectory_metrics(arrays[j], full) for j in JUDGES}
    cells = design_cells(trajectories)
    rng2 = np.random.default_rng(np.random.SeedSequence(SEED).spawn(3)[1])
    reps2: dict[tuple[str, str], list[float]] = {}
    delta_reps: dict[str, list[float]] = {}
    for _ in range(B):
        idx = np.concatenate([rng2.choice(c, size=len(c), replace=True) for c in cells])
        rep = {j: trajectory_metrics(arrays[j], idx) for j in JUDGES}
        for j in JUDGES:
            for k, v in rep[j].items():
                reps2.setdefault((j, k), []).append(v)
        eces = {
            j: ece(arrays[j].conf[idx], arrays[j].correct[idx].astype(float))
            for j in (OUTCOME, STEP)
        }
        delta_reps.setdefault("outcome_minus_step_ece", []).append(eces[OUTCOME] - eces[STEP])
        delta_reps.setdefault("outcome_minus_step_brier", []).append(
            rep[OUTCOME]["brier"] - rep[STEP]["brier"]
        )
        delta_reps.setdefault("selfcons_minus_step_brier", []).append(
            rep[SELFCONS]["brier"] - rep[STEP]["brier"]
        )
    result["trajectory"] = {
        j: {k: {"point": v, "ci": percentile_ci(reps2[(j, k)])} for k, v in points[j].items()}
        for j in JUDGES
    }
    full_ece = {j: ece(arrays[j].conf, arrays[j].correct.astype(float)) for j in (OUTCOME, STEP)}
    result["calibration_deltas"] = {
        "outcome_minus_step_ece": {
            "point": full_ece[OUTCOME] - full_ece[STEP],
            "ci": percentile_ci(delta_reps["outcome_minus_step_ece"]),
        },
        "outcome_minus_step_brier": {
            "point": points[OUTCOME]["brier"] - points[STEP]["brier"],
            "ci": percentile_ci(delta_reps["outcome_minus_step_brier"]),
        },
        "selfcons_minus_step_brier": {
            "point": points[SELFCONS]["brier"] - points[STEP]["brier"],
            "ci": percentile_ci(delta_reps["selfcons_minus_step_brier"]),
        },
    }

    # Localisation counts, asserted against the hand check in the plan.
    loc: dict[str, dict] = {}
    for j in JUDGES:
        if j == OUTCOME:
            continue
        a = arrays[j]
        detected = a.faulty & a.flag
        valid = detected & (a.vstep >= 0)
        exact = valid & (a.vstep == a.step)
        omission = a.type == "premature_stop"
        loc[j] = {
            "detected": int(detected.sum()),
            "valid": int(valid.sum()),
            "invalid": int((detected & (a.vstep < 0)).sum()),
            "exact": int(exact.sum()),
            "nonomit_exact": int((exact & ~omission).sum()),
            "nonomit_detected": int((detected & ~omission).sum()),
        }
    assert (loc[STEP]["exact"], loc[STEP]["valid"], loc[STEP]["detected"]) == (217, 223, 257)
    assert loc[STEP]["nonomit_exact"] == loc[STEP]["nonomit_detected"] == 209
    result["localisation_counts"] = loc

    # Step judge mistyping, PPV floors for judges with no clean flag.
    step_a = arrays[STEP]
    detected = step_a.faulty & step_a.flag
    result["step_mistyped"] = int((detected & (step_a.vtype != step_a.type)).sum())
    result["ppv_floor_5pct"] = {
        j: ppv(points[j]["sens"], CP_UPPER_0_OF_100, 0.05) for j in JUDGES if points[j]["fa"] == 0.0
    }

    # Self-consistency: votes are recorded in the rationale ("k/3 votes for faulty").
    votes = {}
    for tid in order:
        m = re.match(r"(\d)/3 votes", vmap[(SELFCONS, tid)]["rationale"])
        assert m, tid
        votes[tid] = int(m.group(1))
    unsup = [
        t["trajectory_id"] for t in faults if t["label"]["failure_type"] == "unsupported_claim"
    ]
    missed = [tid for tid in unsup if votes[tid] < 2]
    any_flag = {tid: votes[tid] >= 1 for tid in order}
    clean_ids = [tid for tid in order if not labels[tid]["faulty"]]
    faulty_ids = [tid for tid in order if labels[tid]["faulty"]]
    any_sens = sum(any_flag[t] for t in faulty_ids) / len(faulty_ids)
    any_fa = sum(any_flag[t] for t in clean_ids) / len(clean_ids)
    rng3 = np.random.default_rng(np.random.SeedSequence(SEED).spawn(3)[2])
    any_unsup_reps = []
    for _ in range(B):
        sample = rng3.choice(np.array(unsup), size=len(unsup), replace=True)
        any_unsup_reps.append(float(np.mean([any_flag[t] for t in sample])))
    result["selfcons"] = {
        "unsup_missed": len(missed),
        "unsup_missed_unanimous": sum(1 for tid in missed if votes[tid] == 0),
        "unsup_missed_one_vote": sum(1 for tid in missed if votes[tid] == 1),
        "any_unsup": float(np.mean([any_flag[t] for t in unsup])),
        "any_unsup_ci": percentile_ci(any_unsup_reps),
        "any_fa": any_fa,
        "any_ppv_5pct": ppv(any_sens, any_fa, 0.05),
        "majority_ppv_5pct": points[SELFCONS]["ppv_0.05"],
    }

    # Rationales of the step judge on unsupported_claim (keyword-based, not hand-coded).
    templ_of = {}
    for tid in unsup:
        answer = by_id[tid]["final_answer"]
        templ_of[tid] = next(name for name, (text, _) in TEMPLATES.items() if text in answer)
    step_v = {tid: vmap[(STEP, tid)] for tid in unsup}
    step_missed = [tid for tid in unsup if not step_v[tid]["faulty"]]
    names_claim = 0
    for tid in step_missed:
        stems = TEMPLATES[templ_of[tid]][1]
        names_claim += any(s in step_v[tid]["rationale"].lower() for s in stems)
    result["step_unsup_rationales"] = {
        "missed": len(step_missed),
        "missed_mention_reply": sum(
            1 for tid in step_missed if REPLY_WORDS.search(step_v[tid]["rationale"])
        ),
        "missed_names_claim": names_claim,
        "missed_capped": sum(1 for tid in step_missed if len(step_v[tid]["rationale"]) >= 2000),
        "by_template": {
            name: {
                "caught": sum(1 for t in unsup if templ_of[t] == name and step_v[t]["faulty"]),
                "n": sum(1 for t in unsup if templ_of[t] == name),
            }
            for name in TEMPLATES
        },
    }

    # The 8B judge: localisation of what it flags, and how varied its rationales are.
    l_v = [vmap[(LLAMA, tid)] for tid in order]
    result["llama"] = {
        "exact_of_detected": loc[LLAMA]["exact"],
        "detected": loc[LLAMA]["detected"],
        "distinct_rationales": len({v["rationale"] for v in l_v}),
    }

    # A generic required-order rule, applied to every trajectory (iGXR #6).
    result["sequence_rule"] = sequence_rule(trajectories, agent)

    # Agent episodes: do refund replies state the amount actually refunded, and does any
    # reply equal the oracle's reply on the same instance?
    clean_reply = {
        t["instance_id"]: t["final_answer"] for t in trajectories if not t["label"]["faulty"]
    }
    exact_amount = refunds = 0
    for ep in agent:
        refunded = [
            s for s in ep["steps"] if s["call"]["tool"] == "issue_refund" and s["observation"]["ok"]
        ]
        if refunded:
            refunds += 1
            amount = float(refunded[-1]["call"]["args"]["amount_eur"])
            stated = [float(x) for x in NUMBER.findall(ep["final_answer"])]
            exact_amount += any(abs(x - amount) < 0.005 for x in stated)
    result["agent"] = {
        "refund_replies": refunds,
        "refund_replies_state_amount": exact_amount,
        "replies_identical_to_oracle": sum(
            1 for ep in agent if ep["final_answer"] == clean_reply.get(ep["instance_id"])
        ),
        "n": len(agent),
    }
    return result


def sequence_rule(trajectories: list[dict], agent: list[dict]) -> dict:
    """Flag any trajectory whose calls are not exactly the procedure's order.

    The order is get_customer, lookup_order, get_policy for the SKU the lookup returned,
    check_eligibility, exactly one of issue_refund or escalate, and reply last.
    """

    def violates(t: dict) -> bool:
        tools = [s["call"]["tool"] for s in t["steps"]]
        if tools[:4] != ["get_customer", "lookup_order", "get_policy", "check_eligibility"]:
            return True
        lookup = t["steps"][1]["observation"]
        if not lookup["ok"] or t["steps"][2]["call"]["args"].get("sku") != lookup["data"].get(
            "sku"
        ):
            return True
        rest = tools[4:]
        return rest not in (["issue_refund", "reply"], ["escalate", "reply"])

    def violates_lenient(t: dict) -> bool:
        """Extra calls allowed: only the procedure's order, the policy's SKU and reply last."""
        steps = t["steps"]
        wanted = ["get_customer", "lookup_order", "get_policy", "check_eligibility"]
        position = 0
        sku = None
        for s in steps:
            tool = s["call"]["tool"]
            if position < len(wanted) and tool == wanted[position]:
                if tool == "lookup_order" and s["observation"]["ok"]:
                    sku = s["observation"]["data"].get("sku")
                if tool == "get_policy" and s["call"]["args"].get("sku") != sku:
                    return True
                position += 1
            elif position == len(wanted) and tool in ("issue_refund", "escalate"):
                position += 1
        return position < len(wanted) + 1 or not steps or steps[-1]["call"]["tool"] != "reply"

    caught = {ft: 0 for ft in TYPES}
    hosts = {ft: 0 for ft in TYPES}
    clean_flags = clean_n = 0
    for t in trajectories:
        lab = t["label"]
        if lab["faulty"]:
            hosts[lab["failure_type"]] += 1
            caught[lab["failure_type"]] += violates(t)
        else:
            clean_n += 1
            clean_flags += violates(t)
    organic_clean = [
        ep for ep in agent if not ep["label"]["faulty"] and ep["label"]["outcome_correct"]
    ]
    return {
        "caught": caught,
        "hosts": hosts,
        "clean_flags": clean_flags,
        "clean_n": clean_n,
        "organic_clean_flags": sum(violates(ep) for ep in organic_clean),
        "organic_clean_flags_lenient": sum(violates_lenient(ep) for ep in organic_clean),
        "organic_clean_n": len(organic_clean),
        "lenient_caught": {
            ft: sum(
                violates_lenient(t)
                for t in trajectories
                if t["label"]["faulty"] and t["label"]["failure_type"] == ft
            )
            for ft in TYPES
        },
        "lenient_clean_flags": sum(
            violates_lenient(t) for t in trajectories if not t["label"]["faulty"]
        ),
    }


# ---------------------------------------------------------------------------- ablation

ABL = {
    "A": "outview-outtask:qwen2.5:14b",
    "B": "outview-proctask:qwen2.5:14b",
    "C": "stepview-outtask:qwen2.5:14b",
    "D": "stepview-proctask:qwen2.5:14b",
    "Ao": "outcome:qwen2.5:14b",
}


def ablation(published_verdicts: list[dict]) -> dict | None:
    """The view-by-task cells, if their verdicts are in data/. See PREDICTIONS.md."""
    path = DATA / "ablation_verdicts.jsonl"
    if not path.exists():
        return None
    trajectories = read_jsonl(DATA / "ablation_trajectories.jsonl")
    verdicts = read_jsonl(path)
    vmap = by_key(verdicts)
    present = {j for j, _ in vmap}
    cells = {name: jid for name, jid in ABL.items() if jid in present}
    for name, jid in cells.items():
        n = sum(1 for (j, _) in vmap if j == jid)
        assert n == len(trajectories), (name, n, len(trajectories))

    rng = np.random.default_rng(np.random.SeedSequence(SEED + 1).spawn(2)[0])
    pairs = Pairs(trajectories, vmap, list(cells.values()))
    assert len(pairs.rows) == 300, len(pairs.rows)
    table, reps = paired_table(pairs, list(cells.values()), rng)

    clean = [t["trajectory_id"] for t in trajectories if not t["label"]["faulty"]]
    rng_fa = np.random.default_rng(np.random.SeedSequence(SEED + 1).spawn(2)[1])
    fa_reps: dict[str, list[float]] = {name: [] for name in cells}
    flags = {
        name: np.array([vmap[(jid, tid)]["faulty"] for tid in clean]) for name, jid in cells.items()
    }
    for _ in range(B):
        idx = rng_fa.integers(0, len(clean), size=len(clean))
        for name in cells:
            fa_reps[name].append(float(flags[name][idx].mean()))
    fa = {
        name: {
            "point": float(flags[name].mean()),
            "k": int(flags[name].sum()),
            "n": len(clean),
            "ci": percentile_ci(fa_reps[name]),
        }
        for name in cells
    }

    def dd(a: str, b: str, cell: str) -> tuple[float, np.ndarray]:
        ja, jb = cells[a], cells[b]
        point = table[ja][cell]["delta"] - table[jb][cell]["delta"]
        return point, reps[(ja, cell)] - reps[(jb, cell)]

    contrasts: dict[str, dict] = {}
    if {"A", "B", "C", "D"} <= cells.keys():
        spec = {
            "C1_view_process_task": dd("D", "B", "reply_changed"),
            "C2_task_full_view_process_faults": dd("D", "C", "reply_same"),
            "C3_task_full_view_unsupported": dd("C", "D", "unsupported_claim"),
            "C5_published_comparison_one_session": dd("D", "A", "all"),
        }
        b_minus_a = np.array(fa_reps["B"]) - np.array(fa_reps["A"])
        spec["C4_task_outcome_view_false_alarms"] = (
            fa["B"]["point"] - fa["A"]["point"],
            b_minus_a,
        )
        raw_p = {name: bootstrap_p(samples) for name, (_, samples) in spec.items()}
        adjusted = holm(raw_p)
        for name, (point, samples) in spec.items():
            contrasts[name] = {
                "point": point,
                "ci": percentile_ci(samples),
                "p": raw_p[name],
                "p_holm": adjusted[name],
            }

    # Reproducibility against v0.1.0 (engine 0.30.11) on the 400 shared trajectories.
    pub = by_key(published_verdicts)
    repro: dict[str, dict] = {}
    for name, old in [("D", STEP), ("A", OUTCOME), ("Ao", OUTCOME)]:
        if name not in cells:
            continue
        shared = [tid for (j, tid) in pub if j == old and (cells[name], tid) in vmap]
        fields = ("faulty", "failure_type", "failure_step", "confidence", "rationale")
        repro[name] = {"n": len(shared)}
        for f in fields:
            repro[name][f] = sum(
                1 for tid in shared if vmap[(cells[name], tid)][f] == pub[(old, tid)][f]
            )
    if {"A", "Ao"} <= cells.keys():
        same = sum(
            1
            for t in trajectories
            if all(
                vmap[(cells["A"], t["trajectory_id"])][f]
                == vmap[(cells["Ao"], t["trajectory_id"])][f]
                for f in ("faulty", "rationale")
            )
        )
        repro["schema_swap_identical"] = {"k": same, "n": len(trajectories)}

    # Raw steps D' named on premature_stop, from the kept responses.
    raw_steps: dict[str, int] = {}
    resp_path = DATA / "ablation_responses.jsonl"
    if resp_path.exists() and "D" in cells:
        labels = {t["trajectory_id"]: t["label"] for t in trajectories}
        for r in read_jsonl(resp_path):
            if r["judge_id"] != cells["D"]:
                continue
            if labels[r["trajectory_id"]]["failure_type"] != "premature_stop":
                continue
            body = json.loads(r["text"])
            if not body.get("faulty"):
                continue
            key = str(body.get("failure_step"))
            raw_steps[key] = raw_steps.get(key, 0) + 1

    out = {
        "cells": cells,
        "paired": table,
        "false_alarms": fa,
        "contrasts": contrasts,
        "reproducibility": repro,
        "premature_raw_steps": raw_steps,
    }
    organic = DATA / "organic_verdicts.jsonl"
    if organic.exists():
        agent = read_jsonl(DATA / "agent_trajectories.jsonl")
        ov = by_key(read_jsonl(organic))
        out["organic"] = {}
        for name in ("A", "D"):
            jid = ABL[name]
            flagged = {
                ep["trajectory_id"]: ov[(jid, ep["trajectory_id"])]["faulty"] for ep in agent
            }
            faulty = [
                ep for ep in agent if ep["label"]["faulty"] or not ep["label"]["outcome_correct"]
            ]
            clean_eps = [ep for ep in agent if ep not in faulty]
            out["organic"][name] = {
                "faulty_flagged": sum(flagged[ep["trajectory_id"]] for ep in faulty),
                "faulty_n": len(faulty),
                "clean_flagged": sum(flagged[ep["trajectory_id"]] for ep in clean_eps),
                "clean_n": len(clean_eps),
            }
    return out


# ---------------------------------------------------------------------------- entry point


def rounded(obj):
    if isinstance(obj, float):
        return round(obj, 6)
    if isinstance(obj, tuple):
        return [rounded(v) for v in obj]
    if isinstance(obj, dict):
        return {k: rounded(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [rounded(v) for v in obj]
    return obj


def main() -> None:
    trajectories = read_jsonl(DATA / "trajectories.jsonl")
    verdicts = read_jsonl(DATA / "verdicts.jsonl")
    agent = read_jsonl(DATA / "agent_trajectories.jsonl")
    result = {
        "meta": {
            "B": B,
            "seed": SEED,
            "method": "instance-clustered pair bootstrap; "
            "design-cell trajectory bootstrap for per-trajectory metrics",
        }
    }
    result["published"] = published(trajectories, verdicts, agent)
    abl = ablation(verdicts)
    if abl is not None:
        result["ablation"] = abl
    OUT.write_text(json.dumps(rounded(result), indent=2, sort_keys=True) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
