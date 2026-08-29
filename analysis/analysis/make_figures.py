"""Regenerate every paper figure as a vector PDF from the committed raw verdicts.

Reads data/*.jsonl and analysis/ci.json only — no model calls, no network. The
committed PNGs in the code repository are never embedded; the paper's figures are
re-drawn here so they carry CI whiskers and print-grade vector text.

Palette: four categorical slots from a CVD-validated reference palette for the
four model judges; the rule engine is drawn as a recessive dashed gray reference
line because its confidence is hand-set, not measured. The confusion heatmap is
sequential (one hue, light to dark), which is the correct encoding for magnitude.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FIGS = ROOT / "paper" / "figures"

JUDGES = [
    "programmatic",
    "outcome:qwen2.5:14b",
    "step:qwen2.5:14b",
    "step:llama3.1:8b",
    "selfcons3:qwen2.5:14b",
]
SHORT = {
    "programmatic": "rules",
    "outcome:qwen2.5:14b": "outcome (14B)",
    "step:qwen2.5:14b": "step (14B)",
    "step:llama3.1:8b": "step (8B)",
    "selfcons3:qwen2.5:14b": "selfcons k=3 (14B)",
}
# Categorical slots (validated adjacent order) for the four model judges; gray for rules.
COLOR = {
    "programmatic": "#8a8f98",
    "outcome:qwen2.5:14b": "#2a78d6",
    "step:qwen2.5:14b": "#eb6834",
    "step:llama3.1:8b": "#eda100",
    "selfcons3:qwen2.5:14b": "#1baf7a",
}
MARKER = {
    "programmatic": "o",
    "outcome:qwen2.5:14b": "s",
    "step:qwen2.5:14b": "^",
    "step:llama3.1:8b": "D",
    "selfcons3:qwen2.5:14b": "v",
}
SILENT_BAR = "#2a78d6"
LOUD_BAR = "#b0b7c3"
ALARM = "#a23b33"
INK = "#1a1a1a"
TYPES = [
    "wrong_tool",
    "hallucinated_argument",
    "skipped_precondition",
    "ignored_observation",
    "premature_stop",
    "unsupported_claim",
]

plt.rcParams.update(
    {
        "font.size": 8.5,
        "axes.titlesize": 9,
        "axes.labelsize": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "axes.edgecolor": "#555555",
        "xtick.color": "#555555",
        "ytick.color": "#555555",
        "text.color": INK,
        "axes.labelcolor": INK,
        "grid.color": "#e6e6e6",
        "grid.linewidth": 0.5,
        "legend.frameon": False,
        "pdf.fonttype": 42,
    }
)


def load():
    trajectories = [
        json.loads(line) for line in (DATA / "trajectories.jsonl").read_text().splitlines()
    ]
    verdicts: dict[str, dict[str, dict]] = {j: {} for j in JUDGES}
    for line in (DATA / "verdicts.jsonl").read_text().splitlines():
        v = json.loads(line)
        if v["judge_id"] in verdicts:
            verdicts[v["judge_id"]][v["trajectory_id"]] = v
    ci = json.loads((ROOT / "analysis" / "ci.json").read_text())
    return trajectories, verdicts, ci


def whisker(cell: dict) -> tuple[float, float] | None:
    lo, hi, point = cell["lo"], cell["hi"], cell["point"]
    if hi - lo < 1e-9:
        return None  # degenerate under the design-conditioned bootstrap (rule engine)
    return point - lo, hi - point


def fig_silent_vs_loud(ci: dict) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 2.9))
    x = np.arange(len(JUDGES))
    width = 0.32
    for offset, key, color, label in [
        (-0.18, "loud_recall", LOUD_BAR, "loud (answer broke)"),
        (0.18, "silent_recall", SILENT_BAR, "silent (answer survived)"),
    ]:
        points = [ci["judges"][j][key]["point"] for j in JUDGES]
        bars = ax.bar(x + offset, points, width, color=color, label=label, zorder=2)
        errs = [whisker(ci["judges"][j][key]) for j in JUDGES]
        for xi, point, err in zip(x + offset, points, errs):
            if err is not None:
                ax.errorbar(
                    xi, point, yerr=[[err[0]], [err[1]]],
                    fmt="none", ecolor=INK, elinewidth=0.7, capsize=1.6, zorder=3,
                )
        for bar, point in zip(bars, points):
            ax.text(
                bar.get_x() + bar.get_width() / 2, 0.02, f"{point:.2f}".lstrip("0"),
                ha="center", va="bottom", fontsize=6.4, color="white"
                if point > 0.12 else INK, zorder=4,
            )
    fa = [ci["judges"][j]["false_alarm_rate"]["point"] for j in JUDGES]
    ax.scatter(
        x, fa, marker="x", s=42, color=ALARM, linewidths=1.6,
        label="false alarms (clean)", zorder=5,
    )
    ax.set_xticks(x, [SHORT[j].replace(" (", "\n(") for j in JUDGES], fontsize=7.4)
    ax.set_ylim(0, 1.14)
    ax.set_ylabel("recall")
    ax.yaxis.grid(True, zorder=0)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.04), fontsize=6.9, ncols=3,
              columnspacing=1.0, handletextpad=0.4)
    fig.tight_layout()
    fig.savefig(FIGS / "fig2_silent_vs_loud.pdf")
    plt.close(fig)


def reliability(trajectories, verdicts, judge) -> list[tuple[float, float, int]]:
    label = {t["trajectory_id"]: t["label"] for t in trajectories}
    pairs = [
        (v["confidence"], label[tid]["faulty"] == v["faulty"])
        for tid, v in verdicts[judge].items()
    ]
    curve = []
    for b in range(10):
        low, high = b / 10, (b + 1) / 10
        members = [(c, ok) for c, ok in pairs if low < c <= high]
        if members:
            curve.append(
                (
                    float(np.mean([c for c, _ in members])),
                    float(np.mean([ok for _, ok in members])),
                    len(members),
                )
            )
    return curve


def fig_calibration(trajectories, verdicts) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.3))
    ax.plot([0, 1.05], [0, 1.05], ls=":", lw=0.8, color="#999999", zorder=1)
    max_count = 0
    curves = {j: reliability(trajectories, verdicts, j) for j in JUDGES}
    for curve in curves.values():
        max_count = max(max_count, max(c for _, _, c in curve))
    for j in JUDGES:
        curve = curves[j]
        xs = [p[0] for p in curve]
        ys = [p[1] for p in curve]
        sizes = [14 + 130 * c / max_count for _, _, c in curve]
        recessive = j == "programmatic"
        ax.plot(
            xs, ys, lw=1.1, color=COLOR[j], zorder=2,
            ls="--" if recessive else "-", alpha=0.85 if recessive else 1.0,
        )
        ax.scatter(
            xs, ys, s=sizes, marker=MARKER[j], zorder=3,
            facecolors="white" if recessive else COLOR[j],
            edgecolors=COLOR[j], linewidths=0.9,
            label=SHORT[j] + (" (hand-set confidence)" if recessive else ""),
        )
        for cx, cy, count in curve:
            if count <= 5:  # tiny bins read as dramatic dives; say how tiny they are
                ax.annotate(
                    f"n={count}", xy=(cx, cy), xytext=(4, -9),
                    textcoords="offset points", fontsize=6.2, color="#666666",
                )
    # Direct labels on the two lines the analysis discusses most.
    out_curve = curves["outcome:qwen2.5:14b"]
    ax.annotate(
        "outcome (14B)", xy=out_curve[-1][:2], xytext=(6, -12),
        textcoords="offset points", fontsize=7.2, color=INK,
    )
    step_curve = curves["step:qwen2.5:14b"]
    ax.annotate(
        "step (14B)", xy=step_curve[-1][:2], xytext=(6, 4),
        textcoords="offset points", fontsize=7.2, color=INK,
    )
    ax.set_xlim(0.45, 1.06)
    ax.set_ylim(-0.02, 1.06)
    ax.set_xlabel("stated confidence (bin mean)")
    ax.set_ylabel("observed accuracy")
    ax.grid(True, zorder=0)
    ax.legend(loc="upper left", fontsize=6.9, handletextpad=0.4, borderaxespad=0.2)
    fig.tight_layout()
    fig.savefig(FIGS / "fig3_calibration.pdf")
    plt.close(fig)


def fig_confusion(trajectories, verdicts) -> None:
    label = {t["trajectory_id"]: t["label"] for t in trajectories}
    judge = "step:qwen2.5:14b"
    matrix = np.zeros((6, 7), dtype=int)
    for tid, v in verdicts[judge].items():
        lab = label[tid]
        if not lab["faulty"] or lab["failure_type"] is None:
            continue
        row = TYPES.index(lab["failure_type"])
        if v["faulty"] and v["failure_type"] in TYPES:
            matrix[row, TYPES.index(v["failure_type"])] += 1
        else:
            matrix[row, 6] += 1
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=50, aspect="auto")
    names = [t.replace("_", " ") for t in TYPES]
    ax.set_xticks(range(7), names + ["missed"], fontsize=6.4, rotation=30, ha="right")
    ax.set_yticks(range(6), [t.replace("_", "\n") for t in TYPES], fontsize=6.4)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true type")
    for r in range(6):
        for c in range(7):
            v = matrix[r, c]
            if v:
                ax.text(
                    c, r, str(v), ha="center", va="center", fontsize=7,
                    color="white" if v > 28 else INK,
                )
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.colorbar(im, ax=ax, shrink=0.8, label="trajectories")
    fig.tight_layout()
    fig.savefig(FIGS / "figA_confusion.pdf")
    plt.close(fig)


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    trajectories, verdicts, ci = load()
    fig_silent_vs_loud(ci)
    fig_calibration(trajectories, verdicts)
    fig_confusion(trajectories, verdicts)
    for f in sorted(FIGS.glob("*.pdf")):
        print(f"wrote {f}")


if __name__ == "__main__":
    main()
