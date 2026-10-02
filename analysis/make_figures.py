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


def signed(x: float) -> str:
    """Two signed decimals with a true minus sign."""
    return f"{x:+.2f}".replace("-", "\u2212")


def whisker(cell: dict) -> tuple[float, float] | None:
    lo, hi, point = cell["lo"], cell["hi"], cell["point"]
    if hi - lo < 1e-9:
        return None  # degenerate under the design-conditioned bootstrap (rule engine)
    return point - lo, hi - point


def paired_rates(trajectories, verdicts) -> dict[str, dict[str, tuple[float, float, int]]]:
    """Per judge and stratum: share of faults flagged, share of their clean parents flagged.

    Only faults whose clean parent was judged are used, and a parent counts once per fault it
    hosts, so the difference of the two shares is the paired discrimination.
    """
    by_id = {t["trajectory_id"]: t for t in trajectories}
    pairs = []
    for t in trajectories:
        lab = t["label"]
        parent = f"{t['instance_id']}-clean"
        if lab["faulty"] and parent in by_id:
            pairs.append((t["trajectory_id"], parent, not lab["outcome_correct"]))
    rates: dict[str, dict[str, tuple[float, float, int]]] = {}
    for j in JUDGES:
        rates[j] = {}
        for stratum, loud in (("silent", False), ("loud", True)):
            subset = [(f, p) for f, p, is_loud in pairs if is_loud == loud]
            fault = float(np.mean([verdicts[j][f]["faulty"] for f, _ in subset]))
            parent = float(np.mean([verdicts[j][p]["faulty"] for _, p in subset]))
            rates[j][stratum] = (fault, parent, len(subset))
    return rates


def fig_silent_vs_loud(trajectories, verdicts) -> None:
    """Silent and loud faults, each against the clean runs they were derived from.

    Per judge a hollow circle marks how often it flags the clean parents and a filled marker
    how often it flags the faults; the segment between them is the paired discrimination.
    Recall alone is the filled marker, which is how a judge that flags everything looks
    perfect. Identity is carried by the row label, so colour is never needed to read it.
    """
    rates = paired_rates(trajectories, verdicts)
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.35), sharey=True)
    rows = list(range(len(JUDGES)))[::-1]
    for ax, stratum, title in [
        (axes[0], "silent", "silent faults"),
        (axes[1], "loud", "loud faults"),
    ]:
        n = rates[JUDGES[0]][stratum][2]
        for y, j in zip(rows, JUDGES, strict=True):
            fault, parent, _ = rates[j][stratum]
            ax.plot(
                [parent, fault], [y, y], color=COLOR[j], lw=2.0, zorder=2, solid_capstyle="round"
            )
            ax.scatter(
                [parent], [y], s=62, facecolors="white", edgecolors=INK, linewidths=0.9, zorder=3
            )
            ax.scatter(
                [fault],
                [y],
                s=20,
                marker=MARKER[j],
                color=COLOR[j],
                edgecolors=INK,
                linewidths=0.5,
                zorder=4,
            )
            ax.text(
                1.04,
                y,
                signed(fault - parent),
                va="center",
                ha="left",
                fontsize=6.8,
                color=INK,
                transform=ax.get_yaxis_transform(),
            )
        ax.set_xlim(-0.07, 1.07)
        ax.set_xticks([0, 0.5, 1.0], ["0", ".5", "1"])
        ax.xaxis.grid(True, zorder=0)
        ax.set_title(f"{title} ({n} pairs)", fontsize=7.6, loc="left")
        ax.text(
            1.04,
            1.0,
            "\u0394",
            va="bottom",
            ha="left",
            fontsize=7.2,
            color=INK,
            transform=ax.transAxes,
        )
        ax.set_xlabel("share flagged", fontsize=7.6)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(rows, [SHORT[j] for j in JUDGES], fontsize=7.2)
    handles = [
        plt.Line2D(
            [],
            [],
            marker="o",
            ls="",
            markersize=7,
            markerfacecolor="white",
            markeredgecolor=INK,
            label="clean parents flagged",
        ),
        plt.Line2D(
            [], [], marker="P", ls="", markersize=4.5, color="#777777", label="faults flagged"
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncols=2,
        fontsize=6.9,
        bbox_to_anchor=(0.55, -0.02),
        handletextpad=0.3,
        columnspacing=1.2,
    )
    fig.tight_layout(rect=(0, 0.07, 0.97, 1))
    fig.savefig(FIGS / "fig2_silent_vs_loud.pdf")
    plt.close(fig)


# Fault types in the order of the by-type figure: the four that leave the final reply
# unchanged, then the two that change it. The tag says whether the environment outcome
# survives the fault (a silent fault), does not (a loud one), or depends on the scenario.
BY_TYPE_ROWS = [
    ("wrong_tool", "silent"),
    ("hallucinated_argument", "silent"),
    ("skipped_precondition", "silent or loud"),
    ("ignored_observation", "loud"),
    ("unsupported_claim", "silent"),
    ("premature_stop", "loud"),
]
N_REPLY_UNCHANGED = 4


def paired_rates_by_type(trajectories, verdicts, judges) -> dict[str, dict[str, tuple]]:
    """Per judge and fault type: share of faults flagged, share of their parents flagged, pairs."""
    by_id = {t["trajectory_id"]: t for t in trajectories}
    rates: dict[str, dict[str, tuple]] = {j: {} for j in judges}
    for ftype, _ in BY_TYPE_ROWS:
        pairs = [
            (t["trajectory_id"], f"{t['instance_id']}-clean")
            for t in trajectories
            if t["label"]["failure_type"] == ftype and f"{t['instance_id']}-clean" in by_id
        ]
        for j in judges:
            fault = float(np.mean([verdicts[j][f]["faulty"] for f, _ in pairs]))
            parent = float(np.mean([verdicts[j][p]["faulty"] for _, p in pairs]))
            rates[j][ftype] = (fault, parent, len(pairs))
    return rates


def fig_recall_vs_parent(trajectories, verdicts) -> None:
    """Each fault type against its clean parents, for the two single-pass 14B judges.

    A ring marks how often the judge flags the clean parents and a filled marker how often it
    flags the faults derived from them; where the two coincide the judge's recall is its flag
    rate on correct runs. The band groups the four types that leave the final reply unchanged,
    on which the reply-only judge receives its parent's input.
    """
    judges = ["outcome:qwen2.5:14b", "step:qwen2.5:14b"]
    titles = {
        "outcome:qwen2.5:14b": "outcome judge",
        "step:qwen2.5:14b": "step judge",
    }
    rates = paired_rates_by_type(trajectories, verdicts, judges)
    # Row positions, with a gap between the reply-unchanged and reply-changed groups.
    ys = [5.5, 4.5, 3.5, 2.5, 1.0, 0.0]
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.05), sharey=True)
    for ax, j in zip(axes, judges, strict=True):
        ax.axhspan(1.95, 6.3, color="#e6e9ee", zorder=0, lw=0)
        for y, (ftype, _) in zip(ys, BY_TYPE_ROWS, strict=True):
            fault, parent, _ = rates[j][ftype]
            ax.plot(
                [parent, fault], [y, y], color=COLOR[j], lw=2.0, zorder=2, solid_capstyle="round"
            )
            ax.scatter(
                [parent], [y], s=62, facecolors="white", edgecolors=INK, linewidths=0.9, zorder=3
            )
            ax.scatter(
                [fault],
                [y],
                s=20,
                marker=MARKER[j],
                color=COLOR[j],
                edgecolors=INK,
                linewidths=0.4,
                zorder=4,
            )
            ax.text(
                1.05,
                y,
                signed(fault - parent),
                va="center",
                ha="left",
                fontsize=6.8,
                color=INK,
                transform=ax.get_yaxis_transform(),
            )
        ax.set_xlim(-0.07, 1.07)
        ax.set_ylim(-0.6, 6.3)
        ax.set_xticks([0, 0.5, 1.0], ["0", ".5", "1"], fontsize=7.5)
        ax.xaxis.grid(True, zorder=0)
        ax.set_title(titles[j], fontsize=7.6, loc="left")
        ax.text(
            1.05,
            1.0,
            "\u0394",
            va="bottom",
            ha="left",
            fontsize=7.2,
            color=INK,
            transform=ax.transAxes,
        )
        ax.set_xlabel("share flagged", fontsize=7.6)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(ys, [t for t, _ in BY_TYPE_ROWS], fontsize=6.9, family="monospace")
    # Third text column: what the fault does to the environment outcome.
    for y, (_, tag) in zip(ys, BY_TYPE_ROWS, strict=True):
        axes[1].text(
            1.31,
            y,
            tag,
            va="center",
            ha="left",
            fontsize=6.6,
            color="#555555",
            transform=axes[1].get_yaxis_transform(),
        )
    axes[1].text(
        1.31,
        1.0,
        "fault is",
        va="bottom",
        ha="left",
        fontsize=7.0,
        color=INK,
        transform=axes[1].transAxes,
    )
    # Group labels go in the left panel, where the coincident marks are the message and the
    # area left of the rings is empty. The caption explains ring and marker, so no legend.
    for y, text in ((6.02, "reply unchanged"), (1.5, "reply changed")):
        axes[0].text(
            -0.05, y, text, va="center", ha="left", fontsize=6.8, color=INK, style="italic"
        )
    # Fixed margins: the text columns sit outside the axes, which tight_layout would shrink.
    fig.subplots_adjust(left=0.235, right=0.80, top=0.90, bottom=0.20, wspace=0.40)
    fig.savefig(FIGS / "fig_recall_vs_parent.pdf")
    plt.close(fig)


def reliability(trajectories, verdicts, judge) -> list[tuple[float, float, int]]:
    label = {t["trajectory_id"]: t["label"] for t in trajectories}
    pairs = [
        (v["confidence"], label[tid]["faulty"] == v["faulty"]) for tid, v in verdicts[judge].items()
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
            xs,
            ys,
            lw=1.1,
            color=COLOR[j],
            zorder=2,
            ls="--" if recessive else "-",
            alpha=0.85 if recessive else 1.0,
        )
        ax.scatter(
            xs,
            ys,
            s=sizes,
            marker=MARKER[j],
            zorder=3,
            facecolors="white" if recessive else COLOR[j],
            edgecolors=COLOR[j],
            linewidths=0.9,
            label=SHORT[j] + (" (hand-set confidence)" if recessive else ""),
        )
        for cx, cy, count in curve:
            if count <= 5:  # tiny bins read as dramatic dives; say how tiny they are
                ax.annotate(
                    f"n={count}",
                    xy=(cx, cy),
                    xytext=(4, -9),
                    textcoords="offset points",
                    fontsize=6.2,
                    color="#666666",
                )
    # Direct labels on the two lines the analysis discusses most.
    out_curve = curves["outcome:qwen2.5:14b"]
    ax.annotate(
        "outcome (14B)",
        xy=out_curve[-1][:2],
        xytext=(6, -12),
        textcoords="offset points",
        fontsize=7.2,
        color=INK,
    )
    step_curve = curves["step:qwen2.5:14b"]
    ax.annotate(
        "step (14B)",
        xy=step_curve[-1][:2],
        xytext=(6, 4),
        textcoords="offset points",
        fontsize=7.2,
        color=INK,
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
                    c,
                    r,
                    str(v),
                    ha="center",
                    va="center",
                    fontsize=7,
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
    fig_recall_vs_parent(trajectories, verdicts)
    fig_silent_vs_loud(trajectories, verdicts)
    fig_calibration(trajectories, verdicts)
    fig_confusion(trajectories, verdicts)
    for f in sorted(FIGS.glob("*.pdf")):
        print(f"wrote {f}")


if __name__ == "__main__":
    main()
