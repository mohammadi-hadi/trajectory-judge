"""Turn committed raw verdicts into the tables and figures the README quotes.

Tables are deterministic text with fixed precision, which is what lets CI regenerate them from
the committed JSONL and fail on any drift. Figures are not byte-reproducible across matplotlib
versions, so they are committed for the README to render but never diffed — the numbers are
guarded, the pixels are not.
"""

from __future__ import annotations

from pathlib import Path

from trajectory_judge import store
from trajectory_judge.metrics import MISSED, Scores, reliability_curve, score, type_confusion
from trajectory_judge.trace import FailureType, Trajectory, Verdict

ACCENT = "#1E3A5F"
MUTED = "#B0B7C3"
ALARM = "#A23B33"

SUMMARY = "summary.md"
PER_TYPE = "per_type_recall.md"
CONFUSION = "confusion.md"
AGENT = "agent_runs.md"


def _fmt(value: float, places: int = 3) -> str:
    return f"{value:.{places}f}"


def _judge_order(verdicts: list[Verdict]) -> list[str]:
    """Stable ordering: rule engine first, then judges in first-seen order."""
    seen: list[str] = []
    for verdict in verdicts:
        if verdict.judge_id not in seen:
            seen.append(verdict.judge_id)
    # sorted(), not list.sort(): the key reads positions in `seen`, which an in-place sort moves.
    return sorted(seen, key=lambda judge_id: (judge_id != "programmatic", seen.index(judge_id)))


def score_all(trajectories: list[Trajectory], verdicts: list[Verdict]) -> list[Scores]:
    by_judge: dict[str, list[Verdict]] = {}
    for verdict in verdicts:
        by_judge.setdefault(verdict.judge_id, []).append(verdict)
    return [score(judge, trajectories, by_judge[judge]) for judge in _judge_order(verdicts)]


def summary_table(scores: list[Scores]) -> str:
    header = (
        "| Judge | n | Detection F1 | Silent recall | Loud recall | False alarms "
        "| Step exact | Type macro-F1 | ECE | s / trajectory |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    rows = "".join(
        f"| `{s.judge_id}` | {s.n} | {_fmt(s.f1)} | {_fmt(s.silent_recall)} "
        f"| {_fmt(s.loud_recall)} | {_fmt(s.false_alarm_rate)} | {_fmt(s.step_exact)} "
        f"| {_fmt(s.type_macro_f1)} | {_fmt(s.ece)} | {_fmt(s.mean_latency_s, 1)} |\n"
        for s in scores
    )
    # Per judge, not once for the table: a judge run on a subset has different denominators, and
    # a single strata line under the table would invite reading every row against the wrong ones.
    footer = (
        "\nStrata each judge actually saw:\n\n"
        "| Judge | clean | silent faults | loud faults |\n|---|---:|---:|---:|\n"
    )
    footer += "".join(
        f"| `{s.judge_id}` | {s.clean_n} | {s.silent_n} | {s.loud_n} |\n" for s in scores
    )
    footer += (
        "\nA *silent* fault left the customer-visible outcome correct; a *loud* one did not.\n"
    )
    return header + rows + footer


def per_type_table(scores: list[Scores]) -> str:
    types = [f.value for f in FailureType]
    header = "| Failure type | " + " | ".join(f"`{s.judge_id}`" for s in scores) + " |\n"
    header += "|---" * (len(scores) + 1) + "|\n"
    rows = ""
    for failure_type in types:
        cells = " | ".join(_fmt(s.per_type_recall.get(failure_type, 0.0), 2) for s in scores)
        rows += f"| {failure_type} | {cells} |\n"
    # The last row is the reference line, not a failure type: a judge that flags a third of
    # correct trajectories has no signal on a type it "detects" a third of the time, and a
    # recall column read on its own hides that.
    baseline = " | ".join(_fmt(s.false_alarm_rate, 2) for s in scores)
    rows += f"| *(false alarms on clean)* | {baseline} |\n"
    return (
        header
        + rows
        + (
            "\nRecall on each injected failure type. Read each column against its last row: recall "
            "at or near a judge's false-alarm rate is not detection, it is the judge's baseline "
            "willingness to say *faulty*.\n"
        )
    )


def confusion_table(judge_id: str, trajectories: list[Trajectory], verdicts: list[Verdict]) -> str:
    matrix = type_confusion(trajectories, [v for v in verdicts if v.judge_id == judge_id])
    columns = [f.value for f in FailureType] + [MISSED]
    out = f"### `{judge_id}`\n\n| true \\ predicted | " + " | ".join(columns) + " |\n"
    out += "|---" * (len(columns) + 1) + "|\n"
    for true_type in [f.value for f in FailureType]:
        cells = " | ".join(str(matrix[true_type][column]) for column in columns)
        out += f"| {true_type} | {cells} |\n"
    return out + "\n"


def agent_table(episodes: list[Trajectory]) -> str:
    """What a model actually does in this environment, labelled by the checker, not by an oracle.

    Reported separately from the judge comparison and never mixed into it: the checker cannot
    see two of the six failure types, so "no rule fired" is not the same claim as "clean".
    """
    total = len(episodes)
    if not total:
        return ""
    flagged = sum(1 for t in episodes if t.label.faulty)
    wrong_outcome = sum(1 for t in episodes if not t.label.outcome_correct)
    silent = sum(1 for t in episodes if t.label.silent)
    counts: dict[str, int] = {}
    for episode in episodes:
        if episode.label.failure_type is not None:
            key = episode.label.failure_type.value
            counts[key] = counts.get(key, 0) + 1

    out = f"Episodes played: **{total}**\n\n| Observation | Count | Share |\n|---|---:|---:|\n"
    for label, value in (
        ("flagged by the rule checker or wrong outcome", flagged),
        ("wrong customer-visible outcome", wrong_outcome),
        ("faulty but outcome still correct", silent),
    ):
        out += f"| {label} | {value} | {_fmt(value / total, 2)} |\n"
    if counts:
        out += "\n| Rule-visible failure type | Count |\n|---|---:|\n"
        for key in sorted(counts):
            out += f"| {key} | {counts[key]} |\n"
    return out + (
        "\nLabels here come from the rule checker, which is blind to `wrong_tool` and "
        "`unsupported_claim`, so the true fault rate is at least this high.\n"
    )


def _figures(
    out_dir: Path, scores: list[Scores], trajectories: list[Trajectory], verdicts: list[Verdict]
) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    # Silent versus loud recall: the headline picture.
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    labels = [s.judge_id for s in scores]
    positions = range(len(labels))
    width = 0.38
    ax.bar(
        [p - width / 2 for p in positions],
        [s.loud_recall for s in scores],
        width,
        label="outcome already broken",
        color=MUTED,
    )
    ax.bar(
        [p + width / 2 for p in positions],
        [s.silent_recall for s in scores],
        width,
        label="outcome still correct",
        color=ACCENT,
    )
    # Without this the always-say-faulty judge has two bars at 1.0 and looks like the best one
    # on the chart. Recall is only meaningful next to the rate at which a judge cries wolf.
    ax.plot(
        list(positions),
        [s.false_alarm_rate for s in scores],
        marker="x",
        markersize=8,
        markeredgewidth=2,
        linestyle="none",
        color=ALARM,
        label="false alarms on clean",
    )
    ax.set_xticks(list(positions))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("recall")
    ax.set_ylim(0, 1.05)
    ax.set_title("Fault recall, split by whether the fault changed the answer", fontsize=10)
    # Legend under the axes: every bar reaches the top of the plot, so there is no room inside.
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.30))
    ax.spines[["top", "right"]].set_visible(False)
    path = out_dir / "silent_vs_loud.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    written.append(path)

    # Reliability curves.
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.plot([0, 1], [0, 1], linestyle="--", color=MUTED, linewidth=1)
    by_judge = {s.judge_id: s for s in scores}
    for index, judge_id in enumerate(by_judge):
        subset = [v for v in verdicts if v.judge_id == judge_id]
        truth = {t.trajectory_id: t.label.faulty for t in trajectories}
        curve = reliability_curve(
            [v.confidence for v in subset if v.trajectory_id in truth],
            [v.faulty == truth[v.trajectory_id] for v in subset if v.trajectory_id in truth],
        )
        if not curve:
            continue
        ax.plot(
            [c for c, _, _ in curve],
            [a for _, a, _ in curve],
            marker="o",
            markersize=4,
            linewidth=1.4,
            alpha=1.0 - 0.14 * index,
            color=ACCENT,
            label=judge_id,
        )
    ax.set_xlabel("stated confidence")
    ax.set_ylabel("observed accuracy")
    ax.set_xlim(0.45, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_title("Calibration", fontsize=10)
    ax.legend(frameon=False, fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = out_dir / "calibration.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    written.append(path)
    return written


def build(raw_dir: Path, out_dir: Path) -> dict[str, list[Path]]:
    """Rebuild every table and figure from raw verdicts. No model calls, no network."""
    trajectories = store.read_trajectories(raw_dir)
    verdicts = store.read_verdicts(raw_dir)
    scores = score_all(trajectories, verdicts)

    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    # Provenance is read off the verdicts themselves, not off run.json: the judges that
    # actually produced these numbers cannot disagree with the header describing them.
    judges = ", ".join(_judge_order(verdicts)) or "none"
    provenance = (
        f"<!-- generated by `trajectory-judge report` from {len(verdicts)} verdicts over "
        f"{len(trajectories)} trajectories; judges: {judges} -->\n\n"
    )

    written: list[Path] = []
    for name, content in (
        (SUMMARY, summary_table(scores)),
        (PER_TYPE, per_type_table(scores)),
        (
            CONFUSION,
            "".join(confusion_table(s.judge_id, trajectories, verdicts) for s in scores),
        ),
    ):
        path = tables_dir / name
        path.write_text(provenance + content, encoding="utf-8")
        written.append(path)

    episodes = store.read_trajectories(raw_dir.parent / "agent")
    if episodes:
        path = tables_dir / AGENT
        path.write_text(provenance + agent_table(episodes), encoding="utf-8")
        written.append(path)

    figures = _figures(out_dir / "figures", scores, trajectories, verdicts)
    return {"tables": written, "figures": figures}
