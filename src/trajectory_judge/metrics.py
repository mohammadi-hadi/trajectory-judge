"""Scoring: detection, localisation, typing, calibration, cost.

Two choices worth stating, because they change the numbers:

**Type scoring is restricted to genuinely faulty trajectories.** A judge that flags a clean
trajectory and names a type for it is already punished by detection precision; counting it a
second time in the type confusion matrix would charge the same mistake twice. A *missed*
detection, however, does count as a miss for its true type — a judge cannot earn type credit
for faults it never noticed.

**Calibration scores the judge's own verdict, not the class.** ``correct`` is whether the
faulty/clean call matched the ground truth, and ``confidence`` is the judge's stated
probability that its call was right. That makes ECE comparable across judges that disagree
about how much is wrong.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from trajectory_judge.trace import FailureType, Trajectory, Verdict

N_CALIBRATION_BINS = 10
MISSED = "missed"


@dataclass
class Scores:
    """Everything the results table reports for one judge."""

    judge_id: str
    n: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    #: Recall restricted to faults that left the customer-visible outcome correct.
    silent_recall: float = 0.0
    silent_n: int = 0
    #: Recall on faults that also broke the outcome, i.e. the easy half.
    loud_recall: float = 0.0
    loud_n: int = 0
    #: Share of clean trajectories wrongly flagged.
    false_alarm_rate: float = 0.0
    clean_n: int = 0
    step_exact: float = 0.0
    step_within_one: float = 0.0
    step_scored_n: int = 0
    type_macro_f1: float = 0.0
    ece: float = 0.0
    brier: float = 0.0
    mean_latency_s: float = 0.0
    completion_tokens: int = 0
    errors: int = 0
    per_type_recall: dict[str, float] = field(default_factory=dict)


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _pair(
    trajectories: list[Trajectory], verdicts: list[Verdict]
) -> list[tuple[Trajectory, Verdict]]:
    """Match verdicts to trajectories, dropping verdicts with no trajectory in this run."""
    by_id = {t.trajectory_id: t for t in trajectories}
    return [(by_id[v.trajectory_id], v) for v in verdicts if v.trajectory_id in by_id]


def expected_calibration_error(
    confidences: list[float], correct: list[bool], bins: int = N_CALIBRATION_BINS
) -> float:
    """Equal-width binned |accuracy - confidence|, weighted by bin population."""
    if not confidences:
        return 0.0
    total = len(confidences)
    error = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        members = [
            (c, ok)
            for c, ok in zip(confidences, correct, strict=True)
            if (low < c <= high) or (index == 0 and c == 0.0)
        ]
        if not members:
            continue
        mean_confidence = sum(c for c, _ in members) / len(members)
        accuracy = sum(1 for _, ok in members if ok) / len(members)
        error += (len(members) / total) * abs(accuracy - mean_confidence)
    return error


def reliability_curve(
    confidences: list[float], correct: list[bool], bins: int = N_CALIBRATION_BINS
) -> list[tuple[float, float, int]]:
    """``(mean confidence, accuracy, count)`` per populated bin, for the calibration figure."""
    out: list[tuple[float, float, int]] = []
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        members = [(c, ok) for c, ok in zip(confidences, correct, strict=True) if low < c <= high]
        if not members:
            continue
        out.append(
            (
                sum(c for c, _ in members) / len(members),
                sum(1 for _, ok in members if ok) / len(members),
                len(members),
            )
        )
    return out


def type_confusion(
    trajectories: list[Trajectory], verdicts: list[Verdict]
) -> dict[str, dict[str, int]]:
    """True failure type against predicted type, with ``missed`` for "judge said clean"."""
    columns = [f.value for f in FailureType] + [MISSED]
    matrix = {f.value: dict.fromkeys(columns, 0) for f in FailureType}
    for trajectory, verdict in _pair(trajectories, verdicts):
        if not trajectory.label.faulty or trajectory.label.failure_type is None:
            continue
        true_type = trajectory.label.failure_type.value
        # "Flagged but unable to name a type" is a miss, not a free pass.
        if not verdict.faulty or verdict.failure_type is None:
            predicted = MISSED
        else:
            predicted = verdict.failure_type.value
        matrix[true_type][predicted] += 1
    return matrix


def _macro_f1(matrix: dict[str, dict[str, int]]) -> float:
    scores = []
    for label in matrix:
        true_positive = matrix[label][label]
        false_negative = sum(matrix[label].values()) - true_positive
        false_positive = sum(row[label] for other, row in matrix.items() if other != label)
        precision = _safe_div(true_positive, true_positive + false_positive)
        recall = _safe_div(true_positive, true_positive + false_negative)
        scores.append(_safe_div(2 * precision * recall, precision + recall))
    return sum(scores) / len(scores) if scores else 0.0


def score(judge_id: str, trajectories: list[Trajectory], verdicts: list[Verdict]) -> Scores:
    """Compute every reported metric for one judge over one set of trajectories."""
    pairs = _pair(trajectories, verdicts)
    result = Scores(judge_id=judge_id, n=len(pairs))
    if not pairs:
        return result

    true_positive = sum(1 for t, v in pairs if t.label.faulty and v.faulty)
    false_positive = sum(1 for t, v in pairs if not t.label.faulty and v.faulty)
    false_negative = sum(1 for t, v in pairs if t.label.faulty and not v.faulty)

    result.precision = _safe_div(true_positive, true_positive + false_positive)
    result.recall = _safe_div(true_positive, true_positive + false_negative)
    result.f1 = _safe_div(2 * result.precision * result.recall, result.precision + result.recall)

    silent = [(t, v) for t, v in pairs if t.label.silent]
    loud = [(t, v) for t, v in pairs if t.label.faulty and not t.label.outcome_correct]
    clean = [(t, v) for t, v in pairs if not t.label.faulty]
    result.silent_n, result.loud_n, result.clean_n = len(silent), len(loud), len(clean)
    result.silent_recall = _safe_div(sum(1 for _, v in silent if v.faulty), len(silent))
    result.loud_recall = _safe_div(sum(1 for _, v in loud if v.faulty), len(loud))
    result.false_alarm_rate = _safe_div(sum(1 for _, v in clean if v.faulty), len(clean))

    # Localisation is only meaningful where the judge both saw the fault and can point at steps.
    localisable = [
        (t, v)
        for t, v in pairs
        if t.label.faulty
        and v.faulty
        and t.label.failure_step is not None
        and v.failure_step is not None
    ]
    result.step_scored_n = len(localisable)
    result.step_exact = _safe_div(
        sum(1 for t, v in localisable if v.failure_step == t.label.failure_step), len(localisable)
    )
    result.step_within_one = _safe_div(
        sum(
            1
            for t, v in localisable
            if abs((v.failure_step or 0) - (t.label.failure_step or 0)) <= 1
        ),
        len(localisable),
    )

    matrix = type_confusion(trajectories, verdicts)
    result.type_macro_f1 = _macro_f1(matrix)

    faulty_by_type: Counter[str] = Counter()
    caught_by_type: Counter[str] = Counter()
    for trajectory, verdict in pairs:
        if trajectory.label.failure_type is None:
            continue
        key = trajectory.label.failure_type.value
        faulty_by_type[key] += 1
        caught_by_type[key] += int(verdict.faulty)
    result.per_type_recall = {
        key: _safe_div(caught_by_type[key], faulty_by_type[key]) for key in sorted(faulty_by_type)
    }

    confidences = [v.confidence for _, v in pairs]
    correct = [t.label.faulty == v.faulty for t, v in pairs]
    result.ece = expected_calibration_error(confidences, correct)
    result.brier = sum(
        (c - float(ok)) ** 2 for c, ok in zip(confidences, correct, strict=True)
    ) / len(pairs)

    result.mean_latency_s = sum(v.latency_s for _, v in pairs) / len(pairs)
    result.completion_tokens = sum(v.completion_tokens for _, v in pairs)
    result.errors = sum(1 for _, v in pairs if v.error)
    return result
