"""Metrics are checked against numbers worked out by hand, not against their own output."""

from __future__ import annotations

import pytest

from trajectory_judge.metrics import (
    MISSED,
    expected_calibration_error,
    reliability_curve,
    score,
    type_confusion,
)
from trajectory_judge.trace import FailureType, Label, Trajectory, Verdict


def _trajectory(
    tid: str,
    *,
    faulty: bool = False,
    failure_type: FailureType | None = None,
    failure_step: int | None = None,
    outcome_correct: bool = True,
) -> Trajectory:
    return Trajectory(
        trajectory_id=tid,
        instance_id=tid,
        goal="",
        label=Label(
            faulty=faulty,
            failure_type=failure_type,
            failure_step=failure_step,
            outcome_correct=outcome_correct,
        ),
    )


def _verdict(
    tid: str,
    *,
    faulty: bool = False,
    failure_type: FailureType | None = None,
    failure_step: int | None = None,
    confidence: float = 0.5,
) -> Verdict:
    return Verdict(
        trajectory_id=tid,
        judge_id="j",
        faulty=faulty,
        failure_type=failure_type,
        failure_step=failure_step,
        confidence=confidence,
    )


def test_expected_calibration_error_matches_hand_calculation() -> None:
    # Two at 0.9 with one right (|0.5 - 0.9| = 0.4, weight 0.5) and two at 0.6 both right
    # (|1.0 - 0.6| = 0.4, weight 0.5) give 0.2 + 0.2.
    ece = expected_calibration_error([0.9, 0.9, 0.6, 0.6], [True, False, True, True])
    assert ece == pytest.approx(0.4)


def test_a_perfectly_calibrated_judge_scores_zero() -> None:
    confidences = [0.75] * 4
    assert expected_calibration_error(confidences, [True, True, True, False]) == pytest.approx(0.0)


def test_reliability_curve_reports_populated_bins_only() -> None:
    curve = reliability_curve([0.9, 0.9, 0.6], [True, False, True])
    assert [count for _, _, count in curve] == [1, 2]  # bins ascend: (0.5, 0.6] then (0.8, 0.9]
    assert curve[0] == pytest.approx((0.6, 1.0, 1))
    assert curve[1] == pytest.approx((0.9, 0.5, 2))


def test_detection_and_stratified_recall() -> None:
    trajectories = [
        _trajectory("silent1", faulty=True, failure_type=FailureType.WRONG_TOOL, failure_step=2),
        _trajectory("silent2", faulty=True, failure_type=FailureType.WRONG_TOOL, failure_step=2),
        _trajectory(
            "loud1",
            faulty=True,
            failure_type=FailureType.PREMATURE_STOP,
            failure_step=1,
            outcome_correct=False,
        ),
        _trajectory("clean1"),
        _trajectory("clean2"),
    ]
    verdicts = [
        _verdict("silent1", faulty=False),
        _verdict("silent2", faulty=True, failure_type=FailureType.WRONG_TOOL, failure_step=2),
        _verdict("loud1", faulty=True, failure_type=FailureType.PREMATURE_STOP, failure_step=2),
        _verdict("clean1", faulty=True, failure_type=FailureType.WRONG_TOOL, failure_step=0),
        _verdict("clean2", faulty=False),
    ]
    result = score("j", trajectories, verdicts)

    assert (result.silent_n, result.loud_n, result.clean_n) == (2, 1, 2)
    assert result.silent_recall == pytest.approx(0.5)
    assert result.loud_recall == pytest.approx(1.0)
    assert result.false_alarm_rate == pytest.approx(0.5)
    assert result.precision == pytest.approx(2 / 3)
    assert result.recall == pytest.approx(2 / 3)
    assert result.f1 == pytest.approx(2 / 3)


def test_localisation_is_scored_only_on_detected_faults() -> None:
    trajectories = [
        _trajectory("a", faulty=True, failure_type=FailureType.WRONG_TOOL, failure_step=3),
        _trajectory("b", faulty=True, failure_type=FailureType.WRONG_TOOL, failure_step=3),
        _trajectory("c", faulty=True, failure_type=FailureType.WRONG_TOOL, failure_step=3),
    ]
    verdicts = [
        _verdict("a", faulty=True, failure_type=FailureType.WRONG_TOOL, failure_step=3),
        _verdict("b", faulty=True, failure_type=FailureType.WRONG_TOOL, failure_step=4),
        _verdict("c", faulty=False),  # missed, so it never reaches the localisation score
    ]
    result = score("j", trajectories, verdicts)
    assert result.step_scored_n == 2
    assert result.step_exact == pytest.approx(0.5)
    assert result.step_within_one == pytest.approx(1.0)


def test_a_missed_fault_lands_in_the_missed_column() -> None:
    trajectories = [
        _trajectory("a", faulty=True, failure_type=FailureType.UNSUPPORTED_CLAIM),
        _trajectory("b", faulty=True, failure_type=FailureType.UNSUPPORTED_CLAIM),
    ]
    verdicts = [
        _verdict("a", faulty=False),
        _verdict("b", faulty=True, failure_type=FailureType.WRONG_TOOL),
    ]
    matrix = type_confusion(trajectories, verdicts)
    assert matrix["unsupported_claim"][MISSED] == 1
    assert matrix["unsupported_claim"]["wrong_tool"] == 1


def test_clean_trajectories_are_excluded_from_type_scoring() -> None:
    """A false alarm is charged to precision, not a second time to the confusion matrix."""
    trajectories = [_trajectory("clean")]
    verdicts = [_verdict("clean", faulty=True, failure_type=FailureType.WRONG_TOOL)]
    matrix = type_confusion(trajectories, verdicts)
    assert all(sum(row.values()) == 0 for row in matrix.values())


def test_a_perfect_judge_scores_one_across_the_board() -> None:
    trajectories = [
        _trajectory(f, faulty=True, failure_type=t, failure_step=1)
        for f, t in [(t.value, t) for t in FailureType]
    ] + [_trajectory("clean")]
    verdicts = [
        _verdict(t.value, faulty=True, failure_type=t, failure_step=1, confidence=1.0)
        for t in FailureType
    ] + [_verdict("clean", faulty=False, confidence=1.0)]
    result = score("j", trajectories, verdicts)
    assert result.f1 == pytest.approx(1.0)
    assert result.type_macro_f1 == pytest.approx(1.0)
    assert result.step_exact == pytest.approx(1.0)
    assert result.ece == pytest.approx(0.0)
    assert result.brier == pytest.approx(0.0)


def test_brier_matches_hand_calculation() -> None:
    trajectories = [_trajectory("a", faulty=True), _trajectory("b")]
    verdicts = [
        _verdict("a", faulty=True, confidence=0.8),  # right, (0.8 - 1)^2 = 0.04
        _verdict("b", faulty=True, confidence=0.6),  # wrong, (0.6 - 0)^2 = 0.36
    ]
    assert score("j", trajectories, verdicts).brier == pytest.approx(0.20)


def test_scoring_an_empty_run_does_not_explode() -> None:
    assert score("j", [], []).n == 0
