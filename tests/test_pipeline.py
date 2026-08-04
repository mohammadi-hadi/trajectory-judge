"""End-to-end coverage of the parts that must work without a model: storage, resume, report.

Everything here runs on the mock judge, so the suite stays green on a machine with no Ollama
and in CI with no network.
"""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from trajectory_judge import store
from trajectory_judge.cli import app, build_dataset
from trajectory_judge.report import build
from trajectory_judge.trace import FailureType, Label, Trajectory, Verdict

runner = CliRunner()


def test_dataset_is_balanced_and_deterministic() -> None:
    trajectories, instances = build_dataset(60, seed=7)
    again, _ = build_dataset(60, seed=7)
    assert [t.trajectory_id for t in trajectories] == [t.trajectory_id for t in again]

    counts = {f: 0 for f in FailureType}
    clean = 0
    for trajectory in trajectories:
        if trajectory.label.failure_type is None:
            clean += 1
        else:
            counts[trajectory.label.failure_type] += 1
    assert clean == 15  # a quarter of 60
    assert set(counts.values()) == {7}  # (60 - 15) // 6
    assert all(t.instance_id in instances for t in trajectories)


def test_any_prefix_of_the_dataset_is_a_stratified_sample() -> None:
    """Subset judges score a prefix, so a prefix must not be all-clean or one failure type.

    Built in order the set is clean-block-then-type-blocks, and a 150-trajectory prefix of 400
    would contain no loud faults at all — a subset judge would then report a loud recall of
    zero that says nothing about the judge.
    """
    trajectories, _ = build_dataset(400, seed=7)
    prefix = trajectories[:150]
    assert sum(1 for t in prefix if not t.label.faulty) > 10
    assert sum(1 for t in prefix if t.label.silent) > 10
    assert sum(1 for t in prefix if t.label.faulty and not t.label.outcome_correct) > 10
    assert len({t.label.failure_type for t in prefix if t.label.failure_type}) == len(FailureType)


def test_every_trajectory_id_is_unique() -> None:
    trajectories, _ = build_dataset(120, seed=7)
    ids = [t.trajectory_id for t in trajectories]
    assert len(ids) == len(set(ids))


def test_store_round_trips(tmp_path: Path) -> None:
    trajectory = Trajectory(trajectory_id="a", instance_id="i", goal="g", label=Label(faulty=True))
    store.write_trajectories(tmp_path, [trajectory])
    assert store.read_trajectories(tmp_path) == [trajectory]

    verdict = Verdict(trajectory_id="a", judge_id="j", faulty=True)
    store.append_verdict(tmp_path, verdict)
    store.append_verdict(tmp_path, Verdict(trajectory_id="a", judge_id="k", faulty=False))
    assert store.read_verdicts(tmp_path)[0] == verdict
    assert store.judged_keys(tmp_path) == {("a", "j"), ("a", "k")}


def test_reading_an_empty_directory_is_not_an_error(tmp_path: Path) -> None:
    assert store.read_trajectories(tmp_path) == []
    assert store.judged_keys(tmp_path) == set()
    assert store.read_run_meta(tmp_path) == {}


def test_run_then_report_produces_tables(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    result = runner.invoke(
        app, ["run", "--n", "24", "--judges", "mock,programmatic", "--out", str(raw)]
    )
    assert result.exit_code == 0, result.output

    written = build(raw, tmp_path / "out")
    assert len(written["tables"]) == 3
    summary = (tmp_path / "out" / "tables" / "summary.md").read_text()
    assert "programmatic" in summary and "mock" in summary
    # The rule engine leads the table regardless of the order judges ran in.
    assert summary.index("`programmatic`") < summary.index("`mock`")


def test_report_is_deterministic(tmp_path: Path) -> None:
    """Tables must be byte-identical on a rebuild, or CI cannot diff them against the repo."""
    raw = tmp_path / "raw"
    runner.invoke(app, ["run", "--n", "24", "--judges", "mock", "--out", str(raw)])
    build(raw, tmp_path / "first")
    build(raw, tmp_path / "second")
    for name in ("summary.md", "per_type_recall.md", "confusion.md"):
        assert (tmp_path / "first" / "tables" / name).read_bytes() == (
            tmp_path / "second" / "tables" / name
        ).read_bytes()


def test_a_second_run_judges_nothing_new(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    args = ["run", "--n", "24", "--judges", "mock", "--out", str(raw)]
    runner.invoke(app, args)
    first = len(store.read_verdicts(raw))
    second_run = runner.invoke(app, args)
    assert second_run.exit_code == 0
    assert len(store.read_verdicts(raw)) == first
    assert "0 to judge" in second_run.output


def test_the_mock_judge_is_reproducible(tmp_path: Path) -> None:
    verdicts = []
    for name in ("a", "b"):
        raw = tmp_path / name
        runner.invoke(app, ["run", "--n", "24", "--judges", "mock", "--out", str(raw)])
        verdicts.append(
            [(v.trajectory_id, v.faulty, v.confidence) for v in store.read_verdicts(raw)]
        )
    assert verdicts[0] == verdicts[1]
