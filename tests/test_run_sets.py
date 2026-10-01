"""The sets a run can judge besides the balanced benchmark: missing parents, stored episodes."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from trajectory_judge import store
from trajectory_judge.cli import app, build_dataset, missing_parents

runner = CliRunner()


def test_every_fault_in_the_benchmark_gets_its_clean_parent() -> None:
    trajectories, instances = build_dataset(400, seed=7)
    parents = missing_parents(trajectories, instances)
    assert len(parents) == 41
    assert (parents[0].trajectory_id, parents[-1].trajectory_id) == (
        "INS-00102-clean",
        "INS-00295-clean",
    )
    assert all(not p.label.faulty and p.label.outcome_correct for p in parents)

    present = {t.trajectory_id for t in trajectories + parents}
    hosts = {t.instance_id for t in trajectories if t.label.faulty}
    assert all(f"{host}-clean" in present for host in hosts)


def test_with_parents_adds_them_to_the_judged_set(tmp_path: Path) -> None:
    trajectories, instances = build_dataset(24, seed=7)
    expected = len(trajectories) + len(missing_parents(trajectories, instances))
    args = ["run", "--n", "24", "--judges", "mock", "--with-parents", "--out", str(tmp_path)]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert len(store.read_trajectories(tmp_path)) == expected
    assert len(store.read_verdicts(tmp_path)) == expected
    assert store.read_run_meta(tmp_path)["runs"][0]["with_parents"] is True


def test_first_judges_the_named_types_before_the_rest(tmp_path: Path) -> None:
    first = ["premature_stop", "unsupported_claim"]
    args = ["run", "--n", "48", "--judges", "mock", "--first", ",".join(first)]
    result = runner.invoke(app, [*args, "--out", str(tmp_path)])
    assert result.exit_code == 0, result.output
    labels = {t.trajectory_id: t.label for t in store.read_trajectories(tmp_path)}
    judged = [labels[v.trajectory_id].failure_type for v in store.read_verdicts(tmp_path)]
    leading = [t.value if t else None for t in judged[:12]]
    assert leading == ["premature_stop"] * 6 + ["unsupported_claim"] * 6


def test_an_unknown_type_in_first_is_refused(tmp_path: Path) -> None:
    result = runner.invoke(app, ["run", "--n", "8", "--first", "typo", "--out", str(tmp_path)])
    assert result.exit_code != 0
    assert store.read_verdicts(tmp_path) == []


def test_source_judges_a_stored_set_instead_of_building_one(tmp_path: Path) -> None:
    trajectories, _ = build_dataset(24, seed=7)
    episodes = tmp_path / "episodes"
    store.write_trajectories(episodes, trajectories[:5])
    out = tmp_path / "judged"
    args = ["run", "--judges", "programmatic", "--source", str(episodes), "--out", str(out)]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert {v.trajectory_id for v in store.read_verdicts(out)} == {
        t.trajectory_id for t in trajectories[:5]
    }
