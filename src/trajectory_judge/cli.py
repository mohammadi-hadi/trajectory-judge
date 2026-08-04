"""Command line: build a balanced trajectory set, judge it, and rebuild the report.

The trajectory set is stratified rather than sampled. Failure types are not equally easy to
host — ``ignored_observation`` needs an order with a restocking fee, which is one instance in
six — so the generator is asked for enough instances that the rarest type still fills its
quota. A run that quietly returned nine of one fault and ninety of another would produce a
confusion matrix that says more about the generator than about any judge.
"""

from __future__ import annotations

from pathlib import Path

import typer

from trajectory_judge import store
from trajectory_judge.agents.oracle import run_oracle
from trajectory_judge.env.world import Instance, generate_instances
from trajectory_judge.judges import (
    Judge,
    MockJudge,
    OutcomeJudge,
    ProgrammaticJudge,
    SelfConsistencyJudge,
    StepRubricJudge,
)
from trajectory_judge.mutate import mutate
from trajectory_judge.trace import FailureType, Trajectory

app = typer.Typer(add_completion=False, help=__doc__)

DEFAULT_JUDGES = "programmatic,outcome,step"


def build_dataset(n: int, seed: int) -> tuple[list[Trajectory], dict[str, Instance]]:
    """A balanced set: a quarter clean, the rest split evenly across the six failure types."""
    clean_n = max(1, n // 4)
    per_type = max(1, (n - clean_n) // len(FailureType))
    # The rarest failure type is hosted by one instance in six, so ask for six times the quota.
    instances = generate_instances(6 * per_type + 6, seed=seed)
    by_id = {ins.instance_id: ins for ins in instances}
    clean = {ins.instance_id: run_oracle(ins) for ins in instances}

    trajectories: list[Trajectory] = [clean[ins.instance_id] for ins in instances[:clean_n]]
    for failure_type in FailureType:
        taken = 0
        for ins in instances:
            if taken >= per_type:
                break
            mutant = mutate(ins, clean[ins.instance_id], failure_type, seed=seed)
            if mutant is not None:
                trajectories.append(mutant)
                taken += 1
        if taken < per_type:
            typer.echo(
                f"  note: only {taken}/{per_type} instances could host {failure_type.value}",
                err=True,
            )
    return trajectories, by_id


def _make_judge(name: str, model: str, k: int, seed: int) -> Judge:
    if name == "mock":
        return MockJudge()
    if name == "programmatic":
        return ProgrammaticJudge()
    if name == "outcome":
        return OutcomeJudge(model, seed=seed)
    if name == "step":
        return StepRubricJudge(model, seed=seed)
    if name == "selfcons":
        return SelfConsistencyJudge(model, k=k, base_seed=seed)
    raise typer.BadParameter(f"unknown judge {name!r}")


@app.command()
def run(
    n: int = typer.Option(400, help="Total trajectories: a quarter clean, the rest faulty."),
    judges: str = typer.Option(DEFAULT_JUDGES, help="Comma-separated judge names."),
    model: str = typer.Option("qwen2.5:14b", help="Ollama model for the LLM judges."),
    seed: int = typer.Option(7),
    k: int = typer.Option(3, help="Samples per trajectory for the self-consistency judge."),
    selfcons_subset: int = typer.Option(
        150, help="Trajectories the self-consistency judge covers, since it costs k times more."
    ),
    out: Path = typer.Option(Path("results/raw"), help="Where raw verdicts are appended."),
) -> None:
    """Judge a freshly built trajectory set. Resumable: already-judged pairs are skipped."""
    trajectories, instances = build_dataset(n, seed)
    store.write_trajectories(out, trajectories)
    store.write_run_meta(
        out,
        {
            "n_requested": n,
            "n_trajectories": len(trajectories),
            "judges": judges,
            "model": model,
            "seed": seed,
            "k": k,
            "selfcons_subset": selfcons_subset,
        },
    )
    typer.echo(f"{len(trajectories)} trajectories -> {out}")

    done = store.judged_keys(out)
    for name in [j.strip() for j in judges.split(",") if j.strip()]:
        judge = _make_judge(name, model, k, seed)
        targets = trajectories[:selfcons_subset] if name == "selfcons" else trajectories
        pending = [t for t in targets if (t.trajectory_id, judge.judge_id) not in done]
        typer.echo(
            f"{judge.judge_id}: {len(pending)} to judge ({len(targets) - len(pending)} cached)"
        )
        for index, trajectory in enumerate(pending, start=1):
            verdict = judge.judge(trajectory, instances[trajectory.instance_id])
            store.append_verdict(out, verdict)
            if index % 25 == 0 or index == len(pending):
                typer.echo(f"  {index}/{len(pending)}")


@app.command()
def report(
    raw: Path = typer.Option(Path("results/raw"), "--raw", help="Directory holding raw verdicts."),
    out: Path = typer.Option(Path("results"), "--out", help="Where tables and figures go."),
) -> None:
    """Rebuild every table and figure from raw verdicts. No model calls, no network."""
    from trajectory_judge.report import build

    written = build(raw, out)
    for kind, paths in written.items():
        for path in paths:
            typer.echo(f"{kind}: {path}")
    if not written["figures"]:
        typer.echo("figures skipped: install the 'report' extra for matplotlib", err=True)


if __name__ == "__main__":
    app()
