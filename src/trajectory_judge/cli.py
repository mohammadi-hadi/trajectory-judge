"""Command line: build a balanced trajectory set, judge it, and rebuild the report.

The trajectory set is stratified rather than sampled. Failure types are not equally easy to
host — ``ignored_observation`` needs an order with a restocking fee, which is one instance in
six — so the generator is asked for enough instances that the rarest type still fills its
quota. A run that quietly returned nine of one fault and ninety of another would produce a
confusion matrix that says more about the generator than about any judge.
"""

from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Any

import typer

from trajectory_judge import store
from trajectory_judge.agents.oracle import run_oracle
from trajectory_judge.env.world import Instance, generate_instances
from trajectory_judge.judges import (
    Judge,
    LlmJudge,
    MockJudge,
    OutcomeJudge,
    ProgrammaticJudge,
    SelfConsistencyJudge,
    StepRubricJudge,
)
from trajectory_judge.judges.ablation import ABLATION_JUDGES
from trajectory_judge.judges.llm import UNPARSEABLE
from trajectory_judge.judges.ollama_client import Generation
from trajectory_judge.mutate import mutate
from trajectory_judge.trace import FailureType, Trajectory, Verdict

app = typer.Typer(add_completion=False, help=__doc__)

DEFAULT_JUDGES = "programmatic,outcome,step"

#: Failed calls in a row that end a run. One is a hiccup; three is a server that has gone away.
MAX_CONSECUTIVE_FAILURES = 3


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

    # Shuffle, deterministically, so that *any prefix is a stratified sample*. Built in order the
    # set is 100 clean then 50 of each type in turn, which makes `trajectories[:150]` all-clean
    # plus one failure type — a subset judge scored on it would report a loud recall of zero
    # because it never saw a loud fault. That is a bug that looks like a finding.
    random.Random(f"order-{seed}").shuffle(trajectories)
    return trajectories, by_id


def _call_failed(verdict: Verdict) -> bool:
    """Whether the model never answered, as opposed to answering with something unusable.

    Such a verdict is not stored. It says "clean" at chance only because nothing came back, and
    once on disk the resume logic would treat the trajectory as judged: a server that died at 2am
    would leave the rest of the night's queue recorded as clean verdicts.
    """
    return verdict.error is not None and verdict.error != UNPARSEABLE


def _response_row(judge: LlmJudge, trajectory: Trajectory, response: Generation) -> dict[str, Any]:
    """The raw response, keyed like its verdict, with a hash of the exact prompt it answered."""
    prompt = judge.prompt(trajectory).encode("utf-8")
    return {
        "trajectory_id": trajectory.trajectory_id,
        "judge_id": judge.judge_id,
        "prompt_sha256": hashlib.sha256(prompt).hexdigest(),
        "text": response.text,
        "prompt_tokens": response.prompt_tokens,
        "completion_tokens": response.completion_tokens,
        "latency_s": response.latency_s,
        "error": response.error,
    }


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
    if name in ABLATION_JUDGES:
        return ABLATION_JUDGES[name](model, seed=seed)
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
    keep_responses: bool = typer.Option(
        False, help="Also append each LLM judge's raw response to responses.jsonl."
    ),
) -> None:
    """Judge a freshly built trajectory set. Resumable: already-judged pairs are skipped.

    A call that fails outright is reported and left unjudged, so the exit code is non-zero and a
    rerun fills the gap. Three failures in a row stop the run.
    """
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
    unjudged = 0
    for name in [j.strip() for j in judges.split(",") if j.strip()]:
        judge = _make_judge(name, model, k, seed)
        targets = trajectories[:selfcons_subset] if name == "selfcons" else trajectories
        pending = [t for t in targets if (t.trajectory_id, judge.judge_id) not in done]
        typer.echo(
            f"{judge.judge_id}: {len(pending)} to judge ({len(targets) - len(pending)} cached)"
        )
        streak = 0
        for index, trajectory in enumerate(pending, start=1):
            instance = instances[trajectory.instance_id]
            if keep_responses and isinstance(judge, LlmJudge):
                verdict, response = judge.judge_with_response(trajectory, instance)
                if not _call_failed(verdict):
                    store.append_response(out, _response_row(judge, trajectory, response))
            else:
                verdict = judge.judge(trajectory, instance)
            if verdict.error is None:
                streak = 0
            else:
                streak += 1
                typer.echo(f"  {trajectory.trajectory_id}: {verdict.error}", err=True)
            if _call_failed(verdict):
                unjudged += 1
            else:
                store.append_verdict(out, verdict)
            if streak >= MAX_CONSECUTIVE_FAILURES:
                typer.echo(
                    f"stopping: {streak} failed calls in a row from {judge.judge_id}", err=True
                )
                raise typer.Exit(2)
            if index % 25 == 0 or index == len(pending):
                typer.echo(f"  {index}/{len(pending)}")
    if unjudged:
        typer.echo(f"{unjudged} calls failed and were not stored; rerun to judge them", err=True)
        raise typer.Exit(3)


@app.command()
def agent(
    n: int = typer.Option(60, help="Episodes to play."),
    model: str = typer.Option("qwen2.5:14b", help="Ollama model driving the agent."),
    seed: int = typer.Option(7),
    out: Path = typer.Option(Path("results/agent"), help="Where agent episodes are appended."),
) -> None:
    """Let a model play the environment, to check the injected faults resemble real ones."""
    from trajectory_judge.agents.llm_agent import run_llm_agent

    done = {t.trajectory_id for t in store.read_trajectories(out)}
    instances = [
        i for i in generate_instances(n, seed=seed) if f"{i.instance_id}-agent" not in done
    ]
    typer.echo(f"{len(instances)} episodes to play ({len(done)} cached)")
    for index, instance in enumerate(instances, start=1):
        store.append_trajectory(out, run_llm_agent(instance, model, seed=seed))
        if index % 10 == 0 or index == len(instances):
            typer.echo(f"  {index}/{len(instances)}")


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


@app.command()
def serve() -> None:
    """Serve the judges over HTTP. Needs the 'serve' extra.

    uvicorn is imported inside the body, the same way `report` imports matplotlib, so
    `trajectory-judge --help` keeps working on an install without the extra.
    """
    try:
        from trajectory_judge.serve.app import run
    except ImportError:
        typer.echo("serve needs the 'serve' extra: pip install 'trajectory-judge[serve]'", err=True)
        raise typer.Exit(1) from None

    run()


if __name__ == "__main__":
    app()
