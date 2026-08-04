"""JSONL persistence, and the resume logic that makes a long run survive being interrupted.

Verdicts are appended, never rewritten, and keyed by ``(trajectory_id, judge_id)``. A rerun
reads the keys already on disk and skips them, so an overnight run that dies at hour three
continues from hour three. This is also why the raw files are what gets committed: every table
and figure in the repo is rebuilt from them, offline, with no model involved.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from trajectory_judge.trace import Trajectory, Verdict

TRAJECTORIES = "trajectories.jsonl"
VERDICTS = "verdicts.jsonl"
RUN_META = "run.json"


def _read_lines(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_trajectories(directory: Path, trajectories: Iterable[Trajectory]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / TRAJECTORIES
    with path.open("w", encoding="utf-8") as handle:
        for trajectory in trajectories:
            handle.write(trajectory.model_dump_json() + "\n")
    return path


def append_trajectory(directory: Path, trajectory: Trajectory) -> None:
    """Append a single trajectory, so agent episodes resume the same way verdicts do."""
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / TRAJECTORIES).open("a", encoding="utf-8") as handle:
        handle.write(trajectory.model_dump_json() + "\n")


def read_trajectories(directory: Path) -> list[Trajectory]:
    return [Trajectory.model_validate(row) for row in _read_lines(directory / TRAJECTORIES)]


def append_verdict(directory: Path, verdict: Verdict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / VERDICTS).open("a", encoding="utf-8") as handle:
        handle.write(verdict.model_dump_json() + "\n")


def read_verdicts(directory: Path) -> list[Verdict]:
    return [Verdict.model_validate(row) for row in _read_lines(directory / VERDICTS)]


def judged_keys(directory: Path) -> set[tuple[str, str]]:
    """``(trajectory_id, judge_id)`` pairs already on disk — the resume set."""
    return {(v.trajectory_id, v.judge_id) for v in read_verdicts(directory)}


def write_run_meta(directory: Path, meta: dict[str, Any]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / RUN_META).write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")


def read_run_meta(directory: Path) -> dict[str, Any]:
    path = directory / RUN_META
    if not path.exists():
        return {}
    loaded: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return loaded
