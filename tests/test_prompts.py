"""The published judge prompts, pinned byte for byte.

Every committed verdict was produced from these exact prompt strings. A refactor that moves a
line break or a space would still pass every behavioural test while silently changing what the
model reads, so the full set of 400 prompts is hashed against the values the v0.1.0 verdicts
were generated with.

The hash is ``sha256`` over the prompts in dataset order, each followed by a NUL byte.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable

import pytest

from trajectory_judge.cli import build_dataset
from trajectory_judge.judges.llm import OutcomeJudge, StepRubricJudge
from trajectory_judge.trace import Trajectory

PUBLISHED_OUTCOME = "bdd0a3e1df58fa1b33503a46d1e0c8ce0c7d1ecbb323b88929d73d08f56c3e49"
PUBLISHED_STEP = "6dce19cae7ab2917aef783e30905d129d0da03eb8c3a5d22a1a9898eef658b82"


@pytest.fixture(scope="module")
def published_set() -> list[Trajectory]:
    """The 400 trajectories behind the committed verdicts, rebuilt in their committed order."""
    trajectories, _ = build_dataset(400, 7)
    return trajectories


def prompt_hash(prompt: Callable[[Trajectory], str], trajectories: list[Trajectory]) -> str:
    joined = "".join(prompt(t) + "\0" for t in trajectories)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def test_the_outcome_prompt_is_the_one_the_verdicts_were_made_with(
    published_set: list[Trajectory],
) -> None:
    assert prompt_hash(OutcomeJudge("m").prompt, published_set) == PUBLISHED_OUTCOME


def test_the_step_prompt_is_the_one_the_verdicts_were_made_with(
    published_set: list[Trajectory],
) -> None:
    assert prompt_hash(StepRubricJudge("m").prompt, published_set) == PUBLISHED_STEP
