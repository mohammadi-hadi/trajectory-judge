"""Policies that produce trajectories: the scripted oracle and a tool-calling LLM agent."""

from trajectory_judge.agents.oracle import run_oracle

__all__ = ["run_oracle"]
