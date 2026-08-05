"""Terra Mystica agents: the Agent protocol and its implementations."""

from __future__ import annotations

from bgai.agents.base import Agent
from bgai.agents.greedy import GreedyAgent
from bgai.agents.random_agent import RandomAgent

__all__ = ["Agent", "GreedyAgent", "RandomAgent"]
