"""Terra Mystica agents: the Agent protocol and its implementations."""

from __future__ import annotations

from bgai.agents.base import Agent
from bgai.agents.greedy import GreedyAgent
from bgai.agents.random_agent import RandomAgent

# ImitationAgent is deliberately NOT re-exported here: it imports torch,
# and the arena/engine must stay usable (and fast to import) without a
# deep-learning dependency. Import it directly from bgai.agents.imitation.
__all__ = ["Agent", "GreedyAgent", "RandomAgent"]
