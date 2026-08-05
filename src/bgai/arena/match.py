"""Match specs, the agent registry, seat rotation, and single-match runs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from bgai.agents.base import Agent
from bgai.agents.heuristic import HeuristicAgent
from bgai.agents.random_agent import RandomAgent
from bgai.arena.driver import GameResult, run_game
from bgai.arena.setup_factory import fresh_setup


@dataclass(frozen=True)
class MatchSpec:
    seed: int
    agent_names: tuple[str, str, str, str]  # seat i (setup.factions[i]) gets agent i


@dataclass(frozen=True)
class MatchOutcome:
    spec: MatchSpec
    result: GameResult
    agent_by_faction: Mapping[str, str]


def build_agent(name: str, seed: int) -> Agent:
    if name.startswith("random"):
        return RandomAgent(seed=seed, name=name)
    if name.startswith("heuristic"):
        return HeuristicAgent(seed=seed, name=name)
    raise ValueError(f"unknown agent {name!r} (registry: random*, heuristic*)")


def rotation(agent_names: tuple[str, ...], games: int, base_seed: int) -> tuple[MatchSpec, ...]:
    """Game g uses seed base_seed + g and the seat assignment rotated by
    g % 4, so each agent name occupies each seat equally over multiples
    of 4 games.
    """
    if len(agent_names) != 4:
        raise ValueError(f"need exactly 4 agent names, got {len(agent_names)}")
    specs = []
    for g in range(games):
        r = g % 4
        rotated = tuple(agent_names[(i + r) % 4] for i in range(4))
        specs.append(MatchSpec(seed=base_seed + g, agent_names=rotated))
    return tuple(specs)


def run_match(spec: MatchSpec) -> MatchOutcome:
    setup = fresh_setup(seed=spec.seed)
    agent_by_faction = dict(zip(setup.factions, spec.agent_names, strict=True))
    agents = {
        faction: build_agent(name, seed=spec.seed * 31 + seat)
        for seat, (faction, name) in enumerate(agent_by_faction.items())
    }
    result = run_game(setup, agents)
    return MatchOutcome(spec=spec, result=result, agent_by_faction=agent_by_faction)
