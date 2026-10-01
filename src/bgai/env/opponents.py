"""Opponent policies for the single-agent wrapper.

A policy is anything that picks an index into the offer. The existing
``bgai.agents`` family already picks a ``ParsedCommand`` from the offer, so
wrapping one is just "find the index it returned" -- the env never has to
know how the agent decided.

Checkpoint-backed policies (imitation, mcts) are constructed lazily and
imported lazily: ``bgai.agents.imitation`` pulls in torch, and
``bgai.env`` must stay importable without it on the training-free path.

Player-count note: the imitation net and the MCTS agent both encode state
with ``training.encode_state``, which is pinned at four seats, and MCTS's
value vectors are 4-wide. They are therefore 4-player only, and
:func:`build_opponent` says so rather than producing wrong numbers.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from bgai.arena.driver import SimState
from bgai.data.ledger_parser import ParsedCommand
from bgai.env.config import EnvError

__all__ = [
    "AgentPolicy",
    "OpponentSpec",
    "Policy",
    "UnknownOpponentError",
    "build_opponent",
    "known_opponents",
    "random_policy",
]


class UnknownOpponentError(EnvError, ValueError):
    """An opponent name this module does not know how to build."""


class Policy(Protocol):
    """Pick an index into a canonically ordered offer.

    ``sim`` is the driver's whole paused state, because a search agent has
    to branch from it (``arena.sim.run_game`` passes exactly this through
    its optional ``choose_sim`` extension). A policy that only needs the
    board reads ``sim.game``.
    """

    name: str

    def __call__(
        self,
        sim: SimState,
        faction: str,
        offer: tuple[ParsedCommand, ...],
        rng: random.Random,
    ) -> int: ...

    def begin_episode(self) -> None:
        """Called once per episode, before the first decision."""

    def observe_move(self, move: ParsedCommand) -> None:
        """Called after every applied move, any seat's."""


class _BasePolicy:
    """Default no-op episode hooks, so every policy satisfies ``Policy``."""

    name = "policy"

    def begin_episode(self) -> None:
        return None

    def observe_move(self, move: ParsedCommand) -> None:
        return None


class _NamedPolicy(_BasePolicy):
    def __init__(self, name: str, fn: Callable[..., int]) -> None:
        self.name = name
        self._fn = fn

    def __call__(
        self,
        sim: SimState,
        faction: str,
        offer: tuple[ParsedCommand, ...],
        rng: random.Random,
    ) -> int:
        return self._fn(sim, faction, offer, rng)


def random_policy(name: str = "random") -> Policy:
    """Uniform over the offer -- the floor of the rating scale."""

    def pick(
        sim: SimState,
        faction: str,
        offer: tuple[ParsedCommand, ...],
        rng: random.Random,
    ) -> int:
        return rng.randrange(len(offer))

    return _NamedPolicy(name, pick)


class AgentPolicy(_BasePolicy):
    """Adapt a ``bgai.agents`` ``Agent`` to the index-returning protocol.

    Routing matches ``arena.sim.run_game`` exactly: an agent exposing
    ``choose_sim`` gets the whole ``SimState`` (MCTS must branch), anything
    else gets ``sim.game``. ``reset_tree``/``note_advance`` are forwarded
    when present, which is what keeps MCTS's tree reuse -- and so its
    measured strength -- intact through the wrapper.

    The offer the agent sees is the same tuple the env built its mask
    from, so "which index did it return" is an exact lookup by identity
    first and equality second -- no re-derivation of legality.
    """

    def __init__(self, agent: object, name: str | None = None) -> None:
        self._choose_sim = getattr(agent, "choose_sim", None)
        if self._choose_sim is None and not hasattr(agent, "choose"):
            raise TypeError(f"{agent!r} implements neither choose nor choose_sim")
        self.agent = agent
        self.name = name or getattr(agent, "name", agent.__class__.__name__)

    def begin_episode(self) -> None:
        reset = getattr(self.agent, "reset_tree", None)
        if reset is not None:
            reset()  # games are independent; never reuse a tree across them

    def observe_move(self, move: ParsedCommand) -> None:
        note = getattr(self.agent, "note_advance", None)
        if note is not None:
            note(move)

    def __call__(
        self,
        sim: SimState,
        faction: str,
        offer: tuple[ParsedCommand, ...],
        rng: random.Random,
    ) -> int:
        if self._choose_sim is not None:
            choice = self._choose_sim(sim, faction, offer, rng)
        else:
            choice = self.agent.choose(sim.game, faction, offer, rng)  # type: ignore[attr-defined]
        for index, move in enumerate(offer):
            if move is choice:
                return index
        for index, move in enumerate(offer):
            if move == choice:
                return index
        raise EnvError(
            f"opponent {self.name!r} returned a move outside its offer: {choice!r}"
        )


_CHECKPOINT_OPPONENTS = ("imitation", "mcts")
_SIMPLE_OPPONENTS = ("random", "greedy")


def known_opponents() -> tuple[str, ...]:
    return _SIMPLE_OPPONENTS + _CHECKPOINT_OPPONENTS


OpponentSpec = str
"""An opponent name: ``random``, ``greedy``, ``imitation`` or ``mcts``."""


def build_opponent(
    spec: OpponentSpec,
    *,
    checkpoint: Path | None = None,
    player_count: int = 4,
    **kwargs: object,
) -> Policy:
    """Construct an opponent policy by name.

    ``imitation``/``mcts`` need ``checkpoint`` (a ``data/checkpoints/<run>/
    current.pt``) and are 4-player only; ``kwargs`` is forwarded to the
    underlying agent (``simulations``, ``top_k``, ``c_puct``, ...).
    """
    if spec == "random":
        return random_policy()
    if spec == "greedy":
        from bgai.agents.greedy import GreedyAgent

        return AgentPolicy(GreedyAgent(), name="greedy")
    if spec in _CHECKPOINT_OPPONENTS:
        if checkpoint is None:
            raise UnknownOpponentError(
                f"opponent {spec!r} needs a checkpoint path "
                "(e.g. data/checkpoints/selfplay_leg5b/current.pt)"
            )
        if player_count != 4:
            raise UnknownOpponentError(
                f"opponent {spec!r} is 4-player only: training.encode_state and "
                "the max^n value vector are both pinned at four seats, so a "
                f"{player_count}-player table cannot use it. Use 'random' or "
                "'greedy' off 4 players."
            )
        if spec == "imitation":
            from bgai.agents.imitation import ImitationAgent

            return AgentPolicy(
                ImitationAgent(checkpoint_path=Path(checkpoint), **kwargs),  # type: ignore[arg-type]
                name="imitation",
            )
        from bgai.agents.mcts import MCTSAgent

        return AgentPolicy(
            MCTSAgent(checkpoint_path=Path(checkpoint), **kwargs),  # type: ignore[arg-type]
            name="mcts",
        )
    raise UnknownOpponentError(
        f"unknown opponent {spec!r}; known: {known_opponents()}"
    )
