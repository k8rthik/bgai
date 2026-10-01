"""PettingZoo AEC environment for 2-5 player Terra Mystica.

AEC rather than Parallel because Terra Mystica is strictly turn-based and
a seat can face several consecutive decisions inside one turn (one fresh
main action, then its continuations, then ``done``) while *other* seats
interleave forced answers (a leech offer is answered when its holder is
asked, not when its turn comes -- ``state.active_faction``'s docstring has
the corpus citation). An AEC env expresses that natively: ``agent_selection``
is simply whoever ``driver.decision`` says is next. A Parallel env would
have to invent no-op actions for every waiting seat.

Agents are named ``player_0..player_{n-1}`` in seat order (PettingZoo's
recommended id shape); the faction behind a seat is in
``env.factions``/``env.faction_of`` and in each ``info["faction"]``.
"""

from __future__ import annotations

import functools
from typing import Any

import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv

from bgai.env.config import AGENT_NAME_TEMPLATE, EnvConfig, EnvStateError
from bgai.env.core import Episode
from bgai.env.serialize import obs_to_dict
from bgai.env.spaces import action_space, observation_space

__all__ = ["TerraMysticaAECEnv", "env", "raw_env"]

METADATA: dict[str, Any] = {
    "name": "terra_mystica_v0",
    "render_modes": ["ansi"],
    "is_parallelizable": False,
}


class TerraMysticaAECEnv(AECEnv):
    """AEC wrapper over :class:`bgai.env.core.Episode`.

    The rules, the legal mask and the reward all come from ``Episode``,
    which comes from the engine. Nothing in this class knows a Terra
    Mystica rule.
    """

    metadata = METADATA

    def __init__(self, config: EnvConfig | None = None, render_mode: str | None = None) -> None:
        super().__init__()
        if render_mode is not None and render_mode not in METADATA["render_modes"]:
            raise ValueError(
                f"render_mode {render_mode!r} not in {METADATA['render_modes']}"
            )
        self.config = config or EnvConfig()
        self.render_mode = render_mode
        self.possible_agents: list[str] = [
            AGENT_NAME_TEMPLATE.format(index=i) for i in range(self.config.player_count)
        ]
        self.agents: list[str] = []
        self._episode: Episode | None = None
        self._factions: tuple[str, ...] = ()
        self._obs_space = observation_space(self.config.max_candidates)
        self._act_space = action_space(self.config.max_candidates)

    # -- spaces ------------------------------------------------------------
    # api_test requires the *same object* back every call, so these are
    # cached; every seat shares one space (the observation is
    # mover-relative, so a seat index carries no information).

    @functools.cache  # noqa: B019 -- bounded by possible_agents
    def observation_space(self, agent: str) -> spaces.Dict:
        self._check_agent(agent)
        return self._obs_space

    @functools.cache  # noqa: B019 -- bounded by possible_agents
    def action_space(self, agent: str) -> spaces.Discrete:
        self._check_agent(agent)
        return self._act_space

    def _check_agent(self, agent: str) -> None:
        if agent not in self.possible_agents:
            raise KeyError(f"{agent!r} is not one of {self.possible_agents}")

    # -- seat identity -----------------------------------------------------

    @property
    def factions(self) -> tuple[str, ...]:
        """Faction per agent, in ``possible_agents`` order."""
        return self._factions

    def faction_of(self, agent: str) -> str:
        self._check_agent(agent)
        if not self._factions:
            raise EnvStateError("reset() the env before asking which faction a seat holds")
        return self._factions[self.possible_agents.index(agent)]

    def agent_of(self, faction: str) -> str:
        if faction not in self._factions:
            raise KeyError(f"{faction!r} is not at this table {self._factions}")
        return self.possible_agents[self._factions.index(faction)]

    @property
    def episode(self) -> Episode:
        """The live immutable episode. Raises before the first ``reset``."""
        if self._episode is None:
            raise EnvStateError("env has not been reset()")
        return self._episode

    # -- lifecycle ---------------------------------------------------------

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> None:
        """Start a fresh table.

        ``seed`` is the table seed: it fixes the faction lineup, the score
        tiles and the bonus pool, so the same seed plus the same action
        sequence reproduces a trajectory exactly. ``seed=None`` draws one
        from the action space's own RNG so repeated resets differ.
        """
        if seed is None:
            seed = int(self._act_space.np_random.integers(0, 2**31 - 1))
        self._act_space.seed(seed)
        self._obs_space.seed(seed)
        self._episode = Episode.start(seed, self.config)
        self._factions = self._episode.seats
        self.agents = list(self.possible_agents)
        self.rewards = dict.fromkeys(self.agents, 0.0)
        self._cumulative_rewards = dict.fromkeys(self.agents, 0.0)
        self.terminations = dict.fromkeys(self.agents, False)
        self.truncations = dict.fromkeys(self.agents, False)
        self.infos = {agent: self._info(agent) for agent in self.agents}
        self.agent_selection = self._next_agent()

    def _next_agent(self) -> str:
        acting = self.episode.acting_seat()
        if acting is None:
            # Game over: whoever is still listed drains via dead steps.
            return self.agents[0] if self.agents else self.possible_agents[0]
        return self.agent_of(acting)

    def _info(self, agent: str) -> dict[str, Any]:
        episode = self.episode
        faction = self.faction_of(agent)
        info: dict[str, Any] = {
            "faction": faction,
            "seed": episode.seed,
            "game_id": episode.setup.game_id,
            "decisions": episode.decisions,
            "vp": episode.game.factions[faction].vp,
            "round": episode.game.round,
            "phase": episode.game.phase.name,
        }
        if episode.engine_error is not None:
            info["engine_error"] = episode.engine_error
        if episode.illegal_action is not None:
            info["illegal_action"] = episode.illegal_action
        return info

    # -- observation -------------------------------------------------------

    def observe(self, agent: str) -> dict[str, Any]:
        """``agent``'s own mover-relative view.

        The ``action_mask`` is all zeros for a seat that is not the one to
        decide, and once the game is over -- PettingZoo's convention for
        "not your move" (the same shape ``connect_four_v3`` uses).
        """
        self._check_agent(agent)
        return obs_to_dict(self.episode.observe(self.faction_of(agent)))

    # -- stepping ----------------------------------------------------------

    def step(self, action: Any) -> None:
        agent = self.agent_selection
        if self.terminations.get(agent) or self.truncations.get(agent):
            self._was_dead_step(action)
            return
        if action is None:
            raise ValueError(
                f"{agent!r} is live, so its action must be an index, not None"
            )

        episode, transition = self.episode.step(int(action))
        self._episode = episode

        self._cumulative_rewards[agent] = 0.0
        self.rewards = {
            a: float(transition.rewards[self.faction_of(a)]) for a in self.agents
        }
        if transition.done:
            self.terminations = dict.fromkeys(self.agents, transition.terminated)
            self.truncations = dict.fromkeys(self.agents, transition.truncated)
        self.infos = {a: self._info(a) for a in self.agents}
        self.agent_selection = self._next_agent()
        self._accumulate_rewards()

    # -- render ------------------------------------------------------------

    def render(self) -> str | None:
        """A one-line ANSI scoreboard. The engine's own renderer
        (``bgai.mcp.render``) is the full board view; this is deliberately
        just enough to watch a run.
        """
        if self.render_mode != "ansi":
            return None
        episode = self.episode
        vps = episode.vps()
        acting = episode.acting_seat() or "-"
        line = (
            f"r{episode.game.round} {episode.game.phase.name} "
            f"d{episode.decisions} act={acting} | "
            + " ".join(f"{f}:{vps[f]}" for f in episode.seats)
        )
        print(line)
        return line

    def close(self) -> None:
        return None

    # -- state access ------------------------------------------------------

    def state(self) -> np.ndarray:
        """The global (seat-0-relative) ``globals`` vector.

        AEC's optional ``state()`` hook, for centralized critics.
        """
        return self.episode.observe(self.episode.seats[0]).globals

    def snapshot(self) -> bytes:
        """Serialize the whole table; :meth:`restore` reverses it."""
        return self.episode.to_bytes()

    def restore(self, payload: bytes) -> None:
        """Replace the live episode with a serialized one.

        Agent bookkeeping is rebuilt from it, so a restored env is
        steppable exactly like a freshly reset one.
        """
        episode = Episode.from_bytes(payload)
        if episode.config.player_count != self.config.player_count:
            raise ValueError(
                f"snapshot is a {episode.config.player_count}-player table but "
                f"this env is configured for {self.config.player_count}"
            )
        self._episode = episode
        self.config = episode.config
        self._factions = episode.seats
        self.agents = list(self.possible_agents)
        self.rewards = dict.fromkeys(self.agents, 0.0)
        self._cumulative_rewards = dict.fromkeys(self.agents, 0.0)
        terminal = episode.sim.finished or episode.illegal_action is not None
        self.terminations = dict.fromkeys(self.agents, terminal)
        self.truncations = dict.fromkeys(
            self.agents, episode.finished and not terminal
        )
        self.infos = {agent: self._info(agent) for agent in self.agents}
        self.agent_selection = self._next_agent()


def raw_env(
    config: EnvConfig | None = None, render_mode: str | None = None
) -> TerraMysticaAECEnv:
    """The unwrapped AEC env (PettingZoo's ``raw_env`` convention)."""
    return TerraMysticaAECEnv(config=config, render_mode=render_mode)


def env(
    config: EnvConfig | None = None, render_mode: str | None = None
) -> AECEnv:
    """The AEC env behind PettingZoo's standard assertion wrappers.

    ``TerminateIllegalWrapper`` is deliberately NOT applied: it ends the
    game and penalizes a seat that plays a masked action, which hides
    exactly the bug an illegal action indicates. TM-Env raises
    :class:`~bgai.env.config.IllegalActionError` instead.
    """
    from pettingzoo.utils import AssertOutOfBoundsWrapper, OrderEnforcingWrapper

    base = raw_env(config=config, render_mode=render_mode)
    return OrderEnforcingWrapper(AssertOutOfBoundsWrapper(base))
