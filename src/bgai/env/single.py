"""Gymnasium single-agent view of a Terra Mystica table.

One seat is the learner; the other 1-4 are filled by a policy from
``bgai.env.opponents``. ``step`` applies the learner's move, then runs
opponents until it is the learner's turn again or the game ends -- so one
Gymnasium step is one *learner decision*, not one engine decision, and the
reward it returns is everything the learner earned in between (including
VP it leeched from a neighbour's build, under ``DENSE_VP``).

The learner's seat is fixed by ``learner_seat`` and rotated by
``rotate_seats``: Terra Mystica is strongly seat- and faction-dependent, so
a single fixed seat measures seat strength as much as agent strength
(the arena's mirrored rotation exists for exactly this reason, D4.3).
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from bgai.env.config import (
    EnvConfig,
    EnvStateError,
    IllegalActionError,
)
from bgai.env.core import Episode
from bgai.env.opponents import Policy, build_opponent, random_policy
from bgai.env.serialize import obs_to_dict
from bgai.env.spaces import action_space, observation_space

__all__ = ["TerraMysticaSingleEnv", "legal_actions", "make_single_env"]


class TerraMysticaSingleEnv(gym.Env):
    """Single-agent Gymnasium env over :class:`bgai.env.core.Episode`."""

    metadata = {"render_modes": ["ansi"], "render_fps": 1}

    def __init__(
        self,
        config: EnvConfig | None = None,
        opponent: Policy | None = None,
        learner_seat: int = 0,
        rotate_seats: bool = False,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.config = config or EnvConfig()
        if not 0 <= learner_seat < self.config.player_count:
            raise ValueError(
                f"learner_seat must be in [0, {self.config.player_count - 1}], "
                f"got {learner_seat}"
            )
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"unknown render_mode {render_mode!r}")
        self.opponent: Policy = opponent or random_policy()
        self.learner_seat = learner_seat
        self.rotate_seats = rotate_seats
        self.render_mode = render_mode

        self.observation_space = observation_space(self.config.max_candidates)
        self.action_space = action_space(self.config.max_candidates)

        self._episode: Episode | None = None
        self._rng = random.Random()
        self._seat_index = learner_seat
        self._episode_count = 0
        self._opponent_decisions = 0
        self._carry = 0.0
        """Learner reward earned by opponents acting *before* the learner's
        own next decision. Under a terminal mode it is always 0; under
        DENSE_VP a leech during an opponent's turn is real VP and must be
        paid out, so it rides along to the next ``step``'s return."""

    # -- helpers -----------------------------------------------------------

    @property
    def episode(self) -> Episode:
        if self._episode is None:
            raise EnvStateError("env has not been reset()")
        return self._episode

    @property
    def learner_faction(self) -> str:
        return self.episode.seats[self._seat_index]

    def _obs(self) -> dict[str, Any]:
        return obs_to_dict(self.episode.observe(self.learner_faction))

    def _info(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        episode = self.episode
        info: dict[str, Any] = {
            "faction": self.learner_faction,
            "seat": self._seat_index,
            "seed": episode.seed,
            "game_id": episode.setup.game_id,
            "decisions": episode.decisions,
            "opponent": self.opponent.name,
            "opponent_decisions": self._opponent_decisions,
            "vps": episode.vps(),
        }
        if episode.engine_error is not None:
            info["engine_error"] = episode.engine_error
        if episode.illegal_action is not None:
            info["illegal_action"] = episode.illegal_action
        if extra:
            info.update(extra)
        return info

    # -- lifecycle ---------------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        super().reset(seed=seed)
        if seed is None:
            seed = int(self.np_random.integers(0, 2**31 - 1))
        self._rng = random.Random(seed)
        self._seat_index = (
            (self.learner_seat + self._episode_count) % self.config.player_count
            if self.rotate_seats
            else self.learner_seat
        )
        self._episode_count += 1
        self._opponent_decisions = 0
        self._episode = Episode.start(seed, self.config)
        self.opponent.begin_episode()
        reward, terminated, truncated = self._run_opponents()
        if reward != 0.0:
            # Opponents cannot earn the learner anything before its own
            # first decision under a terminal mode, and a dense VP delta
            # here belongs to the first step, not to reset(); carry it.
            self._carry = reward
        else:
            self._carry = 0.0
        if terminated or truncated:
            raise EnvStateError(
                "the table ended before the learner's first decision -- this "
                "should be impossible (setup placement is every seat's first "
                f"decision); engine_error={self.episode.engine_error!r}"
            )
        return self._obs(), self._info()

    def step(
        self, action: Any
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        episode = self.episode
        if episode.finished:
            raise IllegalActionError(
                "step called on a finished episode -- reset() first"
            )
        acting = episode.acting_seat()
        if acting != self.learner_faction:
            raise EnvStateError(
                f"env is waiting on {acting!r}, not the learner "
                f"{self.learner_faction!r}: _run_opponents left the table in a "
                "state it should not have"
            )

        self._episode, transition = episode.step(int(action))
        self.opponent.observe_move(transition.move)
        reward = float(transition.rewards[self.learner_faction]) + self._carry
        self._carry = 0.0
        terminated, truncated = transition.terminated, transition.truncated

        if not transition.done:
            more, terminated, truncated = self._run_opponents()
            reward += more

        info = self._info({"last_move": transition.move.verb})
        if transition.illegal_action is not None:
            info["illegal_action"] = transition.illegal_action
        if terminated or truncated:
            info["final_vps"] = self.episode.vps()
        return self._obs(), reward, terminated, truncated, info

    def _run_opponents(self) -> tuple[float, bool, bool]:
        """Let opponents act until the learner must decide or the game ends.

        Returns ``(learner reward accrued, terminated, truncated)``.
        """
        reward = 0.0
        while True:
            episode = self.episode
            if episode.finished:
                return reward, episode.sim.finished, not episode.sim.finished
            acting = episode.acting_seat()
            if acting is None:
                return reward, episode.sim.finished, not episode.sim.finished
            if acting == self.learner_faction:
                return reward, False, False
            _, offer = episode.pending()  # type: ignore[misc]
            index = self.opponent(episode.sim, acting, offer, self._rng)
            self._episode, transition = episode.step(index)
            self.opponent.observe_move(transition.move)
            self._opponent_decisions += 1
            reward += float(transition.rewards[self.learner_faction])
            if transition.done:
                return reward, transition.terminated, transition.truncated

    # -- render / state ----------------------------------------------------

    def render(self) -> str | None:
        if self.render_mode != "ansi":
            return None
        episode = self.episode
        vps = episode.vps()
        line = (
            f"r{episode.game.round} {episode.game.phase.name} "
            f"d{episode.decisions} learner={self.learner_faction} | "
            + " ".join(f"{f}:{vps[f]}" for f in episode.seats)
        )
        print(line)
        return line

    def close(self) -> None:
        return None

    def snapshot(self) -> bytes:
        return self.episode.to_bytes()

    def restore(self, payload: bytes) -> None:
        episode = Episode.from_bytes(payload)
        if episode.config.player_count != self.config.player_count:
            raise ValueError(
                f"snapshot is a {episode.config.player_count}-player table but "
                f"this env is configured for {self.config.player_count}"
            )
        self._episode = episode
        self.config = episode.config

def make_single_env(
    player_count: int = 4,
    opponent: str = "random",
    checkpoint: Path | None = None,
    learner_seat: int = 0,
    rotate_seats: bool = False,
    render_mode: str | None = None,
    **config_kwargs: Any,
) -> TerraMysticaSingleEnv:
    """Build a single-agent env from plain arguments.

    ``opponent`` is a name from :func:`bgai.env.opponents.known_opponents`;
    ``checkpoint`` is required for ``imitation``/``mcts``.
    """
    config = EnvConfig(player_count=player_count, **config_kwargs)
    policy = build_opponent(
        opponent, checkpoint=checkpoint, player_count=player_count
    )
    return TerraMysticaSingleEnv(
        config=config,
        opponent=policy,
        learner_seat=learner_seat,
        rotate_seats=rotate_seats,
        render_mode=render_mode,
    )


def legal_actions(obs: dict[str, Any]) -> np.ndarray:
    """Indices an observation's mask permits -- the one-liner every random
    baseline needs, so nobody re-derives it wrongly.
    """
    return np.flatnonzero(np.asarray(obs["action_mask"]))
