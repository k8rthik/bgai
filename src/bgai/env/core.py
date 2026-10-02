"""The engine<->env glue both API wrappers share.

``Episode`` is an immutable snapshot of a table paused at a decision
point. ``Episode.step`` returns a NEW ``Episode`` plus a ``Transition``
describing what the move earned; nothing here is mutated in place, so an
episode can be branched, stored, or replayed exactly like the driver's own
``SimState`` (that was ``driver``'s whole reason for existing -- D6.1).

Neither gymnasium nor pettingzoo is imported in this module. The AEC env
and the single-agent env are both thin adapters over this, which is why
they cannot disagree about the rules, the mask, or the reward.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from bgai.arena.driver import SimState, advance, decision, new_game
from bgai.data.ledger_parser import ParsedCommand
from bgai.engine.tm.apply import EngineError
from bgai.engine.tm.setup import GameSetup
from bgai.engine.tm.state import GameState
from bgai.env.canonical import canonical_offer
from bgai.env.config import (
    ILLEGAL_ACTION_REWARD,
    EnvConfig,
    IllegalActionError,
    IllegalActionPolicy,
    RewardMode,
)
from bgai.env.observation import Observation, encode_observation
from bgai.env.reward import dense_step_rewards, terminal_rewards, zero_rewards
from bgai.env.table import build_setup

__all__ = ["Episode", "Transition", "vp_table"]


def vp_table(game: GameState) -> dict[str, int]:
    """Every seat's current VP, in seat order."""
    return {f: game.factions[f].vp for f in game.setup.factions}


@dataclass(frozen=True)
class Transition:
    """What one applied action produced.

    ``rewards`` carries a value for *every* seat, not just the mover: a
    build hands power to its neighbours and the scoring phase pays
    everyone, so per-step credit is genuinely multi-agent. Terminal
    rewards arrive on the transition that ends the episode.
    """

    move: ParsedCommand
    faction: str
    rewards: dict[str, float]
    terminated: bool
    truncated: bool
    engine_error: str | None = None
    illegal_action: int | None = None
    """The rejected index, when the episode ended because a masked action
    was played under ``IllegalActionPolicy.TERMINATE``."""

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated


@dataclass(frozen=True)
class Episode:
    """One table paused at a decision point (or finished)."""

    config: EnvConfig
    setup: GameSetup
    sim: SimState
    seed: int
    engine_error: str | None = None
    truncated: bool = False
    illegal_action: int | None = None

    # -- construction ------------------------------------------------------

    @classmethod
    def start(cls, seed: int, config: EnvConfig) -> Episode:
        """A fresh table settled to its first decision point."""
        setup = build_setup(seed, config)
        if len(setup.factions) != config.player_count:
            raise ValueError(
                f"setup has {len(setup.factions)} seats but config.player_count "
                f"is {config.player_count}"
            )
        return cls(config=config, setup=setup, sim=new_game(setup), seed=seed)

    # -- read-only views ---------------------------------------------------

    @property
    def seats(self) -> tuple[str, ...]:
        """Faction names in seat order -- the stable seat identity."""
        return self.setup.factions

    @property
    def game(self) -> GameState:
        return self.sim.game

    @property
    def decisions(self) -> int:
        return self.sim.decisions

    @property
    def finished(self) -> bool:
        """True once no further action can be taken, for any reason."""
        return (
            self.sim.finished
            or self.truncated
            or self.engine_error is not None
            or self.illegal_action is not None
        )

    def _raw_pending(
        self,
    ) -> tuple[str, tuple[ParsedCommand, ...], tuple[ParsedCommand, ...]] | None:
        """``(faction, normalized offer, engine originals)``.

        The two offers are index-aligned: the first is what an action index
        names and what the observation encodes, the second is what
        ``driver.advance`` is given (it checks membership by equality, so it
        has to receive the engine's own object). See ``bgai.env.canonical``
        for why they can differ at all.
        """
        if self.finished:
            return None
        current = decision(self.sim)
        if current is None:
            return None
        faction, offer = current
        normalized, originals = canonical_offer(offer)
        return faction, normalized, originals

    def pending(self) -> tuple[str, tuple[ParsedCommand, ...]] | None:
        """``(faction, canonical offer)`` for the seat to decide, or None."""
        current = self._raw_pending()
        return None if current is None else (current[0], current[1])

    def acting_seat(self) -> str | None:
        current = self.pending()
        return None if current is None else current[0]

    def offer_for(self, faction: str) -> tuple[ParsedCommand, ...]:
        """The offer ``faction`` faces -- empty unless it is their decision."""
        current = self.pending()
        if current is None or current[0] != faction:
            return ()
        return current[1]

    def observe(self, faction: str) -> Observation:
        """``faction``'s mover-relative view, with its own legal mask."""
        if faction not in self.game.factions:
            raise KeyError(f"{faction!r} is not a seat at this table {self.seats}")
        return encode_observation(
            self.game, faction, self.offer_for(faction), self.config.max_candidates
        )

    def vps(self) -> dict[str, int]:
        return vp_table(self.game)

    # -- stepping ----------------------------------------------------------

    def _as_index(self, action: int, faction: str, n_legal: int) -> int:
        try:
            return int(action)
        except (TypeError, ValueError) as exc:
            raise IllegalActionError(
                f"action must be an integer index into {faction!r}'s "
                f"{n_legal}-move offer, got {action!r}"
            ) from exc

    def resolve(self, action: int) -> ParsedCommand:
        """The move ``action`` names, validated against the live mask.

        This is the env's input boundary: an index the mask forbids never
        becomes a different, legal move. It raises
        :class:`IllegalActionError` naming what *was* legal, and callers
        that prefer the terminate-with-penalty contract go through
        :meth:`step` with ``IllegalActionPolicy.TERMINATE``.
        """
        current = self._raw_pending()
        if current is None:
            raise IllegalActionError(
                "episode is over: no action is legal "
                f"(finished={self.sim.finished}, truncated={self.truncated}, "
                f"engine_error={self.engine_error!r}, "
                f"illegal_action={self.illegal_action!r})"
            )
        faction, offer, _ = current
        index = self._as_index(action, faction, len(offer))
        if not 0 <= index < len(offer):
            raise IllegalActionError(
                f"action {index} is illegal for {faction!r}: the offer holds "
                f"{len(offer)} moves, so legal indices are 0..{len(offer) - 1} "
                f"(action_mask has exactly those bits set)"
            )
        return offer[index]

    def step(self, action: int) -> tuple[Episode, Transition]:
        """Apply ``action`` and settle forward to the next decision point."""
        current = self._raw_pending()
        if current is None:
            raise IllegalActionError("step called on a finished episode")
        faction, offer, originals = current
        index = self._as_index(action, faction, len(offer))
        if not 0 <= index < len(offer):
            if self.config.illegal_action is IllegalActionPolicy.RAISE:
                self.resolve(index)  # raises with the full legal-range message
            return self._illegal_action(faction, index, len(offer))
        move = offer[index]
        before = self.vps()

        try:
            sim = advance(self.sim, originals[index])
        except EngineError as exc:
            return self._engine_rejection(faction, move, exc)

        nxt = replace(self, sim=sim)
        truncated = not sim.finished and sim.decisions >= self.config.max_decisions
        if truncated:
            nxt = replace(nxt, truncated=True)

        rewards = dense_step_rewards(
            self.config.reward_mode, before, nxt.vps(), self.seats
        )
        terminated = sim.finished
        if terminated:
            final = terminal_rewards(self.config.reward_mode, nxt.vps())
            rewards = {seat: rewards[seat] + final[seat] for seat in self.seats}
        return nxt, Transition(
            move=move,
            faction=faction,
            rewards=rewards,
            terminated=terminated,
            truncated=truncated,
        )

    def _illegal_action(
        self, faction: str, index: int, n_legal: int
    ) -> tuple[Episode, Transition]:
        """End the episode because ``faction`` played a masked index."""
        from bgai.engine.tm.legal_shared import cmd

        nxt = replace(self, illegal_action=index)
        rewards = zero_rewards(self.seats)
        rewards[faction] = ILLEGAL_ACTION_REWARD
        return nxt, Transition(
            move=cmd("wait"),
            faction=faction,
            rewards=rewards,
            terminated=True,
            truncated=False,
            illegal_action=index,
        )

    def _engine_rejection(
        self, faction: str, move: ParsedCommand, exc: EngineError
    ) -> tuple[Episode, Transition]:
        """The engine rejected a move it had itself offered.

        That is a ``legal_moves`` soundness bug, and the repo treats such
        findings as *visible, not fatal* -- random-play arena fuzzing is
        how two of them were found and fixed
        (``tests/test_legal_soundness.py``). So by default the episode ends
        truncated with the rejection recorded on every agent's ``info``,
        and ``EnvConfig.raise_on_engine_error`` turns it back into an
        exception for anyone who wants a hard failure.
        """
        detail = f"decision {self.sim.decisions} ({faction}, {move.verb}): {exc}"
        if self.config.raise_on_engine_error:
            raise
        nxt = replace(self, engine_error=detail, truncated=True)
        return nxt, Transition(
            move=move,
            faction=faction,
            rewards=zero_rewards(self.seats),
            terminated=False,
            truncated=True,
            engine_error=detail,
        )

    # -- serialization -----------------------------------------------------

    def to_dict(self) -> dict[str, object]:
        """A JSON-safe snapshot. Round-trips through :meth:`from_dict`."""
        from bgai.env.codec import sim_to_dict

        return {
            "seed": self.seed,
            "config": {
                "player_count": self.config.player_count,
                "reward_mode": self.config.reward_mode.value,
                "setup_source": self.config.setup_source.value,
                "factions": (
                    None if self.config.factions is None else list(self.config.factions)
                ),
                "max_decisions": self.config.max_decisions,
                "max_candidates": self.config.max_candidates,
                "illegal_action": self.config.illegal_action.value,
                "raise_on_engine_error": self.config.raise_on_engine_error,
            },
            "sim": sim_to_dict(self.sim),
            "engine_error": self.engine_error,
            "truncated": self.truncated,
            "illegal_action": self.illegal_action,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> Episode:
        """Inverse of :meth:`to_dict`."""
        from bgai.env.codec import CodecError, sim_from_dict
        from bgai.env.config import IllegalActionPolicy, SetupSource

        if not isinstance(raw, dict):
            raise CodecError(f"episode payload must be a dict, got {type(raw).__name__}")
        cfg_raw = raw.get("config")
        if not isinstance(cfg_raw, dict):
            raise CodecError("episode payload is missing its 'config' object")
        factions = cfg_raw.get("factions")
        config = EnvConfig(
            player_count=int(cfg_raw["player_count"]),
            reward_mode=RewardMode(cfg_raw["reward_mode"]),
            setup_source=SetupSource(cfg_raw["setup_source"]),
            factions=None if factions is None else tuple(factions),
            max_decisions=int(cfg_raw["max_decisions"]),
            max_candidates=int(cfg_raw["max_candidates"]),
            illegal_action=IllegalActionPolicy(cfg_raw["illegal_action"]),
            raise_on_engine_error=bool(cfg_raw.get("raise_on_engine_error", False)),
        )
        sim = sim_from_dict(raw["sim"])  # type: ignore[arg-type]
        return cls(
            config=config,
            setup=sim.game.setup,
            sim=sim,
            seed=int(raw["seed"]),  # type: ignore[arg-type]
            engine_error=raw.get("engine_error"),  # type: ignore[arg-type]
            truncated=bool(raw.get("truncated", False)),
            illegal_action=(
                None if raw.get("illegal_action") is None
                else int(raw["illegal_action"])  # type: ignore[arg-type]
            ),
        )

    def to_bytes(self) -> bytes:
        import orjson

        return orjson.dumps(self.to_dict())

    @classmethod
    def from_bytes(cls, payload: bytes) -> Episode:
        import orjson

        from bgai.env.codec import CodecError

        try:
            raw = orjson.loads(payload)
        except orjson.JSONDecodeError as exc:
            raise CodecError(f"not a valid encoded episode: {exc}") from exc
        return cls.from_dict(raw)
