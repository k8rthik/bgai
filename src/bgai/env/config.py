"""TM-Env configuration: every tunable constant and the frozen `EnvConfig`.

Nothing in `bgai.env` hard-codes a magic number; it comes from here. The
numbers that were *measured* rather than chosen carry their measurement.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from bgai.engine.tm.factions_data import FACTIONS

# --------------------------------------------------------------------------
# table shape
# --------------------------------------------------------------------------

MIN_PLAYERS = 2
MAX_PLAYERS = 5
"""Terra Mystica's own player-count range. The engine implements all four
counts; only 4 is replay-validated against tournament data (README,
"Validated scope"), so 2/3/5-player tables are reference-rules correct but
unverified against real games. The env exposes them and says so."""

MAX_SEATS = MAX_PLAYERS
"""Seat blocks in the observation. Fixed at 5 so one observation layout
serves every player count -- absent seats are zero and the
``seat_present`` block says which are real. A per-count layout would make
a 2-player checkpoint structurally unusable at 4."""

SCORE_TILES_PER_GAME = 6
BONUS_POOL_EXTRA = 3
"""Bonus-tile pool size is ``player_count + BONUS_POOL_EXTRA``
(``engine/tm/setup.py:_resolve_bonus_tiles``)."""

# --------------------------------------------------------------------------
# action space
# --------------------------------------------------------------------------

MAX_CANDIDATES = 320
"""Size of the candidate-index action space.

The engine offers a *variable* candidate set, so the action space is an
index into the canonically ordered offer (``driver.canonical_moves``) and
the observation carries the candidates' own features. ``MAX_CANDIDATES``
is the padding width, not a rules constant.

Measured: over 600 random-play games at 2/3/4/5 players (85,460
decisions) the largest offer was **232** (a 2-player Mermaids turn); mean
10.7, p99 50. 320 leaves ~38% headroom over the observed maximum. An
offer larger than this raises :class:`OfferOverflowError` rather than
silently truncating the legal set -- a truncated mask would be a
rules-correctness bug, not a capacity nuisance.
"""

DEFAULT_MAX_DECISIONS = 5000
"""Truncation budget per episode, matching ``arena.sim.run_game``'s cap.
Measured episode lengths under random play: 81 (2p), 121 (3p), 164 (4p),
212 (5p) decisions, so this is ~20x the 5-player mean."""

# --------------------------------------------------------------------------
# observation value bounds
# --------------------------------------------------------------------------

HEX_PLANE_DTYPE_MIN = 0
HEX_PLANE_DTYPE_MAX = 1
GLOBAL_MIN = -1
GLOBAL_MAX = 1024
"""Bound on every ``globals`` entry. The largest quantity in there is a
faction's VP; the corpus maximum is 236 and a pathological table total
stays far under 1024. ``GLOBAL_MIN`` is -1 only because an engine
quantity could in principle be stored negative; nothing currently is."""

CANDIDATE_MIN = 0
CANDIDATE_MAX = 512
"""Bound on encoded move fields. The widest field is the hex id (113
hexes) and amounts are clamped to 30 by ``encode_move``."""

VP_START = 20
"""Every faction starts on 20 VP (``FactionState.initial``). Dense VP
rewards are measured against this origin so an episode's dense rewards
sum to ``(final_vp - 20) / scale``."""

# --------------------------------------------------------------------------
# reward
# --------------------------------------------------------------------------


class RewardMode(StrEnum):
    """How an episode's rewards are formed. See ``bgai.env.reward``."""

    TERMINAL_VP_SHARE = "terminal_vp_share"
    TERMINAL_RANK = "terminal_rank"
    TERMINAL_WIN = "terminal_win"
    DENSE_VP = "dense_vp"


DENSE_VP_SCALE = 100.0
"""Divisor on per-decision VP deltas in ``DENSE_VP``. Chosen so a typical
episode's dense return is O(1): our champion scores ~111 VP and Div 1-3
humans ~136, i.e. ~0.9-1.2 after subtracting the 20 VP start."""

DENSE_TERMINAL_WEIGHT = 1.0
"""Weight on the terminal VP-share term added on top of the dense VP
deltas in ``DENSE_VP``. Keeps the mode's argmax tied to the competitive
objective rather than to absolute VP alone."""

# --------------------------------------------------------------------------
# setup sourcing
# --------------------------------------------------------------------------


class IllegalActionPolicy(StrEnum):
    """What an action the live mask forbids does.

    ``TERMINATE`` (default): the episode ends immediately, the offender
    collects :data:`ILLEGAL_ACTION_REWARD`, every other seat collects 0,
    and ``info['illegal_action']`` records the index. This is the standard
    masked-env contract (PettingZoo ships ``TerminateIllegalWrapper`` for
    it) and it is what makes ``gymnasium.utils.env_checker.check_env``
    pass: that checker samples the *whole* ``Discrete`` space with no mask,
    so an env that only ever raises cannot conform.

    ``RAISE``: :class:`IllegalActionError`, naming the legal index range.
    Stricter and better while debugging a policy -- a masked policy can
    never reach it, so reaching it is a bug you want to see immediately
    rather than as a mysterious -1.0.

    Either way the mask is never silently "repaired": projecting an
    illegal index onto a legal one (modulo, clamp, nearest) would make
    every index playable and quietly delete the mask's meaning.
    """

    TERMINATE = "terminate"
    RAISE = "raise"


ILLEGAL_ACTION_REWARD = -1.0
"""Reward for playing a masked action under ``TERMINATE``.

Set at or below the worst legitimate terminal reward in every mode: rank
bottoms out at exactly -1.0, VP-share at ``-1/n`` (>= -0.5), win at
``-1/n``. So illegal play is never preferable to losing badly.
"""


class SetupSource(StrEnum):
    """Where a table's fixed configuration comes from.

    ``SYNTHETIC`` draws score tiles, bonus pool and faction lineup from a
    seeded RNG (the only option at 2/3/5 players, since the corpus is
    entirely 4-player). ``CORPUS`` samples a real replay-validated Div 1-3
    game's setup via ``arena.setups.sample_setup`` -- 4 players only, and
    requires the gitignored ``data/`` tree.
    """

    SYNTHETIC = "synthetic"
    CORPUS = "corpus"


AGENT_NAME_TEMPLATE = "player_{index}"
"""PettingZoo recommends ``<descriptor>_<number>`` agent ids, and its
``api_test`` warns otherwise. Faction names live in ``info['faction']``
and in ``env.factions``."""

# --------------------------------------------------------------------------
# errors
# --------------------------------------------------------------------------


class EnvError(Exception):
    """Base class for every TM-Env boundary failure."""


class IllegalActionError(EnvError, ValueError):
    """An action index outside the offer, or masked out of it."""


class OfferOverflowError(EnvError, RuntimeError):
    """The engine offered more candidates than the action space holds."""


class EnvStateError(EnvError, RuntimeError):
    """The env was used out of order (stepped before reset, etc.)."""


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class EnvConfig:
    """Immutable env configuration. Validated on construction."""

    player_count: int = 4
    reward_mode: RewardMode = RewardMode.TERMINAL_VP_SHARE
    setup_source: SetupSource = SetupSource.SYNTHETIC
    factions: tuple[str, ...] | None = None
    """Fixed faction lineup in seat order. ``None`` draws a colour-legal
    lineup from the episode seed."""
    max_decisions: int = DEFAULT_MAX_DECISIONS
    max_candidates: int = MAX_CANDIDATES
    illegal_action: IllegalActionPolicy = IllegalActionPolicy.TERMINATE
    raise_on_engine_error: bool = False
    """What to do when the engine rejects a move it had itself offered.

    ``False`` (default) ends the episode truncated with the rejection on
    every agent's ``info``, matching ``arena.sim.run_game``: random-play
    fuzzing through the arena is how two real ``legal_moves`` soundness
    bugs were found (``tests/test_legal_soundness.py``) and the repo's rule
    is that such findings must be *visible, not fatal*. ``True`` re-raises
    the ``EngineError``, for anyone who wants a long training run to stop
    at the first one.
    """

    def __post_init__(self) -> None:
        if not MIN_PLAYERS <= self.player_count <= MAX_PLAYERS:
            raise ValueError(
                f"player_count must be in [{MIN_PLAYERS}, {MAX_PLAYERS}], "
                f"got {self.player_count}"
            )
        if not isinstance(self.reward_mode, RewardMode):
            raise TypeError(f"reward_mode must be a RewardMode, got {self.reward_mode!r}")
        if not isinstance(self.setup_source, SetupSource):
            raise TypeError(
                f"setup_source must be a SetupSource, got {self.setup_source!r}"
            )
        if not isinstance(self.illegal_action, IllegalActionPolicy):
            raise TypeError(
                f"illegal_action must be an IllegalActionPolicy, got "
                f"{self.illegal_action!r}"
            )
        if self.max_decisions < 1:
            raise ValueError(f"max_decisions must be >= 1, got {self.max_decisions}")
        if self.max_candidates < 1:
            raise ValueError(f"max_candidates must be >= 1, got {self.max_candidates}")
        if (
            self.setup_source is SetupSource.CORPUS
            and self.player_count != 4
        ):
            raise ValueError(
                "setup_source=CORPUS is 4-player only: every one of the 3,563 "
                f"corpus games is 4-player, got player_count={self.player_count}"
            )
        if self.factions is not None:
            self._validate_factions(self.factions, self.player_count)

    @staticmethod
    def _validate_factions(factions: tuple[str, ...], player_count: int) -> None:
        if len(factions) != player_count:
            raise ValueError(
                f"factions has {len(factions)} entries but player_count is "
                f"{player_count}"
            )
        unknown = [f for f in factions if f not in FACTIONS]
        if unknown:
            raise ValueError(f"unknown factions {unknown}")
        if len(set(factions)) != len(factions):
            raise ValueError(f"duplicate factions in {factions}")
        colors = [FACTIONS[f].color for f in factions]
        if len(set(colors)) != len(colors):
            raise ValueError(
                f"two factions share a colour in {factions} "
                f"(colours {colors}) -- illegal lineup"
            )
