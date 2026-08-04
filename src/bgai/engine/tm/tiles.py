"""Static tile data for Terra Mystica: bonus/favor/town tiles, power actions,
and scoring tiles.

Ported from the reference implementation (jsnell/terra-mystica, MIT):

- ``src/Game/Constants.pm`` ``%actions`` (lines 55-105): the six base power
  actions ACT1-ACT6 (lines 56-65), faction special actions (not used here,
  see ``factions_data.py``), and the tile-triggered special actions merged
  onto BON1/BON2/FAV6 by ``init_tiles`` (lines 101-104).
- ``src/Game/Constants.pm`` ``init_tiles``/``%tiles`` (lines 107-224): bonus
  tiles BON1-10 (lines 127-141), favor tiles FAV1-12 (lines 143-157), scoring
  tiles SCORE1-9 (lines 159-213), and town tiles TW1-8 (lines 215-223). Every
  ``BONx``/``FAVx`` key present in ``%actions`` gets its ``action`` sub-hash
  merged in by ``init_tiles`` (lines 118-120) -- that is the source of
  ``BonusTile.special_action`` / ``FavorTile.special_action`` below.
- ``src/resources.pm`` ``setup_pool`` (lines 16-82), specifically the
  tile-pool loop (lines 64-81): favor tiles default to 3 copies unless the
  tile defines its own ``count`` (FAV1-4 have ``count => 1``, line 143-146);
  town tiles default to 2 copies unless the tile defines its own ``count``
  (TW6 and TW8 have ``count => 1``, lines 220 and 223; TW7 has no ``count``
  and so defaults to 2 even under the mini-expansion option, line 221-222);
  a tile whose ``option`` key is set is left out of the pool entirely unless
  that option is enabled for the game (lines 67-70).
- ``src/commands.pm`` ``option`` command's valid-option list (lines
  1362-1389) confirms the exact option name spellings used by ``%tiles``.

Option -> tile-id mapping (read directly off the ``option`` keys in
Constants.pm ``%tiles``; this is what a later setup loader needs to gate the
pool by the tournament corpus's enabled options):

- ``mini-expansion-1`` adds TW6, TW7, TW8 (Constants.pm lines 220-223).
- ``shipping-bonus`` adds BON10 (Constants.pm line 141).
- ``temple-scoring-tile`` adds SCORE9 (Constants.pm line 212).

No other tile in Constants.pm carries an ``option`` key, so these three
options are the complete set relevant to the tournament corpus (which
enables exactly these three, per the task brief) -- the original design
assumption (BON10 <-> shipping-bonus, SCORE9 <-> temple-scoring-tile,
TW6-8 <-> mini-expansion-1) is exactly what the Perl says; nothing to
correct.

The crawled corpus (``data/raw/games/*.json.gz``) embeds these same
Constants.pm tile hashes verbatim, JSON-serialized, under each game's
``bonus_tiles``/``favors``/``towns``/``score_tiles`` keys -- confirmed by
spot-checking against this module's transcription. ``ScoringTile`` therefore
gets a ``from_snellman`` factory: unlike BON/FAV/TW/ACT (fixed across every
game), which 6 of the 9 SCORE tiles are in play and in what round order
varies per game, so the per-game corpus dict is the runtime source of truth
rather than a static table.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class BonusTile:
    """Bonus tile BON1-10. Constants.pm ``%tiles``, lines 127-141."""

    income: Mapping[str, int]
    special_action: Mapping[str, int] | None  # BON1 spade, BON2 cult step
    pass_vp: tuple[tuple[str, int], ...]  # VP per unit held at pass, e.g. BON6: (("SH",4),("SA",4))
    passive: Mapping[str, int]  # e.g. BON4 {"ship": 1} == +1 shipping level while held


@dataclass(frozen=True)
class FavorTile:
    """Favor tile FAV1-12. Constants.pm ``%tiles``, lines 143-157.

    ``vp`` and ``pass_vp`` are kept separate because they are different Perl
    keys with different semantics: ``vp`` (FAV10/FAV11) is a one-time award
    scored when the tile is taken, scaled by buildings owned at that moment;
    ``pass_vp`` (FAV12 only) is scored at pass time and, unlike every BON
    pass_vp table, is *not* linear in building count (0,2,3,3,4 for 0-4
    Trading Posts) so it is kept as the full count-indexed tuple rather than
    collapsed to a per-unit rate.
    """

    cult: str
    steps: int
    income: Mapping[str, int]
    special_action: Mapping[str, int] | None  # FAV6: %actions gain => {CULT: 1}
    passive: Mapping[str, int]  # e.g. FAV5 {"TOWN_SIZE": -1}
    vp: Mapping[str, int]  # one-time gain-VP, e.g. FAV10 {"TP": 3}
    pass_vp: Mapping[str, tuple[int, ...]]  # count-indexed pass-VP, e.g. FAV12 {"TP": (0,2,3,3,4)}


@dataclass(frozen=True)
class TownTile:
    """Town founding tile TW1-8. Constants.pm ``%tiles``, lines 215-223."""

    vp: int
    gain: Mapping[str, int]  # remaining gain keys besides VP: KEY, C/W/P/PW, cult steps, etc.


@dataclass(frozen=True)
class PowerAction:
    """Base power action ACT1-6. Constants.pm ``%actions``, lines 56-65."""

    cost_power: int
    gain: Mapping[str, int]


@dataclass(frozen=True)
class ScoringTile:
    """Round scoring tile SCORE1-9. Constants.pm ``%tiles``, lines 159-213.

    Built at runtime via :meth:`from_snellman` from the corpus's per-game
    ``score_tiles`` entries, which are the same shape as the Constants.pm
    literals (``vp``, ``cult``, ``vp_mode``, ``income``, ``req``, plus the
    display strings computed by ``init_tiles``, lines 107-124).
    """

    cult: str
    req: int
    vp_mode: str
    vp: tuple[tuple[str, int], ...]
    cult_income: tuple[tuple[str, int], ...]
    vp_display: str = ""
    income_display: str = ""

    @staticmethod
    def from_snellman(d: Mapping[str, object]) -> ScoringTile:
        """Parse one entry of a corpus game's ``score_tiles`` list."""
        required = ("cult", "req", "vp_mode", "vp", "income")
        missing = [k for k in required if k not in d]
        if missing:
            raise ValueError(f"scoring tile dict missing keys {missing}: {d!r}")

        cult = d["cult"]
        req = d["req"]
        vp_mode = d["vp_mode"]
        vp = d["vp"]
        income = d["income"]
        if not isinstance(cult, str) or not isinstance(vp_mode, str):
            raise ValueError(f"scoring tile 'cult'/'vp_mode' must be str: {d!r}")
        if not isinstance(req, int):
            raise ValueError(f"scoring tile 'req' must be int: {d!r}")
        if not isinstance(vp, Mapping) or not isinstance(income, Mapping):
            raise ValueError(f"scoring tile 'vp'/'income' must be dicts: {d!r}")

        return ScoringTile(
            cult=cult,
            req=req,
            vp_mode=vp_mode,
            vp=tuple(sorted(vp.items())),
            cult_income=tuple(sorted(income.items())),
            vp_display=str(d.get("vp_display", "")),
            income_display=str(d.get("income_display", "")),
        )


def scored_vp(tile: ScoringTile, type_: str, mode: str) -> int:
    """Port of ``scoring.pm`` ``maybe_score_current_score_tile`` (lines
    20-30)::

        sub maybe_score_current_score_tile {
            my ($faction, $type, $mode) = @_;
            my $scoring = current_score_tile;
            if ($scoring) {
                my $gain = $scoring->{vp}{$type};
                if ($gain and $mode eq $scoring->{vp_mode}) {
                    adjust_resource $faction, 'VP', $gain, $scoring->{vp_display};
                }
            }
        }

    VP granted the instant a ``type_`` unit is gained/built under ``mode``
    this round -- 0 if ``tile`` doesn't key on ``type_`` at all, or keys on
    it under a *different* ``vp_mode`` (a game's whole 6-tile corpus sample
    only ever uses ``"build"`` (keyed by D/TP/TE/SH/SA -- ``command_build``/
    ``command_upgrade``, ``commands.pm`` 244-245/304) or ``"gain"`` (keyed
    by SPADE or TW1-8 -- the generic positive-resource-gain loop,
    ``resources.pm`` 388-391); no corpus game samples ``"spend"``, the
    third mode Perl's ``command_transform`` fires (``commands.pm`` 636-638,
    for spent SPADE) but no real ``%tiles`` entry ever keys. This is
    applied by the caller directly onto ``FactionState.vp`` -- unlike
    ``other_income_for_faction``/``cult_income_for_faction`` (a bookkeeping
    ledger row of its own, ``round_flow.py``), the corpus never emits a
    separate row for this grant; it is folded into the same ``build``/
    ``upgrade``/``dig``/``+TWx`` row that triggered it (empirically:
    reference game row 47, ``upgrade F6 to TP`` under round 1's ``TP >> 3``
    tile, jumps engineers' VP by exactly 3 with no companion row).
    """
    if mode != tile.vp_mode:
        return 0
    return dict(tile.vp).get(type_, 0)


# Constants.pm %actions, lines 56-65: the six base power-wheel actions.
POWER_ACTIONS: dict[str, PowerAction] = {
    "ACT1": PowerAction(cost_power=3, gain={"BRIDGE": 1}),
    "ACT2": PowerAction(cost_power=3, gain={"P": 1}),
    "ACT3": PowerAction(cost_power=4, gain={"W": 2}),
    "ACT4": PowerAction(cost_power=4, gain={"C": 7}),
    "ACT5": PowerAction(cost_power=4, gain={"SPADE": 1}),
    "ACT6": PowerAction(cost_power=6, gain={"SPADE": 2}),
}

# Constants.pm %tiles, lines 127-141 (income); special_action from %actions
# BON1/BON2 entries (lines 101-103); passive from the tile's own "special"
# key (only BON4 has one, line 130).
BONUS_TILES: dict[str, BonusTile] = {
    "BON1": BonusTile(income={"C": 2}, special_action={"SPADE": 1}, pass_vp=(), passive={}),
    "BON2": BonusTile(income={"C": 4}, special_action={"CULT": 1}, pass_vp=(), passive={}),
    "BON3": BonusTile(income={"C": 6}, special_action=None, pass_vp=(), passive={}),
    "BON4": BonusTile(income={"PW": 3}, special_action=None, pass_vp=(), passive={"ship": 1}),
    "BON5": BonusTile(income={"PW": 3, "W": 1}, special_action=None, pass_vp=(), passive={}),
    "BON6": BonusTile(
        income={"W": 2}, special_action=None, pass_vp=(("SH", 4), ("SA", 4)), passive={}
    ),
    "BON7": BonusTile(income={"W": 1}, special_action=None, pass_vp=(("TP", 2),), passive={}),
    "BON8": BonusTile(income={"P": 1}, special_action=None, pass_vp=(), passive={}),
    "BON9": BonusTile(income={"C": 2}, special_action=None, pass_vp=(("D", 1),), passive={}),
    # option "shipping-bonus" (Constants.pm line 141).
    "BON10": BonusTile(income={"PW": 3}, special_action=None, pass_vp=(("ship", 3),), passive={}),
}


def _fav(
    cult: str,
    steps: int,
    income: Mapping[str, int] | None = None,
    special_action: Mapping[str, int] | None = None,
    passive: Mapping[str, int] | None = None,
    vp: Mapping[str, int] | None = None,
    pass_vp: Mapping[str, tuple[int, ...]] | None = None,
) -> FavorTile:
    return FavorTile(
        cult=cult,
        steps=steps,
        income=income or {},
        special_action=special_action,
        passive=passive or {},
        vp=vp or {},
        pass_vp=pass_vp or {},
    )


# Constants.pm %tiles, lines 143-157.
FAVOR_TILES: dict[str, FavorTile] = {
    "FAV1": _fav("FIRE", 3),
    "FAV2": _fav("WATER", 3),
    "FAV3": _fav("EARTH", 3),
    "FAV4": _fav("AIR", 3),
    "FAV5": _fav("FIRE", 2, passive={"TOWN_SIZE": -1}),
    "FAV6": _fav("WATER", 2, special_action={"CULT": 1}),
    "FAV7": _fav("EARTH", 2, income={"W": 1, "PW": 1}),
    "FAV8": _fav("AIR", 2, income={"PW": 4}),
    "FAV9": _fav("FIRE", 1, income={"C": 3}),
    "FAV10": _fav("WATER", 1, vp={"TP": 3}),
    "FAV11": _fav("EARTH", 1, vp={"D": 2}),
    "FAV12": _fav("AIR", 1, pass_vp={"TP": (0, 2, 3, 3, 4)}),
}

# resources.pm setup_pool (lines 64-75): FAV1-4 have an explicit count => 1
# (Constants.pm lines 143-146); every other favor tile defaults to 3.
FAVOR_POOL_COUNTS: dict[str, int] = {
    "FAV1": 1,
    "FAV2": 1,
    "FAV3": 1,
    "FAV4": 1,
    "FAV5": 3,
    "FAV6": 3,
    "FAV7": 3,
    "FAV8": 3,
    "FAV9": 3,
    "FAV10": 3,
    "FAV11": 3,
    "FAV12": 3,
}

# Constants.pm %tiles, lines 215-223. VP split out of "gain" into TownTile.vp.
TOWN_TILES: dict[str, TownTile] = {
    "TW1": TownTile(vp=5, gain={"KEY": 1, "C": 6}),
    "TW2": TownTile(vp=7, gain={"KEY": 1, "W": 2}),
    "TW3": TownTile(vp=9, gain={"KEY": 1, "P": 1}),
    "TW4": TownTile(vp=6, gain={"KEY": 1, "PW": 8}),
    "TW5": TownTile(vp=8, gain={"KEY": 1, "FIRE": 1, "WATER": 1, "EARTH": 1, "AIR": 1}),
    # option "mini-expansion-1" (Constants.pm lines 220-223).
    "TW6": TownTile(vp=2, gain={"KEY": 2, "FIRE": 2, "WATER": 2, "EARTH": 2, "AIR": 2}),
    "TW7": TownTile(vp=4, gain={"KEY": 1, "GAIN_SHIP": 1, "carpet_range": 1}),
    "TW8": TownTile(vp=11, gain={"KEY": 1}),
}

# resources.pm setup_pool (lines 64-77): town tiles default to 2 copies
# unless the tile defines its own "count" (TW6 and TW8: count => 1, lines
# 220 and 223; TW7 has no "count" key and stays at the default 2).
TOWN_POOL_COUNTS: dict[str, int] = {
    "TW1": 2,
    "TW2": 2,
    "TW3": 2,
    "TW4": 2,
    "TW5": 2,
    "TW6": 1,
    "TW7": 2,
    "TW8": 1,
}

# Which game "option" (Constants.pm %tiles "option" key / resources.pm
# setup_pool lines 67-70) gates each tile's inclusion in the pool. Only
# tiles with an "option" key in Constants.pm appear here; everything else
# is always in the pool.
TILE_OPTIONS: dict[str, str] = {
    "BON10": "shipping-bonus",
    "SCORE9": "temple-scoring-tile",
    "TW6": "mini-expansion-1",
    "TW7": "mini-expansion-1",
    "TW8": "mini-expansion-1",
}

# Constants.pm %tiles SCORE1-9 (lines 159-213), transcribed as raw dicts in
# the exact corpus/Perl shape and parsed through the same from_snellman
# factory the runtime uses -- this is the canonical table; per-game score
# tile selection still comes from the corpus at runtime (see module
# docstring), but this documents what each SCORE id means, including which
# one (SCORE9) needs the temple-scoring-tile option.
_SCORE_RAW: dict[str, dict[str, object]] = {
    "SCORE1": {
        "vp": {"SPADE": 2},
        "vp_display": "SPADE >> 2",
        "vp_mode": "gain",
        "cult": "EARTH",
        "req": 1,
        "income": {"C": 1},
        "income_display": "1 EARTH -> 1 C",
    },
    "SCORE2": {
        "vp": {f"TW{i}": 5 for i in range(1, 9)},
        "vp_display": "TOWN >> 5",
        "vp_mode": "gain",
        "cult": "EARTH",
        "req": 4,
        "income": {"SPADE": 1},
        "income_display": "4 EARTH -> 1 SPADE",
    },
    "SCORE3": {
        "vp": {"D": 2},
        "vp_display": "D >> 2",
        "vp_mode": "build",
        "cult": "WATER",
        "req": 4,
        "income": {"P": 1},
        "income_display": "4 WATER -> 1 P",
    },
    "SCORE4": {
        "vp": {"SA": 5, "SH": 5},
        "vp_display": "SA/SH >> 5",
        "vp_mode": "build",
        "cult": "FIRE",
        "req": 2,
        "income": {"W": 1},
        "income_display": "2 FIRE -> 1 W",
    },
    "SCORE5": {
        "vp": {"D": 2},
        "vp_display": "D >> 2",
        "vp_mode": "build",
        "cult": "FIRE",
        "req": 4,
        "income": {"PW": 4},
        "income_display": "4 FIRE -> 4 PW",
    },
    "SCORE6": {
        "vp": {"TP": 3},
        "vp_display": "TP >> 3",
        "vp_mode": "build",
        "cult": "WATER",
        "req": 4,
        "income": {"SPADE": 1},
        "income_display": "4 WATER -> 1 SPADE",
    },
    "SCORE7": {
        "vp": {"SA": 5, "SH": 5},
        "vp_display": "SA/SH >> 5",
        "vp_mode": "build",
        "cult": "AIR",
        "req": 2,
        "income": {"W": 1},
        "income_display": "2 AIR -> 1 W",
    },
    "SCORE8": {
        "vp": {"TP": 3},
        "vp_display": "TP >> 3",
        "vp_mode": "build",
        "cult": "AIR",
        "req": 4,
        "income": {"SPADE": 1},
        "income_display": "4 AIR -> 1 SPADE",
    },
    # option "temple-scoring-tile" (Constants.pm line 212). "CULT_P" is not
    # a real cult track -- it is the priest-on-temple scoring hook.
    "SCORE9": {
        "vp": {"TE": 4},
        "vp_display": "TE >> 4",
        "vp_mode": "build",
        "cult": "CULT_P",
        "req": 1,
        "income": {"C": 2},
        "income_display": "1 CULT_P -> 2 C",
    },
}

SCORE_TILES: dict[str, ScoringTile] = {
    name: ScoringTile.from_snellman(raw) for name, raw in _SCORE_RAW.items()
}
