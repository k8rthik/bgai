"""Frozen vocabularies for imitation encoding (plan Task 1).

Every categorical the encoders emit indexes into one of these tuples.
``ENCODING_VERSION`` is stamped into every shard manifest; bump it on
ANY change to these vocabularies or to the encoders' layouts -- a
checkpoint is only compatible with shards of the same version.
"""

from __future__ import annotations

from bgai.engine.tm.board import base_board
from bgai.engine.tm.factions_data import CULTS, FACTION_SPECIAL_ACTIONS, FACTIONS
from bgai.engine.tm.tiles import BONUS_TILES, FAVOR_TILES, POWER_ACTIONS, TOWN_TILES

ENCODING_VERSION = 1

HEXES: tuple[str, ...] = tuple(sorted(base_board().hexes))
HEX_INDEX: dict[str, int] = {h: i for i, h in enumerate(HEXES)}

# Every verb the encoders can meet: the 18 corpus decision verbs, plus
# the three verbs only *generated* play produces -- `lose_spade` (the
# arena's income-window forfeit) and `gain_cult`/`lose_marker` (offered
# by legal_moves as pending answers / marker retirement, and so present
# among candidate sets even though no ledger row is a bare one). Both
# invariants are pinned in tests/test_vocab.py.
VERBS: tuple[str, ...] = (
    "action", "advance", "bridge", "build", "burn", "connect", "convert",
    "decline", "dig", "done", "gain_cult", "gain_favor", "gain_town", "leech",
    "lose_marker", "lose_spade", "pass", "send", "transform", "upgrade", "wait",
)
VERB_INDEX: dict[str, int] = {v: i for i, v in enumerate(VERBS)}

TILES: tuple[str, ...] = tuple(
    sorted(BONUS_TILES) + sorted(FAVOR_TILES) + sorted(TOWN_TILES)
    + sorted(POWER_ACTIONS) + sorted(FACTION_SPECIAL_ACTIONS)
)
TILE_INDEX: dict[str, int] = {t: i for i, t in enumerate(TILES)}

COLORS: tuple[str, ...] = ("black", "blue", "brown", "gray", "green", "red", "yellow")
COLOR_INDEX: dict[str, int] = {c: i for i, c in enumerate(COLORS)}

CULTS4: tuple[str, ...] = tuple(CULTS)
CULT_INDEX: dict[str, int] = {c: i for i, c in enumerate(CULTS4)}

RESOURCES: tuple[str, ...] = ("C", "P", "PW", "VP", "W")
RESOURCE_INDEX: dict[str, int] = {r: i for i, r in enumerate(RESOURCES)}

FACTION_NAMES: tuple[str, ...] = tuple(sorted(FACTIONS))
FACTION_INDEX: dict[str, int] = {f: i for i, f in enumerate(FACTION_NAMES)}

BUILDINGS: tuple[str, ...] = ("D", "TP", "TE", "SH", "SA")
BUILDING_INDEX: dict[str, int] = {b: i for i, b in enumerate(BUILDINGS)}


def normalize_color(color: str) -> str:
    """The corpus spells gray both ways; the vocabulary holds one."""
    return "gray" if color == "grey" else color
