"""Per-game options and setup loader for the crawled snellman corpus.

Each ``data/raw/games/<game_id>.json.gz`` file is a gzipped JSON dump of one
finished game from terra.snellman.net. This module turns that raw dict into
the two things a rules engine needs before it can replay a game:

- :class:`GameOptions`: the boolean ruleset flags a game was played under
  (``options`` command list, ``src/commands.pm`` lines 1362-1389 in the
  jsnell/terra-mystica reference implementation). Only the option names
  relevant to this project's rules/scoring are modeled as fields; anything
  else present in a game's option list (e.g. ``email-notify``, which is a
  notification preference, not a rules variant) is ignored.
- :class:`GameSetup`: the fixed, pre-round-1 configuration of one game --
  seat order, the 6 scoring tiles drawn for the 6 rounds, and the bonus-tile
  subset in the pool -- built from the raw JSON's ``factions``, ``options``,
  ``score_tiles``, and ``pool`` keys.

Field discoveries worth recording (the raw JSON is not internally
consistent about where some of this data lives):

- Top-level ``player_count`` is ``None`` on every finished game sampled;
  the real value lives at ``metadata.player_count``. ``data/datasets/
  games_meta.parquet`` is used as a second fallback if even that is
  missing.
- The top-level ``order`` key is *not* seat order (spot-checked against
  several games: it does not match the ledger's actual turn-order rows).
  Real seat order comes from each ``factions[<name>].start_order`` (1-based
  int), sorted ascending.
- ``options`` is a dict of ``{option-name: 1}`` (a set, not a list), and is
  the per-game source of truth; ``games_meta.parquet``'s ``options`` column
  (a CSV string) is used as a fallback if the raw key is absent or null.
"""

from __future__ import annotations

import gzip
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import polars as pl

from bgai.engine.tm.tiles import ScoringTile

_BON_ID_RE = re.compile(r"BON\d+")
_DROP_COMMENT_RE = re.compile(r"^(\w+) dropped from the game$")

# CSV option-name -> GameOptions field name. Anything not in this table
# (e.g. "email-notify", "loose-lose-cult") is a real corpus option but not
# a rules variant this engine models, and is silently ignored.
_OPTION_FIELDS: dict[str, str] = {
    "strict-leech": "strict_leech",
    "errata-cultist-power": "errata_cultist_power",
    "variable-turn-order": "variable_turn_order",
    "maintain-player-order": "maintain_player_order",
    "strict-darkling-sh": "strict_darkling_sh",
    "strict-chaosmagician-sh": "strict_chaosmagician_sh",
    "mini-expansion-1": "mini_expansion_1",
    "shipping-bonus": "shipping_bonus",
    "temple-scoring-tile": "temple_scoring_tile",
    "loose-dig": "loose_dig",
    "merge-income-phases": "merge_income_phases",
}


@dataclass(frozen=True)
class GameOptions:
    """Boolean ruleset flags a game was played under.

    Field names are the CSV option names (``src/commands.pm`` option-list,
    lines 1362-1389) with ``-`` replaced by ``_``.
    """

    strict_leech: bool = False
    errata_cultist_power: bool = False
    variable_turn_order: bool = False
    maintain_player_order: bool = False
    strict_darkling_sh: bool = False
    strict_chaosmagician_sh: bool = False
    mini_expansion_1: bool = False
    shipping_bonus: bool = False
    temple_scoring_tile: bool = False
    loose_dig: bool = False
    merge_income_phases: bool = False

    @classmethod
    def from_csv(cls, text: str) -> GameOptions:
        """Parse a comma-separated option-name list. Unknown names are ignored."""
        names = {part.strip() for part in text.split(",") if part.strip()}
        fields = {
            field: True for name, field in _OPTION_FIELDS.items() if name in names
        }
        return cls(**fields)


@dataclass(frozen=True)
class GameSetup:
    """Fixed pre-round-1 configuration for one crawled game."""

    game_id: str
    options: GameOptions
    factions: tuple[str, ...]  # seat order
    score_tiles: tuple[ScoringTile, ...]  # len 6, round 1..6
    bonus_tiles: tuple[str, ...]  # sorted BON ids in this game's pool
    player_count: int
    dropped_at_row: Mapping[str, int] = field(default_factory=dict)
    """``{faction: ledger row index}`` for every faction the raw JSON's
    final snapshot marks ``"dropped": 1`` -- the exact row of that
    faction's own ``"<faction> dropped from the game"`` ledger *comment*
    (``commands.pm``'s ``drop-faction`` handler, ~1578-1602: sets
    ``$faction->{dropped}=1``/``allowed_actions=0``, releases its held
    bonus tile via ``find_bonus_to_discard``/``adjust_resource(-1)``,
    drops it from ``setup_order``, then ``$game{ledger}->add_comment("$f
    dropped from the game")`` as the last step of that same event -- all
    one atomic ledger moment in real Perl, not a range). A real, fairly
    common corpus phenomenon (an AFK/timed-out player), not covered by
    Task 2's 10 ``nofaction*`` known-bad games (a *different* code path,
    ``drop-faction playerN`` for a player who never even joined --
    ``register_faction`` with ``dropped=>1`` baked in from the start,
    which never emits this comment and has no rows to key off of; those
    10 games already fail to load for unrelated reasons before this
    matters). A dropped (non-``nofaction``) faction still played real
    early-game moves before dropping, so it can't be treated as
    unloadable/excluded outright.

    This is a precise, deterministic signal read straight from the same
    event Perl itself recorded -- not inferred from the faction's last
    surviving ``moves.parquet`` row (an earlier revision's heuristic,
    which conflated "stopped acting" with "the admin actually marked them
    dropped," often several rows or even whole rounds apart, since a
    comment-only row carries no ``commands`` and so is itself invisible
    to ``moves.parquet``/``deltas.parquet``, both of which
    ``build_moves.py`` builds only from rows that have a ``commands``
    key). Verified 1:1 against the full corpus: every one of the 179
    non-``nofaction`` games with at least one ``"dropped": 1`` faction has
    exactly one such comment per dropped faction, no duplicates, no
    misses. ``replay.py`` uses this map to apply the drop -- release the
    bonus tile, exclude the faction from turn order -- proactively, at
    the exact row, rather than reactively once some other faction's
    turn-order mismatch reveals it.
    """


def _read_raw_game(game_id: str, raw_dir: Path) -> dict[str, object]:
    path = raw_dir / f"{game_id}.json.gz"
    with gzip.open(path, "rt", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{game_id}: raw game file is not a JSON object")
    return data


@lru_cache(maxsize=1)
def _games_meta() -> pl.DataFrame:
    return pl.read_parquet(Path("data/datasets/games_meta.parquet"))


def _games_meta_row(game_id: str) -> dict[str, object]:
    """Fetch one game's fallback metadata row, or raise ValueError."""
    df = _games_meta()
    rows = df.filter(pl.col("game_id") == game_id)
    if rows.height == 0:
        raise ValueError(f"{game_id}: not found in raw dir or games_meta.parquet fallback")
    return {column: rows[column][0] for column in df.columns}


def _resolve_options(game_id: str, raw: dict[str, object]) -> GameOptions:
    raw_options = raw.get("options")
    if isinstance(raw_options, dict):
        return GameOptions.from_csv(",".join(raw_options.keys()))
    value = _games_meta_row(game_id)["options"]
    return GameOptions.from_csv(str(value) if value is not None else "")


def _resolve_player_count(game_id: str, raw: dict[str, object]) -> int:
    metadata = raw.get("metadata")
    if isinstance(metadata, dict):
        count = metadata.get("player_count")
        if isinstance(count, int) and count > 0:
            return count
    top_level = raw.get("player_count")
    if isinstance(top_level, int) and top_level > 0:
        return top_level

    value = _games_meta_row(game_id)["player_count"]
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{game_id}: invalid fallback player_count {value!r}")
    return value


def _resolve_seat_order(game_id: str, raw: dict[str, object]) -> tuple[str, ...]:
    factions = raw.get("factions")
    if not isinstance(factions, dict) or not factions:
        raise ValueError(f"{game_id}: raw JSON has no 'factions' data for seat order")

    ordered: list[tuple[int, str]] = []
    for name, info in factions.items():
        # `name` is always str: JSON object keys decode to str via json.load.
        if not isinstance(info, dict):
            raise ValueError(f"{game_id}: malformed 'factions' entry {name!r}")
        start_order = info.get("start_order")
        if not isinstance(start_order, int):
            raise ValueError(f"{game_id}: faction {name!r} missing int 'start_order'")
        ordered.append((start_order, name))
    ordered.sort()

    # `factions` was checked non-empty above, so `ordered` (one entry per
    # faction) is guaranteed non-empty here.
    return tuple(name for _, name in ordered)


def _resolve_score_tiles(game_id: str, raw: dict[str, object]) -> tuple[ScoringTile, ...]:
    raw_tiles = raw.get("score_tiles")
    if not isinstance(raw_tiles, list):
        raise ValueError(f"{game_id}: raw JSON 'score_tiles' is not a list")

    tiles: list[ScoringTile] = []
    for index, entry in enumerate(raw_tiles):
        try:
            tiles.append(ScoringTile.from_snellman(entry))
        except ValueError as exc:
            raise ValueError(f"{game_id}: score_tiles[{index}] invalid: {exc}") from exc

    if len(tiles) != 6:
        raise ValueError(f"{game_id}: expected 6 score tiles, got {len(tiles)}")
    return tuple(tiles)


def _resolve_bonus_tiles(
    game_id: str, raw: dict[str, object], player_count: int
) -> tuple[str, ...]:
    pool = raw.get("pool")
    if not isinstance(pool, dict):
        raise ValueError(f"{game_id}: raw JSON 'pool' is not a dict")
    bonus_tiles = tuple(sorted(k for k in pool if _BON_ID_RE.fullmatch(k)))
    expected = player_count + 3
    if len(bonus_tiles) != expected:
        raise ValueError(
            f"{game_id}: expected {expected} bonus tiles (player_count {player_count} + 3), "
            f"got {len(bonus_tiles)}: {bonus_tiles}"
        )
    return bonus_tiles


def _resolve_dropped_at_row(raw: dict[str, object]) -> Mapping[str, int]:
    """``GameSetup.dropped_at_row`` docstring: scan the raw ledger's own
    comment rows for ``"<faction> dropped from the game"`` -- the exact
    row ``commands.pm``'s ``drop-faction`` handler adds as the last step
    of the drop event itself.
    """
    ledger = raw.get("ledger")
    if not isinstance(ledger, list):
        return {}
    rows: dict[str, int] = {}
    for row_idx, row in enumerate(ledger):
        if not isinstance(row, dict):
            continue
        comment = row.get("comment")
        if not isinstance(comment, str):
            continue
        match = _DROP_COMMENT_RE.match(comment)
        if match:
            rows[match.group(1)] = row_idx
    return rows


def load_setup(game_id: str, raw_dir: Path = Path("data/raw/games")) -> GameSetup:
    """Load and validate one crawled game's pre-round-1 setup."""
    raw = _read_raw_game(game_id, raw_dir)

    player_count = _resolve_player_count(game_id, raw)
    options = _resolve_options(game_id, raw)
    factions = _resolve_seat_order(game_id, raw)
    score_tiles = _resolve_score_tiles(game_id, raw)
    bonus_tiles = _resolve_bonus_tiles(game_id, raw, player_count)
    dropped_at_row = _resolve_dropped_at_row(raw)

    return GameSetup(
        game_id=game_id,
        options=options,
        factions=factions,
        score_tiles=score_tiles,
        bonus_tiles=bonus_tiles,
        player_count=player_count,
        dropped_at_row=dropped_at_row,
    )
