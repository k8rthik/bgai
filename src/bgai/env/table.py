"""Seeded ``GameSetup`` construction for 2-5 player tables.

``arena.setup_factory.fresh_setup`` already does this for exactly four
players; this is the n-player generalization TM-Env needs, kept here
rather than by widening that function so the arena's pinned seeds stay
bit-identical.

Same deliberate simplifications as ``setup_factory``: uniform 6-of-SCORE1..8
sampling (no "no spade tile in rounds 5/6" lobby rule), uniform
``player_count + 3`` draw from BON1..9, default ``GameOptions``. SCORE9 and
BON10 are option-gated by ``tiles.TILE_OPTIONS`` and excluded under
default options.
"""

from __future__ import annotations

import random

from bgai.engine.tm.factions_data import FACTIONS
from bgai.engine.tm.setup import GameOptions, GameSetup
from bgai.engine.tm.tiles import BONUS_TILES, SCORE_TILES, TILE_OPTIONS
from bgai.env.config import (
    BONUS_POOL_EXTRA,
    SCORE_TILES_PER_GAME,
    EnvConfig,
    SetupSource,
)


def _unoptioned(pool: dict[str, object]) -> list[str]:
    return sorted(t for t in pool if TILE_OPTIONS.get(t) is None)


def _factions_by_color() -> dict[str, list[str]]:
    by_color: dict[str, list[str]] = {}
    for name, data in sorted(FACTIONS.items()):
        by_color.setdefault(data.color, []).append(name)
    return by_color


def draw_factions(rng: random.Random, player_count: int) -> tuple[str, ...]:
    """A colour-legal lineup: one faction per distinct colour.

    Terra Mystica pairs its 14 factions into 7 colours and no two players
    may share a colour, which is why this draws colours first.
    """
    by_color = _factions_by_color()
    if player_count > len(by_color):
        raise ValueError(
            f"player_count {player_count} exceeds the {len(by_color)} faction colours"
        )
    colors = rng.sample(sorted(by_color), player_count)
    return tuple(rng.choice(by_color[c]) for c in colors)


def synthetic_setup(seed: int, config: EnvConfig) -> GameSetup:
    """A fresh seeded table for ``config.player_count`` players."""
    rng = random.Random(seed)
    factions = config.factions or draw_factions(rng, config.player_count)
    score_ids = rng.sample(_unoptioned(SCORE_TILES), SCORE_TILES_PER_GAME)
    bonus_ids = tuple(
        sorted(
            rng.sample(
                _unoptioned(BONUS_TILES), config.player_count + BONUS_POOL_EXTRA
            )
        )
    )
    return GameSetup(
        game_id=f"tmenv_{config.player_count}p_{seed}",
        options=GameOptions(),
        factions=factions,
        score_tiles=tuple(SCORE_TILES[t] for t in score_ids),
        bonus_tiles=bonus_ids,
        player_count=config.player_count,
    )


def corpus_setup(seed: int) -> GameSetup:
    """A real Div 1-3 game's setup, drop history cleared (4 players).

    Imported lazily: ``arena.setups`` reads ``data/datasets/moves.parquet``,
    which is gitignored, so importing it at module load would make
    ``bgai.env`` unimportable on a fresh clone.
    """
    from bgai.arena.setups import sample_setup

    return sample_setup(random.Random(seed))


def build_setup(seed: int, config: EnvConfig) -> GameSetup:
    """The table for one episode, from ``config.setup_source``."""
    if config.setup_source is SetupSource.CORPUS:
        setup = corpus_setup(seed)
        if config.factions is not None and setup.factions != config.factions:
            raise ValueError(
                "config.factions cannot be combined with setup_source=CORPUS: "
                "a corpus table's lineup is part of the sampled game"
            )
        return setup
    return synthetic_setup(seed, config)
