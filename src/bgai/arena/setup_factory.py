"""Seeded fresh-game GameSetup factory for arena/MCP play.

Simplifications vs. snellman lobby rules (documented, deliberate):
uniform 6-of-SCORE1..8 sampling (no "no-spade-tile in rounds 5/6" rule),
uniform 7-of-BON1..9 pool, default GameOptions (all flags off). SCORE9
(temple-scoring-tile) and BON10 (shipping-bonus) are option-gated per
tiles.TILE_OPTIONS and excluded under default options.
"""

from __future__ import annotations

import random

from bgai.engine.tm.factions_data import FACTIONS
from bgai.engine.tm.setup import GameOptions, GameSetup
from bgai.engine.tm.tiles import BONUS_TILES, SCORE_TILES, TILE_OPTIONS


def fresh_setup(seed: int, factions: tuple[str, str, str, str] | None = None) -> GameSetup:
    rng = random.Random(seed)
    if factions is None:
        by_color: dict[str, list[str]] = {}
        for name, data in sorted(FACTIONS.items()):
            by_color.setdefault(data.color, []).append(name)
        colors = rng.sample(sorted(by_color), 4)
        factions = tuple(rng.choice(by_color[c]) for c in colors)
    score_pool = sorted(t for t in SCORE_TILES if TILE_OPTIONS.get(t) is None)
    bonus_pool = sorted(t for t in BONUS_TILES if TILE_OPTIONS.get(t) is None)
    score_ids = rng.sample(score_pool, 6)
    bonus_ids = tuple(sorted(rng.sample(bonus_pool, 7)))
    return GameSetup(
        game_id=f"arena_{seed}",
        options=GameOptions(),
        factions=factions,
        score_tiles=tuple(SCORE_TILES[t] for t in score_ids),
        bonus_tiles=bonus_ids,
        player_count=4,
    )
