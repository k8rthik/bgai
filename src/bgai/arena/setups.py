"""Corpus-sampled arena setups (master plan, Phase 4 decision 2026-08-04).

Every arena table reuses a real Div 1-3 game's fixed configuration --
score tiles, bonus-tile pool, faction lineup, seat order, options -- via
the replay-validated ``load_setup``, with the drop history cleared
(arena games have four live seats). No synthetic setup generator: human
lineups are pre-vetted for coherence, and evaluation stays on the
Phase 5 training distribution.
"""

from __future__ import annotations

import random
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

from bgai.engine.tm.setup import GameSetup, load_setup

_DEFAULT_RAW_DIR = Path("data/raw/games")


@lru_cache(maxsize=4)
def clean_game_ids(raw_dir: Path = _DEFAULT_RAW_DIR) -> tuple[str, ...]:
    """Sorted ids of every raw game that loads and never dropped a
    faction. ``load_setup`` raising ``ValueError`` (the 10 ``nofaction*``
    incomplete crawls) and non-empty ``dropped_at_row`` (179 AFK-drop
    games) are the two exclusion classes. First call scans all raw
    games (~3.5k gzipped JSON files) once per process; cached after.
    """
    ids: list[str] = []
    for path in sorted(raw_dir.glob("*.json.gz")):
        game_id = path.name.removesuffix(".json.gz")
        try:
            setup = load_setup(game_id, raw_dir)
        except ValueError:
            continue
        if setup.dropped_at_row:
            continue
        ids.append(game_id)
    return tuple(ids)


def sample_setup(rng: random.Random, raw_dir: Path = _DEFAULT_RAW_DIR) -> GameSetup:
    """A uniformly drawn clean setup, drop history cleared."""
    ids = clean_game_ids(raw_dir)
    game_id = ids[rng.randrange(len(ids))]
    setup = load_setup(game_id, raw_dir)
    return replace(setup, dropped_at_row={})
