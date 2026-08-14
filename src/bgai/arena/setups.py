"""Corpus-sampled arena setups (master plan, Phase 4 decision 2026-08-04).

Every arena table reuses a real Div 1-3 game's fixed configuration --
score tiles, bonus-tile pool, faction lineup, seat order, options -- via
the replay-validated ``load_setup``, with the drop history cleared
(arena games have four live seats). No synthetic setup generator: human
lineups are pre-vetted for coherence, and evaluation stays on the
Phase 5 training distribution.
"""

from __future__ import annotations

import json
import random
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import polars as pl

from bgai.engine.tm.setup import GameSetup, load_setup

_DEFAULT_RAW_DIR = Path("data/raw/games")
_DEFAULT_MOVES = Path("data/datasets/moves.parquet")
_CLEAN_IDS_CACHE = Path("data/datasets/clean_setup_ids.json")


def _moves_stamp(moves_path: Path = _DEFAULT_MOVES) -> list[int]:
    """Identity of the parsed corpus the cleaned-id list derives from."""
    st = moves_path.stat()
    return [int(st.st_mtime), st.st_size]


@lru_cache(maxsize=1)
def parsed_corpus_ids(moves_path: Path = _DEFAULT_MOVES) -> frozenset[str]:
    """Games that have parsed ledger data, i.e. the corpus proper.

    ``data/raw/games/`` is a *cache*, not a corpus definition: the
    population crawl adds tens of thousands of games there long before any
    of them are parsed into ``moves.parquet``. Deriving a game list from
    the directory therefore makes every seeded arena run and every dataset
    build silently depend on how far the crawl happened to get, which
    breaks reproducibility of the pinned Phase 4/5 gates. The parsed
    dataset is the stable definition, so everything downstream keys off it.
    """
    return frozenset(pl.read_parquet(moves_path, columns=["game_id"])["game_id"].unique())


@lru_cache(maxsize=4)
def clean_game_ids(raw_dir: Path = _DEFAULT_RAW_DIR) -> tuple[str, ...]:
    """Sorted ids of every *parsed* game that loads and never dropped a
    faction. ``load_setup`` raising ``ValueError`` (the 10 ``nofaction*``
    incomplete crawls) and non-empty ``dropped_at_row`` (179 AFK-drop
    games) are the two exclusion classes; games present on disk but not
    yet in ``moves.parquet`` are a third (see ``parsed_corpus_ids``).
    """
    # Disk cache (default corpus only): the cleaned list is a pure
    # function of the parsed corpus, but computing it gzip+JSON-decodes
    # all ~76k raw games (~4 min) -- and every fresh self-play worker in
    # every iteration's Pool paid that, ~30% of generation wall time
    # (2026-08-14 profile). Stamped against moves.parquet, the corpus
    # definition, so a recompact invalidates it.
    use_disk = raw_dir == _DEFAULT_RAW_DIR and _DEFAULT_MOVES.exists()
    if use_disk and _CLEAN_IDS_CACHE.exists():
        try:
            data = json.loads(_CLEAN_IDS_CACHE.read_text())
            if data.get("stamp") == _moves_stamp():
                return tuple(data["ids"])
        except (ValueError, OSError):
            pass  # unreadable cache -> recompute below

    parsed = parsed_corpus_ids()
    ids: list[str] = []
    for path in sorted(raw_dir.glob("*.json.gz")):
        game_id = path.name.removesuffix(".json.gz")
        if game_id not in parsed:
            continue
        try:
            setup = load_setup(game_id, raw_dir)
        except ValueError:
            continue
        if setup.dropped_at_row:
            continue
        ids.append(game_id)

    if use_disk:
        _CLEAN_IDS_CACHE.write_text(json.dumps({"stamp": _moves_stamp(), "ids": ids}))
    return tuple(ids)


def sample_setup(rng: random.Random, raw_dir: Path = _DEFAULT_RAW_DIR) -> GameSetup:
    """A uniformly drawn clean setup, drop history cleared."""
    ids = clean_game_ids(raw_dir)
    game_id = ids[rng.randrange(len(ids))]
    setup = load_setup(game_id, raw_dir)
    return replace(setup, dropped_at_row={})
