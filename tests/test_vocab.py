"""Training vocabularies (plan 2026-08-04-imitation-phase5, Task 1)."""

from __future__ import annotations

import polars as pl

from bgai.arena.sim import canonical_moves
from bgai.engine.tm.legal_shared import cmd
from bgai.training.vocab import (
    COLOR_INDEX,
    COLORS,
    ENCODING_VERSION,
    FACTION_INDEX,
    FACTION_NAMES,
    HEX_INDEX,
    HEXES,
    TILE_INDEX,
    TILES,
    VERB_INDEX,
    VERBS,
    normalize_color,
)


def test_hex_vocabulary_covers_full_board() -> None:
    assert len(HEXES) == 113
    assert HEXES == tuple(sorted(HEXES))
    assert all(HEXES[HEX_INDEX[h]] == h for h in HEXES)
    assert "A1" in HEX_INDEX and "r35" in HEX_INDEX


def test_verbs_and_tiles_cover_corpus_decisions() -> None:
    d = pl.read_parquet("data/datasets/moves.parquet").filter(pl.col("kind") == "decision")
    for verb in d["verb"].unique().to_list():
        assert verb in VERB_INDEX, verb
    for tile in d["tile"].unique().to_list():
        if tile is not None:
            assert tile in TILE_INDEX, tile


def test_color_normalization_handles_corpus_spelling() -> None:
    assert normalize_color("grey") == "gray"
    assert normalize_color("gray") == "gray"
    assert "grey" not in COLORS
    assert all(COLORS[COLOR_INDEX[c]] == c for c in COLORS)


def test_faction_vocab_and_version() -> None:
    assert len(FACTION_NAMES) == 14
    assert all(FACTION_NAMES[FACTION_INDEX[f]] == f for f in FACTION_NAMES)
    assert ENCODING_VERSION == 1


def test_canonical_moves_is_public_and_sorted() -> None:
    moves = (cmd("pass", tile="BON2"), cmd("build", loc="A1"), cmd("pass", tile="BON1"))
    out = canonical_moves(moves)
    assert [m.verb for m in out] == ["build", "pass", "pass"]
    assert [m.tile for m in out] == [None, "BON1", "BON2"]


def test_verbs_cover_every_generated_candidate_verb() -> None:
    """legal_moves must never offer a verb the encoders can't express --
    the extraction pipeline meets generated candidates, not just ledger
    rows (this caught gain_cult/lose_marker during Phase 5 Task 5)."""
    import re
    from pathlib import Path

    generated = set()
    for path in Path("src/bgai/engine/tm").glob("legal*.py"):
        generated |= set(re.findall(r'cmd\("([a-z_]+)"', path.read_text()))
    missing = sorted(generated - set(VERB_INDEX))
    assert missing == [], missing
