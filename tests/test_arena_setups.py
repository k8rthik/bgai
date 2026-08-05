"""Corpus setup sampler (plan 2026-08-04-arena-baselines, Task 3)."""

from __future__ import annotations

import random

from bgai.arena.setups import clean_game_ids, sample_setup


def test_clean_ids_exclude_known_bad_classes() -> None:
    ids = clean_game_ids()
    # 3,563 raw games minus 10 nofaction load failures minus 179 dropped-faction games.
    assert len(ids) == 3563 - 10 - 179
    assert ids == tuple(sorted(ids))


def test_sample_setup_is_clean_and_deterministic() -> None:
    a = sample_setup(random.Random(3))
    b = sample_setup(random.Random(3))
    assert a.game_id == b.game_id
    assert dict(a.dropped_at_row) == {}
    assert len(a.factions) == 4
    assert len(a.score_tiles) == 6
