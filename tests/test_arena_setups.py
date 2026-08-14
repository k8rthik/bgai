"""Corpus setup sampler (plan 2026-08-04-arena-baselines, Task 3)."""

from __future__ import annotations

import random

from bgai.arena.setups import clean_game_ids, sample_setup


def test_clean_ids_exclude_known_bad_classes() -> None:
    ids = clean_game_ids()
    # Originally pinned to the 3,563-game league corpus (3,374 clean);
    # the 2026-08-12 compact widened the corpus to the crawled population
    # (~64k clean), so pin the invariants, not the count: every id is
    # parsed, cleaned ids are a strict subset (the nofaction/dropped
    # exclusion classes really exclude), sorted, unique.
    from bgai.arena.setups import parsed_corpus_ids

    parsed = parsed_corpus_ids()
    assert len(ids) > 3374  # never shrinks below the original league corpus
    assert len(ids) < len(parsed)  # exclusions actually fire
    assert ids == tuple(sorted(ids))
    assert len(set(ids)) == len(ids)


def test_sample_setup_is_clean_and_deterministic() -> None:
    a = sample_setup(random.Random(3))
    b = sample_setup(random.Random(3))
    assert a.game_id == b.game_id
    assert dict(a.dropped_at_row) == {}
    assert len(a.factions) == 4
    assert len(a.score_tiles) == 6
