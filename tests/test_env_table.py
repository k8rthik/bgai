"""Seeded 2-5 player table construction."""

from __future__ import annotations

import random

import pytest

from bgai.engine.tm.factions_data import FACTIONS
from bgai.engine.tm.tiles import TILE_OPTIONS
from bgai.env.config import (
    BONUS_POOL_EXTRA,
    MAX_PLAYERS,
    MIN_PLAYERS,
    SCORE_TILES_PER_GAME,
    EnvConfig,
    SetupSource,
)
from bgai.env.table import build_setup, draw_factions, synthetic_setup

COUNTS = list(range(MIN_PLAYERS, MAX_PLAYERS + 1))


@pytest.mark.parametrize("count", COUNTS)
def test_setup_shape_matches_the_rules(count: int) -> None:
    setup = synthetic_setup(42, EnvConfig(player_count=count))
    assert len(setup.factions) == count
    assert setup.player_count == count
    assert len(setup.score_tiles) == SCORE_TILES_PER_GAME
    assert len(setup.bonus_tiles) == count + BONUS_POOL_EXTRA
    assert setup.bonus_tiles == tuple(sorted(setup.bonus_tiles))


@pytest.mark.parametrize("count", COUNTS)
def test_lineups_are_colour_legal(count: int) -> None:
    for seed in range(40):
        factions = draw_factions(random.Random(seed), count)
        colours = [FACTIONS[f].color for f in factions]
        assert len(set(colours)) == count, f"colour clash at seed {seed}: {factions}"
        assert len(set(factions)) == count


@pytest.mark.parametrize("count", COUNTS)
def test_same_seed_same_table(count: int) -> None:
    config = EnvConfig(player_count=count)
    assert synthetic_setup(7, config) == synthetic_setup(7, config)


def test_different_seeds_give_different_tables() -> None:
    config = EnvConfig(player_count=4)
    tables = {synthetic_setup(seed, config).factions for seed in range(20)}
    assert len(tables) > 1


def test_option_gated_tiles_are_excluded_under_default_options() -> None:
    """SCORE9 and BON10 need explicit options; a default table must not
    draw them or the engine would score a tile the rules did not enable.
    """
    gated = {t for t, option in TILE_OPTIONS.items() if option is not None}
    for seed in range(30):
        setup = synthetic_setup(seed, EnvConfig(player_count=4))
        assert not set(setup.bonus_tiles) & gated


def test_explicit_lineup_is_honoured() -> None:
    config = EnvConfig(player_count=3, factions=("witches", "nomads", "dwarves"))
    assert synthetic_setup(1, config).factions == ("witches", "nomads", "dwarves")


def test_draw_factions_rejects_more_players_than_colours() -> None:
    with pytest.raises(ValueError, match="faction colours"):
        draw_factions(random.Random(0), 8)


def test_build_setup_routes_on_source() -> None:
    config = EnvConfig(player_count=5)
    assert build_setup(3, config) == synthetic_setup(3, config)


@pytest.mark.slow
def test_corpus_source_loads_a_real_game() -> None:
    """Needs the gitignored data/ tree, so it rides with the slow marker."""
    setup = build_setup(11, EnvConfig(setup_source=SetupSource.CORPUS))
    assert setup.player_count == 4
    assert setup.dropped_at_row == {}
    assert not setup.game_id.startswith("tmenv_")
