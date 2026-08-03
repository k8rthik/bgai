"""Tile data tests: BON/FAV/TW/ACT/SCORE tables against Constants.pm and corpus."""

import gzip
import json
from pathlib import Path

from bgai.engine.tm.tiles import (
    BONUS_TILES,
    FAVOR_TILES,
    POWER_ACTIONS,
    SCORE_TILES,
    TOWN_TILES,
    ScoringTile,
)


def test_tile_id_coverage() -> None:
    assert set(BONUS_TILES) >= {f"BON{i}" for i in range(1, 11)}
    assert set(FAVOR_TILES) == {f"FAV{i}" for i in range(1, 13)}
    assert set(TOWN_TILES) >= {f"TW{i}" for i in range(1, 9)}
    assert set(POWER_ACTIONS) == {f"ACT{i}" for i in range(1, 7)}
    assert set(SCORE_TILES) == {f"SCORE{i}" for i in range(1, 10)}


def test_known_tile_values() -> None:
    # Anchor literals hand-copied from Constants.pm (independent of tiles.py):
    #   ACT4 => { cost => { PW => 4 }, gain => { C => 7 } }               (line 60)
    #   FAV1 => { gain => { FIRE => 3 }, income => {}, count => 1 }       (line 143)
    #   BON3 => { income => { C => 6 } }                                  (line 129)
    assert POWER_ACTIONS["ACT4"].cost_power == 4
    assert POWER_ACTIONS["ACT4"].gain == {"C": 7}
    assert FAVOR_TILES["FAV1"].cult == "FIRE" and FAVOR_TILES["FAV1"].steps == 3
    assert BONUS_TILES["BON3"].income == {"C": 6}


def test_bonus_tile_special_action_and_passive() -> None:
    # BON1 => %actions entry gain => { SPADE => 1 } (line 101-102)
    assert BONUS_TILES["BON1"].special_action == {"SPADE": 1}
    # BON2 => %actions entry gain => { CULT => 1 } (line 103)
    assert BONUS_TILES["BON2"].special_action == {"CULT": 1}
    # BON4 => special => { ship => 1 } (line 130)
    assert BONUS_TILES["BON4"].passive == {"ship": 1}
    # BON6 pass_vp => { SA => [0, 4], SH => [0, 4] } (lines 132-133)
    assert dict(BONUS_TILES["BON6"].pass_vp) == {"SA": 4, "SH": 4}
    # BON10 is shipping-bonus-option gated.
    assert BONUS_TILES["BON10"].pass_vp == (("ship", 3),)


def test_favor_tile_special_action_and_vp() -> None:
    # FAV6 => %actions entry gain => { CULT => 1 } (line 104)
    assert FAVOR_TILES["FAV6"].special_action == {"CULT": 1}
    # FAV5 => gain => { FIRE => 2, TOWN_SIZE => -1 } (line 148)
    assert FAVOR_TILES["FAV5"].passive == {"TOWN_SIZE": -1}
    # FAV10 => vp => { TP => 3 } (line 154)
    assert FAVOR_TILES["FAV10"].vp == {"TP": 3}
    # FAV12 => pass_vp => { TP => [0, 2, 3, 3, 4] } (line 157) -- non-linear.
    assert FAVOR_TILES["FAV12"].pass_vp == {"TP": (0, 2, 3, 3, 4)}


def test_favor_pool_counts() -> None:
    from bgai.engine.tm.tiles import FAVOR_POOL_COUNTS

    for i in range(1, 5):
        assert FAVOR_POOL_COUNTS[f"FAV{i}"] == 1
    for i in range(5, 13):
        assert FAVOR_POOL_COUNTS[f"FAV{i}"] == 3


def test_town_tile_values_and_pool_counts() -> None:
    from bgai.engine.tm.tiles import TOWN_POOL_COUNTS

    # TW1 => { gain => { KEY => 1, VP => 5, C => 6 } } (line 215)
    assert TOWN_TILES["TW1"].vp == 5
    assert TOWN_TILES["TW1"].gain == {"KEY": 1, "C": 6}
    # TW6/7/8 require mini-expansion-1; base pool count is 2, TW6/TW8 are 1-offs.
    assert TOWN_POOL_COUNTS["TW1"] == 2
    assert TOWN_POOL_COUNTS["TW6"] == 1
    assert TOWN_POOL_COUNTS["TW7"] == 2
    assert TOWN_POOL_COUNTS["TW8"] == 1


def test_option_tile_mapping() -> None:
    from bgai.engine.tm.tiles import TILE_OPTIONS

    assert TILE_OPTIONS["BON10"] == "shipping-bonus"
    assert TILE_OPTIONS["SCORE9"] == "temple-scoring-tile"
    assert TILE_OPTIONS["TW6"] == "mini-expansion-1"
    assert TILE_OPTIONS["TW7"] == "mini-expansion-1"
    assert TILE_OPTIONS["TW8"] == "mini-expansion-1"


def test_scoring_tile_from_snellman() -> None:
    raw = {
        "vp": {"TP": 3},
        "cult": "WATER",
        "vp_mode": "build",
        "income": {"SPADE": 1},
        "req": 4,
        "income_display": "4 WATER -> 1 SPADE",
        "vp_display": "TP >> 3",
    }
    t = ScoringTile.from_snellman(raw)
    assert t.cult == "WATER" and t.req == 4
    assert t.vp_mode == "build" and t.vp == (("TP", 3),)
    assert t.cult_income == (("SPADE", 1),)


def test_score_tiles_canonical_table() -> None:
    # SCORE9 is the temple-scoring-tile option round; SCORE1 is EARTH/gain.
    assert SCORE_TILES["SCORE9"].cult == "CULT_P"
    assert SCORE_TILES["SCORE9"].req == 1
    assert SCORE_TILES["SCORE1"].vp == (("SPADE", 2),)
    assert SCORE_TILES["SCORE1"].vp_mode == "gain"


def test_corpus_pool_coverage() -> None:
    """Every BON/TW/FAV/ACT id appearing in a 50-game sample is defined."""
    games = sorted(Path("data/raw/games").glob("*.json.gz"))[:50]
    known = set(BONUS_TILES) | set(FAVOR_TILES) | set(TOWN_TILES) | set(POWER_ACTIONS)
    for path in games:
        with gzip.open(path) as f:
            pool = json.load(f)["pool"]
        ids = {k for k in pool if k[:3] in ("BON", "FAV", "ACT") or k[:2] == "TW"}
        assert ids <= known, f"{path.name}: unknown tiles {ids - known}"


def test_corpus_score_tiles_parse_via_from_snellman() -> None:
    """Every game's live score_tiles list round-trips through ScoringTile.from_snellman."""
    games = sorted(Path("data/raw/games").glob("*.json.gz"))[:50]
    for path in games:
        with gzip.open(path) as f:
            data = json.load(f)
        for raw in data.get("score_tiles") or []:
            tile = ScoringTile.from_snellman(raw)
            assert tile.cult
            assert tile.req >= 1
