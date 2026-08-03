"""Game options + per-game setup loader tests."""

from __future__ import annotations

import copy
import gzip
import json
from pathlib import Path
from typing import Any

import pytest

from bgai.engine.tm.setup import GameOptions, load_setup

_VALID_SCORE_TILE: dict[str, Any] = {
    "vp": {"TP": 3},
    "cult": "WATER",
    "vp_mode": "build",
    "income": {"SPADE": 1},
    "req": 4,
    "income_display": "4 WATER -> 1 SPADE",
    "vp_display": "TP >> 3",
}


def _valid_raw(player_count: int = 2) -> dict[str, Any]:
    """A minimal raw game dict that satisfies every boundary check in setup.py."""
    bonus_count = player_count + 3
    return {
        "metadata": {"player_count": player_count},
        "options": {"strict-leech": 1},
        "factions": {
            "nomads": {"start_order": 1},
            "darklings": {"start_order": 2},
        },
        "score_tiles": [copy.deepcopy(_VALID_SCORE_TILE) for _ in range(6)],
        "pool": {f"BON{i}": {} for i in range(1, bonus_count + 1)},
    }


def _write_game(tmp_path: Path, game_id: str, raw: dict[str, Any]) -> Path:
    """Write ``raw`` as ``<tmp_path>/<game_id>.json.gz`` and return the dir."""
    path = tmp_path / f"{game_id}.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(raw, f)
    return tmp_path


def test_options_from_csv() -> None:
    opts = GameOptions.from_csv(
        "email-notify,errata-cultist-power,strict-leech,mini-expansion-1"
    )
    assert opts.errata_cultist_power and opts.strict_leech and opts.mini_expansion_1
    assert not opts.variable_turn_order


def test_load_reference_game() -> None:
    s = load_setup("4pLeague_S10_D1L1_G1")
    assert s.player_count == 4
    assert s.factions == ("engineers", "darklings", "nomads", "mermaids")
    assert len(s.score_tiles) == 6
    assert s.score_tiles[0].cult == "WATER" and s.score_tiles[0].req == 4
    assert set(s.bonus_tiles) == {"BON1", "BON3", "BON4", "BON5", "BON7", "BON9", "BON10"}
    assert s.options.strict_leech


def test_minimal_valid_game_loads(tmp_path: Path) -> None:
    """Sanity check on the synthetic fixture itself before mutating it below."""
    game_id = "synthetic_ok"
    raw_dir = _write_game(tmp_path, game_id, _valid_raw())
    s = load_setup(game_id, raw_dir=raw_dir)
    assert s.player_count == 2
    assert s.factions == ("nomads", "darklings")
    assert len(s.score_tiles) == 6
    assert s.bonus_tiles == ("BON1", "BON2", "BON3", "BON4", "BON5")


def test_factions_not_a_dict_raises_with_game_id(tmp_path: Path) -> None:
    game_id = "synthetic_bad_factions_type"
    raw = _valid_raw()
    raw["factions"] = ["nomads", "darklings"]
    raw_dir = _write_game(tmp_path, game_id, raw)

    with pytest.raises(ValueError) as exc_info:
        load_setup(game_id, raw_dir=raw_dir)
    assert game_id in str(exc_info.value)


def test_malformed_faction_entry_raises_with_game_id(tmp_path: Path) -> None:
    game_id = "synthetic_bad_faction_entry"
    raw = _valid_raw()
    raw["factions"]["nomads"] = "not-a-dict"
    raw_dir = _write_game(tmp_path, game_id, raw)

    with pytest.raises(ValueError) as exc_info:
        load_setup(game_id, raw_dir=raw_dir)
    assert game_id in str(exc_info.value)
    assert "factions" in str(exc_info.value)


def test_non_int_start_order_raises_with_game_id(tmp_path: Path) -> None:
    game_id = "synthetic_bad_start_order"
    raw = _valid_raw()
    raw["factions"]["nomads"]["start_order"] = "1"  # str, not int
    raw_dir = _write_game(tmp_path, game_id, raw)

    with pytest.raises(ValueError) as exc_info:
        load_setup(game_id, raw_dir=raw_dir)
    assert game_id in str(exc_info.value)
    assert "start_order" in str(exc_info.value)


def test_wrong_score_tile_count_raises_with_game_id(tmp_path: Path) -> None:
    game_id = "synthetic_bad_score_tile_count"
    raw = _valid_raw()
    raw["score_tiles"] = raw["score_tiles"][:5]  # 5, not 6
    raw_dir = _write_game(tmp_path, game_id, raw)

    with pytest.raises(ValueError) as exc_info:
        load_setup(game_id, raw_dir=raw_dir)
    assert game_id in str(exc_info.value)
    assert "6 score tiles" in str(exc_info.value)


def test_invalid_score_tile_entry_raises_with_game_id(tmp_path: Path) -> None:
    """A ScoringTile.from_snellman failure must be wrapped with the game_id."""
    game_id = "synthetic_bad_score_tile_entry"
    raw = _valid_raw()
    del raw["score_tiles"][0]["cult"]  # required key for ScoringTile.from_snellman
    raw_dir = _write_game(tmp_path, game_id, raw)

    with pytest.raises(ValueError) as exc_info:
        load_setup(game_id, raw_dir=raw_dir)
    assert game_id in str(exc_info.value)
    assert "score_tiles[0]" in str(exc_info.value)


def test_non_dict_pool_raises_with_game_id(tmp_path: Path) -> None:
    game_id = "synthetic_bad_pool_type"
    raw = _valid_raw()
    raw["pool"] = ["BON1", "BON2"]
    raw_dir = _write_game(tmp_path, game_id, raw)

    with pytest.raises(ValueError) as exc_info:
        load_setup(game_id, raw_dir=raw_dir)
    assert game_id in str(exc_info.value)
    assert "pool" in str(exc_info.value)


def test_wrong_bonus_tile_count_raises_with_game_id(tmp_path: Path) -> None:
    game_id = "synthetic_bad_bonus_tile_count"
    raw = _valid_raw()
    del raw["pool"]["BON5"]  # player_count=2 needs 5 BON tiles, now only 4
    raw_dir = _write_game(tmp_path, game_id, raw)

    with pytest.raises(ValueError) as exc_info:
        load_setup(game_id, raw_dir=raw_dir)
    assert game_id in str(exc_info.value)
    assert "bonus tiles" in str(exc_info.value)
