"""Tests for tmtour API response parsing (network-free)."""

import pytest

from bgai.data.tmtour import GameName, parse_game_name, qualifying_games

SAMPLE_GAMES = [
    {
        "gameNumber": 1,
        "name": "4pLeague_S10_D1L1_G1",
        "finished": True,
        "players": [
            {"username": "c.love", "faction": "engineers", "score": 147, "rank": 1, "seat": 1},
            {"username": "a", "faction": "witches", "score": 120, "rank": 2, "seat": 2},
            {"username": "b", "faction": "nomads", "score": 110, "rank": 3, "seat": 3},
            {"username": "d", "faction": "giants", "score": 90, "rank": 4, "seat": 4},
        ],
    },
    {
        "gameNumber": 2,
        "name": "4pLeague_S10_D4L2_G2",  # division 4 — excluded
        "finished": True,
        "players": [],
    },
    {
        "gameNumber": 3,
        "name": "4pLeague_S10_D2L3_G3",  # unfinished — excluded
        "finished": False,
        "players": [],
    },
    {
        "gameNumber": 4,
        "name": "weird_name_G4",  # unparseable — excluded, reported
        "finished": True,
        "players": [],
    },
]


def test_parse_game_name_standard() -> None:
    parsed = parse_game_name("4pLeague_S74_D1L12_G7")
    assert parsed == GameName(season=74, division=1, league=12, game_number=7)


def test_parse_game_name_rejects_other_formats() -> None:
    assert parse_game_name("SomeOtherGame_123") is None
    assert parse_game_name("4pLeague_S74_D1L12") is None
    assert parse_game_name("") is None


@pytest.mark.parametrize("max_division", [1, 2, 3])
def test_qualifying_games_filters_division_and_finished(max_division: int) -> None:
    result = qualifying_games(SAMPLE_GAMES, max_division=max_division)
    kept = result.games
    assert all(g.name_parts.division <= max_division for g in kept)
    assert all(g.finished for g in kept)


def test_qualifying_games_keeps_expected_and_reports_unparseable() -> None:
    result = qualifying_games(SAMPLE_GAMES, max_division=3)
    assert [g.game_id for g in result.games] == ["4pLeague_S10_D1L1_G1"]
    assert result.unparseable == ["weird_name_G4"]
    game = result.games[0]
    assert game.name_parts.season == 10
    assert [p.username for p in game.players][0] == "c.love"
    assert [p.faction for p in game.players] == ["engineers", "witches", "nomads", "giants"]
