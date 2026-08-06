"""Population index + self-computed ratings from the permitted events feeds."""

from __future__ import annotations

import polars as pl

from bgai.data.player_ratings import BASE_MAP, playable, rate
from bgai.data.population import _game_rows


def _feed_game(game_id: str, scores: dict[str, int], options: list[str] | None = None) -> dict:
    """A minimal events-feed entry shaped like the real /data/events payload."""
    factions = [
        {"player": f"p{i}", "faction": faction} for i, faction in enumerate(scores)
    ]
    return {
        "game": game_id,
        "base_map": BASE_MAP,
        "player_count": len(scores),
        "last_update": "2024-06-01 00:00:00",
        "factions": factions,
        "events": {
            "global": {f"option-{name}": 1 for name in (options or [])},
            "faction": {
                faction: {"vp": {"round": {"all": vp}}} for faction, vp in scores.items()
            },
        },
    }


def test_game_rows_extracts_one_row_per_seat_with_scores() -> None:
    rows = _game_rows(_feed_game("g1", {"witches": 139, "darklings": 141}), "2024-06")
    assert [r["player"] for r in rows] == ["p0", "p1"]
    assert [r["vp"] for r in rows] == [139, 141]
    assert all(r["game_id"] == "g1" and r["month"] == "2024-06" for r in rows)


def test_game_rows_drops_games_where_any_seat_lacks_a_score() -> None:
    """A partially-scored game would bias placements, so it is skipped whole."""
    game = _feed_game("g2", {"witches": 100, "darklings": 90})
    del game["events"]["faction"]["darklings"]["vp"]
    assert _game_rows(game, "2024-06") == []


def test_options_are_normalised_off_the_global_event_keys() -> None:
    rows = _game_rows(
        _feed_game("g3", {"witches": 1, "darklings": 2}, options=["strict-leech"]),
        "2024-06",
    )
    assert rows[0]["options"] == ["strict-leech"]


def test_playable_excludes_fire_and_ice_and_other_maps() -> None:
    """The engine is base-game/14 factions, so Fire & Ice games are out of scope."""
    seats = pl.DataFrame(
        [
            {"game_id": "ok", "base_map": BASE_MAP, "player_count": 4,
             "options": ["strict-leech"], "player": "a", "place": 0, "last_update": "x"},
            {"game_id": "fi", "base_map": BASE_MAP, "player_count": 4,
             "options": ["fire-and-ice-factions/ice"], "player": "b", "place": 0,
             "last_update": "x"},
            {"game_id": "map", "base_map": "other", "player_count": 4,
             "options": [], "player": "c", "place": 0, "last_update": "x"},
        ]
    )
    assert set(playable(seats)["game_id"]) == {"ok"}


def test_rate_orders_players_by_observed_results() -> None:
    """A player who always wins outranks one who always loses."""
    seats = pl.DataFrame(
        [
            {"game_id": f"g{n}", "player": p, "place": place, "last_update": f"2024-06-{n:02d}"}
            for n in range(1, 11)
            for p, place in (("winner", 0), ("loser", 1))
        ]
    )
    table = rate(seats)
    ranked = table.sort("conservative", descending=True)["player"].to_list()
    assert ranked[0] == "winner"
    assert table.filter(pl.col("player") == "winner")["games"].item() == 10
