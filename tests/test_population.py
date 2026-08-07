"""Population index + self-computed ratings from the permitted events feeds."""

from __future__ import annotations

import polars as pl

from pathlib import Path

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


def test_population_order_interleaves_strength_so_any_prefix_is_balanced() -> None:
    """A 24h crawl gets interrupted; every prefix must still span the skill range."""
    from bgai.data.crawl import population_order

    seats = pl.DataFrame(
        [
            {"game_id": f"g{n:03d}", "base_map": BASE_MAP, "player_count": 4,
             "options": [], "player": f"p{n:03d}", "place": 0, "last_update": "x"}
            for n in range(100)
        ]
    )
    # player p000 weakest .. p099 strongest, so game strength tracks its index
    ratings = pl.DataFrame(
        {"player": [f"p{n:03d}" for n in range(100)],
         "conservative": [float(n) for n in range(100)]}
    )
    order = population_order(seats, ratings)
    assert len(order) == 100 and len(set(order)) == 100

    # the first 10 picks should touch every decile, not just the weakest games
    first_ten = {int(g.removeprefix("g")) // 10 for g in order[:10]}
    assert len(first_ten) == 10, f"prefix clustered in deciles {sorted(first_ten)}"


def test_crawl_loop_paces_on_request_starts_not_response_ends(monkeypatch) -> None:
    """The rate promise is >= 1 s between request *starts*, whatever latency does.

    A fixed post-response sleep would let a fast server push the real rate past
    the ceiling; pacing start-to-start must not.
    """
    import time as time_module

    from bgai.data import crawl as crawl_module

    clock = {"now": 0.0}
    starts: list[float] = []

    monkeypatch.setattr(crawl_module.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(
        crawl_module.time, "sleep", lambda s: clock.__setitem__("now", clock["now"] + s)
    )

    def fake_fetch(_client, game_id):
        starts.append(clock["now"])
        clock["now"] += 0.05  # a very fast server: 50 ms responses
        return b'{"ledger": [1], "finished": 1}'

    monkeypatch.setattr(crawl_module, "_fetch_game", fake_fetch)
    monkeypatch.setattr(crawl_module.httpx, "Client", lambda **_: _NullClient())
    monkeypatch.setattr(crawl_module.gzip, "open", lambda *a, **k: _NullFile())

    crawl_module._crawl_loop([f"g{i}" for i in range(5)], Path("/tmp"), 1.0)

    gaps = [b - a for a, b in zip(starts, starts[1:], strict=False)]
    assert gaps, "no requests issued"
    assert all(gap >= 1.0 - 1e-9 for gap in gaps), f"rate ceiling breached: {gaps}"
    assert time_module is not None  # keep the import meaningful for linters


class _NullClient:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


class _NullFile:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def write(self, _data):
        return None
