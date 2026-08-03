"""Tests for crawl ordering and cache-path logic (network-free)."""

import polars as pl

from bgai.data.crawl import cache_path, crawl_order


def test_crawl_order_div_first_then_newest_season() -> None:
    games = pl.DataFrame(
        {
            "game_id": ["a", "b", "c", "d"],
            "division": [3, 1, 2, 1],
            "season": [74, 10, 74, 74],
            "league": [1, 1, 1, 1],
            "game_number": [1, 1, 1, 1],
        }
    )
    ordered = crawl_order(games)["game_id"].to_list()
    # Div 1 before 2 before 3; within a division, newest season first.
    assert ordered == ["d", "b", "c", "a"]


def test_cache_path_is_gzipped_json_named_by_game_id(tmp_path) -> None:
    path = cache_path(tmp_path, "4pLeague_S74_D1L1_G1")
    assert path.name == "4pLeague_S74_D1L1_G1.json.gz"
    assert path.parent == tmp_path
