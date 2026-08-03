"""Board geometry tests, validated against real snellman game data when available."""

import glob
import gzip
from collections import Counter

import orjson
import pytest

from bgai.engine.tm.board import RIVER, base_board, hex_distance


def test_hex_and_river_counts() -> None:
    board = base_board()
    assert len(board.hexes) == 113
    rivers = [k for k, h in board.hexes.items() if h.color == RIVER]
    assert len(rivers) == 36
    assert len(board.land_hexes()) == 77


def test_each_terrain_color_has_eleven_hexes() -> None:
    board = base_board()
    counts = Counter(h.color for h in board.hexes.values() if h.color != RIVER)
    assert counts == {
        "yellow": 11, "brown": 11, "black": 11, "blue": 11,
        "green": 11, "gray": 11, "red": 11,
    }


def test_known_corner_hexes() -> None:
    board = base_board()
    a1 = board.hexes["A1"]
    assert (a1.row, a1.col, a1.color) == (0, 0, "brown")
    # Row I contains one river hex, so its last land hex is I12.
    i12 = board.hexes["I12"]
    assert (i12.row, i12.col, i12.color) == (8, 12, "red")


def test_adjacency_is_symmetric_and_bounded() -> None:
    board = base_board()
    for key, neighbors in board.adjacent.items():
        assert 2 <= len(neighbors) <= 6
        for n in neighbors:
            assert key in board.adjacent[n]


def test_a1_neighbors() -> None:
    board = base_board()
    # A1 (row 0, col 0): A2 to the right; B row is offset (odd row, cols 0..).
    assert board.adjacent["A1"] == frozenset({"A2", "B1"})


def test_distance_identity_and_neighbors() -> None:
    board = base_board()
    assert hex_distance("A1", "A1") == 0
    for n in board.adjacent["E5"]:
        assert hex_distance("E5", n) == 1


@pytest.mark.skipif(
    not glob.glob("data/raw/games/*.json.gz"), reason="no crawled games available"
)
def test_coordinates_match_real_snellman_game() -> None:
    """Every hex key in a real game map must exist here with identical row/col."""
    board = base_board()
    path = sorted(glob.glob("data/raw/games/*.json.gz"))[0]
    game = orjson.loads(gzip.decompress(open(path, "rb").read()))  # noqa: SIM115
    real = {
        k: v for k, v in game["map"].items()
        if isinstance(v, dict) and v.get("row") is not None
    }
    assert set(real) == set(board.hexes)
    for key, cell in real.items():
        ours = board.hexes[key]
        # snellman serializes some rows as strings — normalize before comparing
        assert (ours.row, ours.col) == (int(cell["row"]), int(cell["col"])), key
        if key.startswith("r"):
            assert cell["color"] == RIVER
