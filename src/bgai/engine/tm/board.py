"""Base-map board geometry: hex layout, coordinates, direct adjacency.

Ported from the reference implementation (jsnell/terra-mystica, MIT):
``Game/Constants.pm`` (@base_map) and ``map.pm`` (setup_base_map,
setup_direct_adjacencies, hex_distance). Land hexes are keyed "A1".."I13"
(letter row, 1-based number skipping rivers); river hexes are "r0".."r35".
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

# Row-major terrain layout; "x" is river, "E" ends a row. Verbatim from
# Game/Constants.pm in jsnell/terra-mystica.
BASE_MAP = """
brown gray green blue yellow red brown black red green blue red black E
yellow x x brown black x x yellow black x x yellow E
x x black x gray x green x green x gray x x E
green blue yellow x x red blue x red x red brown E
black brown red blue black brown gray yellow x x green black blue E
gray green x x yellow green x x x brown gray brown E
x x x gray x red x green x yellow black blue yellow E
yellow blue brown x x x blue black x gray brown gray E
red black gray blue red green yellow brown gray x blue green red E
""".split()  # noqa: SIM905 — row layout readability beats a flat list literal

RIVER = "white"


@dataclass(frozen=True)
class Hex:
    key: str
    color: str  # original terrain color, or "white" for river
    row: int
    col: int


@dataclass(frozen=True)
class Board:
    hexes: dict[str, Hex]
    adjacent: dict[str, frozenset[str]]

    def land_hexes(self) -> list[str]:
        return [k for k, h in self.hexes.items() if h.color != RIVER]


def _layout() -> dict[str, Hex]:
    hexes: dict[str, Hex] = {}
    row_label = "A"
    row_idx = 0
    col_idx = 0
    land_col = 1
    river_idx = 0
    for cell in BASE_MAP:
        if cell == "E":
            row_label = chr(ord(row_label) + 1)
            row_idx += 1
            col_idx = 0
            land_col = 1
            continue
        if cell == "x":
            key = f"r{river_idx}"
            hexes[key] = Hex(key=key, color=RIVER, row=row_idx, col=col_idx)
            river_idx += 1
        else:
            key = f"{row_label}{land_col}"
            hexes[key] = Hex(key=key, color=cell, row=row_idx, col=col_idx)
            land_col += 1
        col_idx += 1
    return hexes


def _adjacency(hexes: dict[str, Hex]) -> dict[str, frozenset[str]]:
    by_coord = {(h.row, h.col): h.key for h in hexes.values()}
    adjacent: dict[str, frozenset[str]] = {}
    for h in hexes.values():
        col = h.col
        neighbors = [by_coord.get((h.row, col + 1)), by_coord.get((h.row, col - 1))]
        # Adjacent rows: offset the column by one for every other row.
        offset_col = col - 1 if h.row % 2 == 0 else col
        neighbors += [
            by_coord.get((h.row - 1, offset_col)),
            by_coord.get((h.row - 1, offset_col + 1)),
            by_coord.get((h.row + 1, offset_col)),
            by_coord.get((h.row + 1, offset_col + 1)),
        ]
        adjacent[h.key] = frozenset(n for n in neighbors if n is not None)
    return adjacent


@lru_cache(maxsize=1)
def base_board() -> Board:
    hexes = _layout()
    return Board(hexes=hexes, adjacent=_adjacency(hexes))


def hex_distance(a: str, b: str, board: Board | None = None) -> int:
    """Crow-flies hex distance (port of map.pm hex_distance)."""
    board = board or base_board()
    ha, hb = board.hexes[a], board.hexes[b]
    if a == b:
        return 0
    rdelta = abs(ha.row - hb.row)
    cdelta = abs((ha.col * 2 + ha.row % 2) - (hb.col * 2 + hb.row % 2))
    if cdelta < rdelta:
        return rdelta
    dist = 0
    while rdelta > 1:
        rdelta -= 2
        cdelta -= 2
        dist += 2
    if rdelta:
        cdelta -= 1
        dist += 1
    return dist + cdelta // 2
