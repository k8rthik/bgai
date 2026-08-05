"""ParsedCommand -> fixed-width int16 feature ids, v1 (plan Task 4).

Categorical fields use id 0 as the reserved null ("field absent");
real vocabulary ids are offset by +1. ``target`` (the other faction in
leech/decline) is encoded as a mover-relative seat id so the policy is
seat-equivariant. n1/n2 are clamped raw amounts, not vocabulary ids.

Field order (MOVE_FIELDS = 12):
  0 verb | 1 loc | 2 loc2 | 3 building | 4 tile | 5 cult | 6 color |
  7 target-rel-seat | 8 res1 | 9 res2 | 10 n1 | 11 n2
"""

from __future__ import annotations

import numpy as np

from bgai.data.ledger_parser import ParsedCommand
from bgai.engine.tm.state import GameState
from bgai.training.vocab import (
    BUILDING_INDEX,
    COLOR_INDEX,
    CULT_INDEX,
    HEX_INDEX,
    RESOURCE_INDEX,
    TILE_INDEX,
    VERB_INDEX,
    normalize_color,
)

MOVE_FIELDS = 12
_N_CLAMP = 30


def _id(index: dict[str, int], key: str | None) -> int:
    return 0 if key is None else index[key] + 1


def encode_move(cmd: ParsedCommand, state: GameState, faction: str) -> np.ndarray:
    out = np.zeros(MOVE_FIELDS, dtype=np.int16)
    out[0] = VERB_INDEX[cmd.verb] + 1
    out[1] = _id(HEX_INDEX, cmd.loc)
    out[2] = _id(HEX_INDEX, cmd.loc2)
    out[3] = _id(BUILDING_INDEX, cmd.building)
    out[4] = _id(TILE_INDEX, cmd.tile)
    out[5] = _id(CULT_INDEX, cmd.cult)
    out[6] = _id(COLOR_INDEX, normalize_color(cmd.color) if cmd.color else None)
    if cmd.target is not None:
        seats = state.setup.factions
        rel = (seats.index(cmd.target) - seats.index(faction)) % len(seats)
        out[7] = rel + 1
    out[8] = _id(RESOURCE_INDEX, cmd.res1)
    out[9] = _id(RESOURCE_INDEX, cmd.res2)
    out[10] = min(cmd.n1, _N_CLAMP) if cmd.n1 is not None else 0
    out[11] = min(cmd.n2, _N_CLAMP) if cmd.n2 is not None else 0
    return out
