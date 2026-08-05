"""Terra Mystica state -> text, for LLM players (Phase 7, rung L0).

The serialization is the LLM track's equivalent of the imitation
encoder: everything the model is allowed to see, in a form it can
actually reason over. Three constraints shaped it:

1. **Mover-relative, like the tensor encoder.** "You" is always the
   acting faction, opponents are listed in turn order after you.
2. **Numbers, not prose.** Resources, cult positions, and costs are the
   things an LLM gets wrong when it has to infer them; spelling them out
   removes an entire error class.
3. **Legal moves are numbered.** The model answers with an index, so a
   malformed or illegal answer is impossible to express -- the same
   "legality is structural" property the imitation net gets from
   candidate scoring.

The board is rendered as an occupancy list rather than ASCII art: the
hex grid's adjacency does not survive a text picture, and a wrong
mental picture is worse than none. Adjacency the model needs for a
specific move is available through the `neighbors` tool (L2).
"""

from __future__ import annotations

from bgai.data.ledger_parser import ParsedCommand
from bgai.engine.tm.board import base_board
from bgai.engine.tm.factions_data import CULTS, FACTIONS
from bgai.engine.tm.state import GameState

_BUILDING_ORDER = ("D", "TP", "TE", "SH", "SA")


def describe_move(cmd: ParsedCommand) -> str:
    """One legal move, in the ledger's own vocabulary (which is also the
    vocabulary of every strategy article the model has read)."""
    bits = [cmd.verb]
    if cmd.loc:
        bits.append(cmd.loc)
    if cmd.loc2:
        bits.append(f"-{cmd.loc2}")
    if cmd.building:
        bits.append(f"to {cmd.building}")
    if cmd.color:
        bits.append(f"to {cmd.color}")
    if cmd.tile:
        bits.append(cmd.tile)
    if cmd.cult:
        bits.append(cmd.cult)
    if cmd.res1 and cmd.res2:
        bits.append(f"{cmd.n1 or 1}{cmd.res1}->{cmd.n2 or 1}{cmd.res2}")
    elif cmd.n1 is not None and cmd.verb in ("burn", "dig", "leech"):
        bits.append(str(cmd.n1))
    if cmd.target:
        bits.append(f"from {cmd.target}")
    return " ".join(bits)


def _faction_block(state: GameState, faction: str, *, is_you: bool) -> str:
    fs = state.factions[faction]
    data = FACTIONS[faction]
    buildings = ", ".join(
        f"{b}x{len(fs.buildings.get(b, ()))}" for b in _BUILDING_ORDER if fs.buildings.get(b)
    )
    cults = " ".join(f"{c[0]}{state.cults[faction][c]}" for c in CULTS)
    header = "YOU" if is_you else faction
    lines = [
        f"{header} ({faction}, {data.color}): {fs.vp} VP"
        + (" [passed]" if fs.passed else ""),
        f"  C{fs.coins} W{fs.workers} P{fs.priests}  power {fs.power.bowl1}/"
        f"{fs.power.bowl2}/{fs.power.bowl3}",
        f"  cults {cults}  shipping {fs.shipping}  dig {fs.dig_level}  keys {fs.keys}",
        f"  buildings {buildings or 'none'}",
    ]
    held = []
    if fs.bonus:
        held.append(fs.bonus)
    held.extend(fs.favors)
    held.extend(fs.towns)
    if held:
        lines.append(f"  tiles {' '.join(held)}")
    return "\n".join(lines)


def _board_block(state: GameState) -> str:
    occupied = [
        (hex_key, hx)
        for hex_key, hx in sorted(state.hexes.items())
        if hx.building is not None
    ]
    if not occupied:
        return "board: empty"
    by_faction: dict[str, list[str]] = {}
    for hex_key, hx in occupied:
        by_faction.setdefault(hx.owner or "?", []).append(f"{hex_key}:{hx.building}")
    lines = ["board (occupied hexes):"]
    for faction in state.setup.factions:
        entries = by_faction.get(faction)
        if entries:
            lines.append(f"  {faction}: {' '.join(entries)}")
    return "\n".join(lines)


def _scoring_block(state: GameState) -> str:
    lines = ["round scoring tiles:"]
    for index, tile in enumerate(state.setup.score_tiles, start=1):
        marker = " <- this round" if index == state.round else ""
        vp = ", ".join(f"{k}:{v}" for k, v in tile.vp) or "-"
        income = ", ".join(f"{k}:{v}" for k, v in tile.cult_income) or "-"
        lines.append(
            f"  R{index}: {tile.vp_mode} {vp} | cult {tile.cult} req {tile.req} "
            f"income {income}{marker}"
        )
    return "\n".join(lines)


def serialize_state(state: GameState, faction: str) -> str:
    """Full public state from ``faction``'s point of view."""
    seats = state.setup.factions
    pivot = seats.index(faction)
    order = seats[pivot:] + seats[:pivot]
    blocks = [
        f"Terra Mystica, 4 players, round {state.round}/6, phase {state.phase.name}",
        f"turn order: {' -> '.join(state.turn_order)}",
        "",
        *(_faction_block(state, name, is_you=(name == faction)) for name in order),
        "",
        _board_block(state),
        "",
        _scoring_block(state),
        f"bonus tiles in pool: {' '.join(sorted(state.setup.bonus_tiles))}",
        f"power actions taken this round: {' '.join(sorted(state.power_actions_taken)) or 'none'}",
    ]
    return "\n".join(blocks)


def serialize_offer(offer: tuple[ParsedCommand, ...]) -> str:
    """The numbered legal-move list the model chooses from."""
    return "\n".join(f"{i}. {describe_move(m)}" for i, m in enumerate(offer))


def neighbors(hex_key: str) -> tuple[str, ...]:
    """Directly adjacent hexes -- the adjacency an occupancy list cannot
    show. Exposed as an L2 tool."""
    return tuple(sorted(base_board().adjacent.get(hex_key, frozenset())))
