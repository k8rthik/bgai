"""Rung 2: factual analysis tools (preview_move, score_projection).

Scratch-paper arithmetic only -- exact deltas of a hypothetical move, and
mechanically guaranteed payouts "if the round/game ended now". No
desirability signal of any kind (spec: the facts/judgment boundary); the
wording deliberately sticks to numbers and conditional framing.
"""

from __future__ import annotations

from mcp.server import MCPServer

from bgai.data.ledger_parser import parse_command
from bgai.engine.tm.apply import EngineError, apply
from bgai.engine.tm.factions_data import CULTS
from bgai.engine.tm.scoring import (
    _apply_score_resources,
    compute_cult_scoring,
    compute_network_scoring,
    network_size,
)
from bgai.engine.tm.state import GameState
from bgai.engine.tm.tiles import BONUS_TILES, FAVOR_TILES
from bgai.mcp.render import diff_factions
from bgai.mcp.session import Session, _same_move


def preview(session: Session, text: str) -> str:
    """Apply `text` on a scratch copy of the live game (the immutable state
    IS the branch) and report the factual consequences; never commits.
    """
    if session.state is None:
        return "ERROR: no game started -- call new_game first"
    actor = session.current_actor()
    if actor is None:
        return "ERROR: no decision pending"
    move = parse_command(text.strip())
    if move is None:
        return f"ERROR: could not parse {text!r}"
    match = next((m for m in session.offered() if _same_move(m, move)), None)
    if match is None:
        return f"ERROR: {text!r} is not a legal move right now"
    try:
        after = apply(session.state, actor, match)
    except EngineError as exc:
        return f"ERROR: engine rejected the move: {exc}"
    diff = diff_factions(session.state, after)
    return f"preview of {text!r} for {actor} (nothing committed):\n{diff}"


def _round_tile_income(state: GameState) -> list[str]:
    if not 1 <= state.round <= 6:
        return []
    tile = state.setup.score_tiles[state.round - 1]
    lines = [
        f"this round's scoring tile pays cult income at cleanup:"
        f" per {tile.req} steps on {tile.cult}:"
    ]
    if not tile.cult_income:
        return []
    for faction in state.setup.factions:
        position = state.cults[faction][tile.cult]
        units = position // tile.req
        payout = ", ".join(f"{units * amount}{name}" for name, amount in tile.cult_income)
        lines.append(f"  {faction}: {tile.cult} {position} -> {payout}")
    return lines


def _cult_standings(state: GameState) -> list[str]:
    scores = compute_cult_scoring(state)
    lines = ["end-game cult VP (8/4/2, ties split) IF standings froze now:"]
    for cult in CULTS:
        entries = ", ".join(
            f"{faction} +{vp}" for faction, vp in sorted(scores[cult].items())
        ) or "nobody scores"
        lines.append(f"  {cult}: {entries}")
    return lines


def _network_standings(state: GameState) -> list[str]:
    sizes = {f: network_size(state, f) for f in state.setup.factions}
    scores = compute_network_scoring(state)
    lines = ["network sizes and end-game network VP (18/12/6) IF the game ended now:"]
    for faction in state.setup.factions:
        vp = scores.get(faction, 0)
        lines.append(f"  {faction}: network {sizes[faction]} -> +{vp} VP")
    return lines


def _pass_vp_arithmetic(state: GameState) -> list[str]:
    lines = ["pass-VP arithmetic of currently held tiles:"]
    for faction in state.setup.factions:
        fs = state.factions[faction]
        parts = []
        if fs.bonus is not None:
            tile = BONUS_TILES[fs.bonus]
            total = sum(
                per * len(fs.buildings.get(b, frozenset())) for b, per in tile.pass_vp
            )
            if tile.pass_vp:
                parts.append(f"{fs.bonus} pays {total} VP at pass")
        for favor in fs.favors:
            tile = FAVOR_TILES[favor]
            if tile.pass_vp:
                count = len(fs.buildings.get("TP", frozenset()))
                idx = min(count, len(tile.pass_vp) - 1)
                parts.append(f"{favor} pays {tile.pass_vp[idx]} VP at pass ({count} TP)")
        if parts:
            lines.append(f"  {faction}: " + "; ".join(parts))
    return lines if len(lines) > 1 else []


def _resource_conversion(state: GameState) -> list[str]:
    lines = ["leftover-resource VP conversion IF the game ended now:"]
    for faction in state.setup.factions:
        fs = state.factions[faction]
        converted = _apply_score_resources(fs, faction)
        lines.append(f"  {faction}: +{converted.vp - fs.vp} VP")
    return lines


def projection(session: Session) -> str:
    """Mechanically guaranteed scoring arithmetic from the frozen state."""
    if session.state is None:
        return "ERROR: no game started -- call new_game first"
    state = session.state
    sections = [
        _round_tile_income(state),
        _cult_standings(state),
        _network_standings(state),
        _pass_vp_arithmetic(state),
        _resource_conversion(state),
    ]
    return "\n".join("\n".join(s) for s in sections if s)


def register(mcp: MCPServer, session: Session) -> None:
    @mcp.tool()
    def preview_move(move: str) -> str:
        """Factual preview of one legal move: exact resource/VP/board deltas
        and any decisions it would trigger. Arithmetic only, no evaluation;
        nothing is committed."""
        return preview(session, move)

    @mcp.tool()
    def score_projection() -> str:
        """Mechanically guaranteed scoring arithmetic from the current
        state: this round's tile payout, cult/network VP if standings froze
        now, pass-VP of held tiles, leftover-resource conversion. No
        evaluation -- numbers only."""
        return projection(session)
