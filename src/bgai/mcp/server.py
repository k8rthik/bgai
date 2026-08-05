"""MCPServer wiring for the Terra Mystica session (stdio transport).

Pure delegation to :class:`bgai.mcp.session.Session`; rung-2/3 tools are
registered only when the session config enables those rungs. Run with
`uv run python -m bgai.mcp.server` (the repo's .mcp.json does exactly that).
"""

from __future__ import annotations

from mcp.server import MCPServer

from bgai.mcp.config import load_config
from bgai.mcp.session import Session

mcp = MCPServer("tm")
SESSION = Session(load_config())


@mcp.tool()
def new_game(seed: int | None = None) -> str:
    """Start (or restart) the configured Terra Mystica game. Bots fill the
    other seats; returns the state once your seat owes a decision.
    Optional seed overrides the configured one (interactive use).
    """
    return SESSION.start(seed=seed)


@mcp.tool()
def get_state() -> str:
    """The full current game state (factual render: board, factions,
    cults, tiles, pending decisions) plus your current legal moves."""
    return SESSION.get_state()


@mcp.tool()
def legal_moves() -> str:
    """Your seat's legal moves right now, as submittable command strings."""
    return SESSION.legal_moves()


@mcp.tool()
def play_move(move: str) -> str:
    """Submit one command for your seat (e.g. 'build A5', 'dig 1',
    'leech 2 from nomads', 'pass BON3', or 'done' to end an open compound
    turn). Bots then play until your seat owes the next decision; returns
    what happened plus the new state.
    """
    return SESSION.submit(move)


def _register_rung_tools() -> None:
    if 2 in SESSION.config.rungs:
        from bgai.mcp import analysis

        analysis.register(mcp, SESSION)
    if 3 in SESSION.config.rungs:
        from bgai.mcp import branching

        branching.register(mcp, SESSION)


_register_rung_tools()


if __name__ == "__main__":
    mcp.run()
