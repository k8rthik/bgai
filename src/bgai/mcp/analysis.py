"""Rung 2: factual analysis tools (preview_move, score_projection).

Registered by server.py only when rung 2 is enabled. Implementations land
in plan task 9; register() is the stable entry point.
"""

from __future__ import annotations

from mcp.server import MCPServer

from bgai.mcp.session import Session


def register(mcp: MCPServer, session: Session) -> None:
    """Attach rung-2 tools to `mcp` (task 9 fills the tool bodies)."""
