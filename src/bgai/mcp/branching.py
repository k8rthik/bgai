"""Rung 3: sandbox branching tools (branch, branch_play, ...).

Registered by server.py only when rung 3 is enabled. Implementations land
in plan task 10; register() is the stable entry point.
"""

from __future__ import annotations

from mcp.server import MCPServer

from bgai.mcp.session import Session


def register(mcp: MCPServer, session: Session) -> None:
    """Attach rung-3 tools to `mcp` (task 10 fills the tool bodies)."""
