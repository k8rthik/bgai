"""Rung 3: sandbox branching -- LLM-driven search over scratch game copies.

The LLM plays ALL seats inside a branch (that is the point: it models the
opponents' replies with its own judgment). Driver bookkeeping auto-runs
between decisions, and a branch played to the end reports factual final
VP via final scoring -- arithmetic, not evaluation. Branches never touch
the live game or its bots.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from mcp.server import MCPServer

from bgai.arena.live_driver import (
    ALWAYS_END_VERBS,
    MAIN_TRACK_VERBS,
    answer_moves,
    end_action,
    must_continue,
    next_actor,
    offered_moves,
    oldest_blocking_pending,
    play_until_decision,
    spade_holder,
)
from bgai.data.ledger_parser import ParsedCommand, parse_command
from bgai.engine.tm.apply import EngineError, apply
from bgai.engine.tm.state import GameState, Phase
from bgai.mcp.render import render_command, render_moves, render_state
from bgai.mcp.session import Session, _same_move

MAX_BRANCHES = 8


@dataclass(frozen=True)
class BranchState:
    state: GameState
    turn_open_verb: str | None
    created_at_round: int
    created_at_phase: str


def _get(session: Session, label: str) -> BranchState | None:
    bs = session.branches.get(label)
    return bs if isinstance(bs, BranchState) else None


def _branch_offered(bs: BranchState) -> tuple[ParsedCommand, ...]:
    state = bs.state
    actor = next_actor(state)
    if actor is None:
        return ()
    pending = oldest_blocking_pending(state)
    if pending is not None and pending.faction == actor:
        return answer_moves(state, actor)
    if spade_holder(state) == actor:
        return offered_moves(state, actor, turn_open=False)
    return offered_moves(state, actor, turn_open=bs.turn_open_verb is not None)


def _header(bs: BranchState, label: str) -> str:
    return (
        f"branch {label!r} (branched from: round {bs.created_at_round},"
        f" phase {bs.created_at_phase})"
    )


def _advance_branch(bs: BranchState) -> BranchState:
    """Run bookkeeping until some faction owes a decision or FINISHED.
    All seats are external in a branch -- play_until_decision never
    consults bots here.
    """
    events: list[str] = []
    state = play_until_decision(
        bs.state, {}, frozenset(bs.state.setup.factions), events, max_commands=10000
    )
    return replace(bs, state=state)


def branch_create(session: Session, label: str) -> str:
    if session.state is None:
        return "ERROR: no game started -- call new_game first"
    if label in session.branches:
        return f"ERROR: branch {label!r} already exists ({branch_list(session)})"
    if len(session.branches) >= MAX_BRANCHES:
        return f"ERROR: branch limit ({MAX_BRANCHES}) reached; discard one ({branch_list(session)})"
    bs = BranchState(
        state=session.state,
        turn_open_verb=session.turn_open_verb,
        created_at_round=session.state.round,
        created_at_phase=session.state.phase.name,
    )
    session.branches[label] = bs
    return f"created {_header(bs, label)}; you play ALL seats in it"


def branch_list(session: Session) -> str:
    if not session.branches:
        return "no branches"
    return "branches: " + ", ".join(sorted(session.branches))


def branch_discard(session: Session, label: str) -> str:
    if _get(session, label) is None:
        return f"ERROR: no branch {label!r} ({branch_list(session)})"
    del session.branches[label]
    return f"discarded branch {label!r} ({branch_list(session)})"


def branch_state_text(session: Session, label: str) -> str:
    bs = _get(session, label)
    if bs is None:
        return f"ERROR: no branch {label!r} ({branch_list(session)})"
    if bs.state.phase is Phase.FINISHED:
        vp = {f: bs.state.factions[f].vp for f in bs.state.setup.factions}
        table = "\n".join(f"  {f}: {v} VP" for f, v in vp.items())
        return f"{_header(bs, label)}\nFINISHED. Final scores:\n{table}"
    return _header(bs, label) + "\n" + render_state(bs.state, viewer=next_actor(bs.state))


def branch_moves_text(session: Session, label: str) -> str:
    bs = _get(session, label)
    if bs is None:
        return f"ERROR: no branch {label!r} ({branch_list(session)})"
    actor = next_actor(bs.state)
    if actor is None or bs.state.phase is Phase.FINISHED:
        return f"{_header(bs, label)}: FINISHED -- no moves"
    return f"{_header(bs, label)}, {actor} to move\n" + render_moves(_branch_offered(bs), actor)


def branch_play(session: Session, label: str, text: str) -> str:
    bs = _get(session, label)
    if bs is None:
        return f"ERROR: no branch {label!r} ({branch_list(session)})"
    if bs.state.phase is Phase.FINISHED:
        return f"ERROR: branch {label!r} is finished\n\n" + branch_state_text(session, label)
    actor = next_actor(bs.state)
    if actor is None:
        return f"ERROR: branch {label!r} owes no decision"
    move = parse_command(text.strip())
    if move is None:
        return f"ERROR: could not parse {text!r}\n\n" + branch_moves_text(session, label)
    offered = _branch_offered(bs)
    match = next((m for m in offered if _same_move(m, move)), None)
    if match is None:
        return (
            f"ERROR: {text!r} is not legal in branch {label!r}\n\n"
            + branch_moves_text(session, label)
        )

    pending = oldest_blocking_pending(bs.state)
    answering = pending is not None and pending.faction == actor
    forced_spade = not answering and spade_holder(bs.state) == actor

    if match.verb == "done":
        bs = replace(bs, state=end_action(bs.state, actor), turn_open_verb=None)
    else:
        try:
            state = apply(bs.state, actor, match)
        except EngineError as exc:
            return f"ERROR: engine rejected the move: {exc}"
        bs = replace(bs, state=state)
        if not answering and not forced_spade and match.verb in MAIN_TRACK_VERBS:
            if match.verb in ALWAYS_END_VERBS:
                bs = replace(bs, state=end_action(bs.state, actor), turn_open_verb=None)
            elif match.verb in ("transform", "dig") or must_continue(bs.state, actor):
                bs = replace(bs, turn_open_verb=match.verb)
            else:
                bs = replace(bs, state=end_action(bs.state, actor), turn_open_verb=None)

    bs = _advance_branch(bs)
    session.branches[label] = bs
    played = f"{actor} played {render_command(match)} in branch {label!r}"
    if bs.state.phase is Phase.FINISHED:
        return f"{played}\n\n" + branch_state_text(session, label)
    nxt = next_actor(bs.state)
    return f"{played}\nnext to move: {nxt}\n\n" + branch_moves_text(session, label)


def register(mcp: MCPServer, session: Session) -> None:
    @mcp.tool()
    def branch(label: str) -> str:
        """Fork the live game into a scratch branch (max 8). You play ALL
        seats in a branch -- model the opponents' replies yourself. The
        live game and its bots are never affected."""
        return branch_create(session, label)

    @mcp.tool()
    def branch_moves(label: str) -> str:
        """Legal moves for whichever faction owes the next decision in the
        branch."""
        return branch_moves_text(session, label)

    @mcp.tool()
    def branch_play_move(label: str, move: str) -> str:
        """Play one command in the branch for the faction that owes the
        decision (bookkeeping auto-runs between decisions; a finished
        branch reports factual final VP)."""
        return branch_play(session, label, move)

    @mcp.tool()
    def branch_state(label: str) -> str:
        """Full factual state of the branch (header shows the live-game
        point it forked from)."""
        return branch_state_text(session, label)

    @mcp.tool()
    def discard_branch(label: str) -> str:
        """Delete a branch."""
        return branch_discard(session, label)

    @mcp.tool()
    def list_branches() -> str:
        """List existing branch labels."""
        return branch_list(session)
