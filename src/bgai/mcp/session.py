"""The one sanctioned mutable object: an MCP game session.

Owns the authoritative GameState, the bot opponents, the external seat's
open-turn bookkeeping, and (rung 3) sandbox branches. Every failure path
returns a structured "ERROR: ..." string with the current legal moves --
tool calls never raise out of the session.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from bgai.agents.base import Agent
from bgai.agents.greedy import GreedyAgent
from bgai.agents.random_agent import RandomAgent
from bgai.arena.driver import (
    ALWAYS_END_VERBS,
    MAIN_TRACK_VERBS,
    DriverError,
    answer_moves,
    end_action,
    must_continue,
    new_game,
    next_actor,
    offered_moves,
    oldest_blocking_pending,
    play_until_decision,
    spade_holder,
)
from bgai.arena.setup_factory import fresh_setup
from bgai.data.ledger_parser import ParsedCommand, parse_command
from bgai.engine.tm.apply import EngineError, apply
from bgai.engine.tm.state import GameState, Phase
from bgai.mcp.config import SessionConfig
from bgai.mcp.render import render_command, render_moves, render_state

_COMPARE_FIELDS = (
    "verb", "loc", "loc2", "building", "tile", "cult", "color", "target", "reason",
    "res1", "res2", "n1", "n2",
)


def _same_move(a: ParsedCommand, b: ParsedCommand) -> bool:
    return all(getattr(a, f) == getattr(b, f) for f in _COMPARE_FIELDS)


class Session:
    def __init__(self, config: SessionConfig) -> None:
        self.config = config
        self.rng = random.Random(config.seed * 7919 + 17)
        self.state: GameState | None = None
        self.events: list[str] = []
        self.turn_open_verb: str | None = None
        self.result_written = False
        self.branches: dict[str, object] = {}  # rung 3 (task 10)

    # -- seats -------------------------------------------------------------

    @property
    def llm_faction(self) -> str:
        assert self.state is not None
        if self.config.llm_faction_index == -1:
            return self.current_actor() or "?"
        return self.state.setup.factions[self.config.llm_faction_index]

    @property
    def external(self) -> frozenset[str]:
        assert self.state is not None
        if self.config.llm_faction_index == -1:
            return frozenset(self.state.setup.factions)
        return frozenset({self.state.setup.factions[self.config.llm_faction_index]})

    def _bots(self) -> dict[str, Agent]:
        assert self.state is not None
        bots: dict[str, Agent] = {}
        for i, faction in enumerate(self.state.setup.factions):
            if faction in self.external:
                continue
            if self.config.opponents == "random":
                bots[faction] = RandomAgent(name=f"random{i}")
            else:
                bots[faction] = GreedyAgent(name=f"greedy{i}")
        return bots

    def current_actor(self) -> str | None:
        assert self.state is not None
        return next_actor(self.state)

    # -- lifecycle ---------------------------------------------------------

    def start(self, seed: int | None = None) -> str:
        config = self.config
        setup = fresh_setup(
            seed=config.seed if seed is None else seed, factions=config.factions
        )
        self.state = new_game(setup)
        self.events = []
        self.turn_open_verb = None
        self.result_written = False
        self.branches = {}
        note = self._advance()
        return note + self._status_text()

    def _advance(self) -> str:
        """Run bots + bookkeeping to the next external decision (or game
        end); returns the events that happened as text.
        """
        assert self.state is not None
        events: list[str] = []
        try:
            self.state = play_until_decision(
                self.state, self._bots(), self.external, events, self.config.max_commands,
                rng=self.rng,
            )
        except DriverError as exc:  # pragma: no cover - defensive
            events.append(f"-- driver error: {exc}")
        self.events.extend(events)
        if self.state.phase is Phase.FINISHED:
            self._write_result()
        if not events:
            return ""
        return "events since your last move:\n" + "\n".join(f"  {e}" for e in events) + "\n\n"

    def _write_result(self) -> None:
        assert self.state is not None
        if self.result_written:
            return
        self.result_written = True
        state = self.state
        vp = {f: state.factions[f].vp for f in state.setup.factions}
        top = max(vp.values())
        payload = {
            "game_id": state.setup.game_id,
            "seed": self.config.seed,
            "llm_faction": self.state.setup.factions[self.config.llm_faction_index]
            if self.config.llm_faction_index != -1
            else None,
            "vp": vp,
            "winners": [f for f in state.setup.factions if vp[f] == top],
            "commands_used": sum(1 for e in self.events if not e.startswith("--")),
            "rungs": list(self.config.rungs),
        }
        if self.config.result_path:
            Path(self.config.result_path).write_text(json.dumps(payload, indent=2))

    # -- state & moves -----------------------------------------------------

    def _status_text(self) -> str:
        assert self.state is not None
        if self.state.phase is Phase.FINISHED:
            vp = {f: self.state.factions[f].vp for f in self.state.setup.factions}
            table = "\n".join(f"  {f}: {v} VP" for f, v in vp.items())
            return f"GAME FINISHED. Final scores:\n{table}"
        actor = self.current_actor()
        header = f"you are piloting: {actor}\n\n" if actor else ""
        return (
            header
            + render_state(self.state, viewer=actor)
            + "\n\n"
            + render_moves(self.offered(), actor or "?")
        )

    def get_state(self) -> str:
        if self.state is None:
            return "ERROR: no game started -- call new_game first"
        return self._status_text()

    def offered(self) -> tuple[ParsedCommand, ...]:
        """Moves for the current external actor, per the driver's turn
        protocol (pending answers > forced spade transform > open turn).
        """
        assert self.state is not None
        actor = self.current_actor()
        if actor is None or actor not in self.external:
            return ()
        pending = oldest_blocking_pending(self.state)
        if pending is not None and pending.faction == actor:
            return answer_moves(self.state, actor)
        if spade_holder(self.state) == actor:
            return offered_moves(self.state, actor, turn_open=False)
        return offered_moves(self.state, actor, turn_open=self.turn_open_verb is not None)

    def legal_moves(self) -> str:
        if self.state is None:
            return "ERROR: no game started -- call new_game first"
        actor = self.current_actor()
        if actor is None:
            return "no decision pending (game may be finished)"
        return render_moves(self.offered(), actor)

    # -- the move protocol -------------------------------------------------

    def submit(self, text: str) -> str:
        if self.state is None:
            return "ERROR: no game started -- call new_game first"
        if self.state.phase is Phase.FINISHED:
            return "ERROR: game is finished\n\n" + self._status_text()
        actor = self.current_actor()
        if actor is None or actor not in self.external:
            return "ERROR: it is not your seat's turn\n\n" + self.legal_moves()
        move = parse_command(text.strip())
        if move is None:
            return f"ERROR: could not parse {text!r}\n\n" + self.legal_moves()
        offered = self.offered()
        match = next((m for m in offered if _same_move(m, move)), None)
        if match is None:
            return f"ERROR: {text!r} is not a legal move right now\n\n" + self.legal_moves()

        pending = oldest_blocking_pending(self.state)
        answering = pending is not None and pending.faction == actor
        forced_spade = not answering and spade_holder(self.state) == actor

        if match.verb == "done":
            self.state = end_action(self.state, actor)
            self.turn_open_verb = None
            self.events.append(f"{actor}: done")
            return self._advance() + self._status_text()

        try:
            self.state = apply(self.state, actor, match)
        except EngineError as exc:
            return f"ERROR: engine rejected the move: {exc}\n\n" + self.legal_moves()
        self.events.append(f"{actor}: {render_command(match)}")

        if not answering and not forced_spade and match.verb in MAIN_TRACK_VERBS:
            if match.verb in ALWAYS_END_VERBS:
                self.state = end_action(self.state, actor)
                self.turn_open_verb = None
            elif match.verb in ("transform", "dig") or must_continue(self.state, actor):
                self.turn_open_verb = match.verb
            else:
                self.state = end_action(self.state, actor)
                self.turn_open_verb = None

        return self._advance() + self._status_text()
