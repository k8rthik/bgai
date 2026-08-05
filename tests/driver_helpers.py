"""Shared helpers for driver tests: setup fast-forward + scripted agent."""

from __future__ import annotations

from bgai.arena.driver import new_game, next_actor, offered_moves, progress_moves
from bgai.arena.setup_factory import fresh_setup
from bgai.data.ledger_parser import ParsedCommand
from bgai.engine.tm.apply import apply
from bgai.engine.tm.round_flow import advance_turn
from bgai.engine.tm.state import GameState, Phase


def fast_forward_setup(seed: int) -> GameState:
    """Play both setup phases with first-legal-move picks; stops at round-1
    INCOME. One `advance_turn` per setup build/pass row (Task-12 contract).
    """
    state = new_game(fresh_setup(seed=seed))
    while state.phase in (Phase.SETUP_DWELLINGS, Phase.SETUP_BONUS):
        actor = next_actor(state)
        assert actor is not None, f"no actor during {state.phase}"
        move = progress_moves(offered_moves(state, actor, turn_open=False))[0]
        state = apply(state, actor, move)
        state = advance_turn(state)
    return state


class ScriptedAgent:
    """Pops moves from a fixed script, matching each against the offered set
    by (verb, loc, tile). Raises if the scripted move is not offered.
    """

    def __init__(self, script: list[tuple[str, str | None, str | None]], name: str) -> None:
        self.name = name
        self._script = list(script)

    def choose(
        self,
        state: GameState,
        faction: str,
        moves: tuple[ParsedCommand, ...],
        rng=None,
    ) -> ParsedCommand:
        if not self._script:
            raise AssertionError(f"{self.name}: script exhausted, offered {moves}")
        verb, loc, tile = self._script.pop(0)
        for m in moves:
            if m.verb == verb and m.loc == loc and m.tile == tile:
                return m
        raise AssertionError(f"{self.name}: scripted {(verb, loc, tile)} not in offered {moves}")
