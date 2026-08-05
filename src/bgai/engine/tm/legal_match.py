"""Match a replayed ledger command to its generated legal candidate.

A ledger ``ParsedCommand`` and the ``legal_moves`` candidate that
represents the same decision are not field-identical: the ledger records
resolved bookkeeping the generator has no reason to enumerate (a
``decline``'s target and power amount, a ``leech``'s target, the exact
unit count of a ``convert``), and the generator sometimes enumerates
detail the ledger omits (a ``transform``'s target color). Each verb
therefore has its own *identity key* -- the fields that actually pick out
the decision -- and matching compares only those.

This module is the single owner of that mapping. ``tests/test_legal.py``'s
corpus containment sweep and ``bgai.training.extract``'s label lookup both
use it, so the 29-game slow sweep is also this matcher's regression test.
"""

from __future__ import annotations

from bgai.data.ledger_parser import ParsedCommand

_COLOR_ALIASES = {"grey": "gray"}


def _norm_color(color: str | None) -> str | None:
    return _COLOR_ALIASES.get(color, color) if color else color


def _matches(candidate: ParsedCommand, real: ParsedCommand) -> bool:
    verb = real.verb
    if candidate.verb != verb:
        return False
    if verb == "convert":
        return candidate.res1 == real.res1 and candidate.res2 == real.res2
    if verb in ("dig", "burn"):
        return candidate.n1 == real.n1
    if verb == "transform":
        real_color = _norm_color(real.color)
        if real_color is None:
            return candidate.loc == real.loc
        return candidate.loc == real.loc and _norm_color(candidate.color) == real_color
    if verb in ("gain_town", "gain_favor", "pass", "action"):
        return candidate.tile == real.tile
    if verb in ("leech", "decline"):
        return True
    if verb == "send":
        return candidate.cult == real.cult
    if verb == "advance":
        return candidate.reason == real.reason
    if verb == "bridge":
        return {candidate.loc, candidate.loc2} == {real.loc, real.loc2}
    if verb in ("build", "connect"):
        return candidate.loc == real.loc
    if verb == "upgrade":
        return candidate.loc == real.loc and candidate.building == real.building
    return True


def match_index(legal: tuple[ParsedCommand, ...], real: ParsedCommand) -> int | None:
    """Index of the first candidate in ``legal`` denoting the same
    decision as ``real``; ``None`` if the generator never offered it
    (a containment failure).
    """
    for index, candidate in enumerate(legal):
        if _matches(candidate, real):
            return index
    return None


def is_contained(legal: tuple[ParsedCommand, ...], real: ParsedCommand) -> bool:
    return match_index(legal, real) is not None
