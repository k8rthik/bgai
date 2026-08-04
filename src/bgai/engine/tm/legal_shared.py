"""Shared primitives for legal-move enumeration.

A tiny leaf module so ``legal.py`` (top-level API, pending-decision
answers, setup-phase enumeration) and ``legal_actions.py`` (the bulk of
the ACTIONS-phase main-action enumeration) can both depend on these
without importing each other -- ``legal.py`` calls into
``legal_actions.actions_phase_moves``, so the reverse edge would be a
cycle if these lived in either of those two files instead.
"""

from __future__ import annotations

from collections.abc import Mapping

from bgai.data.ledger_parser import Kind, ParsedCommand
from bgai.engine.tm.state import FactionState, GameState

_RESOURCE_ATTR: dict[str, str] = {"C": "coins", "W": "workers", "P": "priests", "VP": "vp"}


def cmd(verb: str, *, kind: Kind = Kind.DECISION, **fields: object) -> ParsedCommand:
    """Build a synthetic :class:`ParsedCommand` for a legal-moves set.

    ``raw`` is just the verb -- these commands are never re-parsed, only
    fed to ``apply()`` (generative smoke test) or compared structurally
    against real ledger rows (containment test).
    """
    return ParsedCommand(verb=verb, kind=kind, raw=verb, **fields)  # type: ignore[arg-type]


def resource_amount(fs: FactionState, res: str) -> int:
    """Current amount of ``res`` held by ``fs`` (``PW`` = usable/bowl3
    power) -- mirrors ``apply.py``'s private ``_get_resource``, duplicated
    here rather than imported (that helper is an ``apply.py`` internal,
    not exported API).
    """
    if res == "PW":
        return fs.power.usable
    return getattr(fs, _RESOURCE_ATTR[res])


def affordable(fs: FactionState, cost: Mapping[str, int]) -> bool:
    """Whether ``fs`` currently holds every resource in ``cost`` at or
    above the required amount. Non-positive entries (should never occur
    in a real cost dict, but defensive) never fail the check.
    """
    return all(resource_amount(fs, res) >= amt for res, amt in cost.items() if amt > 0)


def own_pending_index(state: GameState, faction: str, kind: str) -> int | None:
    """Index of the first queued ``PendingDecision`` of ``kind`` belonging
    to ``faction``, or ``None``. Mirrors the identically-named-in-spirit
    ``_find_pending_optional`` helpers duplicated across
    ``actions_build.py``/``actions_terraform.py``/``actions_power.py``.
    """
    for i, p in enumerate(state.pending):
        if p.faction == faction and p.kind == kind:
            return i
    return None
