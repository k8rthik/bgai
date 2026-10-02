"""Hash-independent canonical ordering for the env's offers.

``driver.canonical_moves`` sorts every offer by all of a command's fields
so that an action *index* means the same move in two processes. That holds
for every verb but one.

**The finding.** ``legal_actions.bridge_moves`` builds its command from an
unordered hex pair:

    a, b = tuple(pair)            # pair is a frozenset[str]
    moves.append(cmd("bridge", loc=a, loc2=b))

``tuple()`` over a frozenset of two strings yields them in *hash* order, and
CPython randomizes string hashes per process. So the same bridge is offered
as ``bridge loc=E4 loc2=G1`` in one process and ``bridge loc=G1 loc2=E4`` in
another. Sorting cannot repair that: the two commands differ in their field
values, not merely in their position, so they sort to different places and
every later action index shifts. Measured: two subprocesses at
``PYTHONHASHSEED=1`` and ``987654``, same seed and same raw action stream,
agreed for 55 decisions and then diverged at the first bridge offer, ending
191 vs 196 decisions with different final VP
(tests/test_env_determinism.py pins that this no longer happens).

**What this module does about it.** Nothing to the engine -- the rules
engine's behaviour is not this layer's to change, and both spellings are
accepted by ``apply``, so neither is wrong. Instead the env presents an
*orientation-normalized* offer (hex pair spelled low-to-high) in a stable
order, and keeps the engine's own command objects alongside to step with
(``driver.advance`` checks offer membership by equality, so the original
has to be what is applied).

Consequences, deliberately:

* the env's action index is reproducible across processes and hash seeds;
* the observation's candidate features for a bridge are reproducible too,
  since they are encoded from the normalized copy;
* the env's offer order differs from ``driver.canonical_moves``'s whenever
  a bridge is on offer. That is fine for every consumer: agents score
  candidates individually and are handed back their chosen move's index, so
  no checkpoint or dataset depends on the order itself.
"""

from __future__ import annotations

from dataclasses import replace

from bgai.data.ledger_parser import ParsedCommand

__all__ = ["UNORDERED_PAIR_VERBS", "canonical_offer", "normalize_move", "sort_key"]

UNORDERED_PAIR_VERBS = frozenset({"bridge"})
"""Verbs whose ``(loc, loc2)`` is a set, not a sequence.

Only ``bridge``: a bridge spans two hexes with no direction (``state.bridges``
is itself a ``frozenset[frozenset[str]]``). ``connect`` carries a single
``loc``, and every other two-location verb in the grammar is directional.
Listing the verbs explicitly rather than swapping any ``loc > loc2`` pair
keeps the normalization from silently reordering a verb where orientation
means something.
"""


def normalize_move(move: ParsedCommand) -> ParsedCommand:
    """A copy with any unordered hex pair spelled low-to-high.

    Returns ``move`` itself when there is nothing to normalize, so this is
    free for the ~99.9% of offers with no bridge in them.
    """
    if move.verb not in UNORDERED_PAIR_VERBS:
        return move
    if move.loc is None or move.loc2 is None or move.loc <= move.loc2:
        return move
    return replace(move, loc=move.loc2, loc2=move.loc)


def sort_key(move: ParsedCommand) -> tuple[object, ...]:
    """Total order over commands, using every field the grammar can set.

    Same field list as ``driver.canonical_moves`` (so the two agree
    wherever the engine's own spelling is already canonical), with ``None``
    mapped to a sentinel so ints and strings never compare against None.
    """
    return (
        move.verb,
        move.loc or "",
        move.loc2 or "",
        move.building or "",
        move.tile or "",
        move.cult or "",
        move.color or "",
        move.target or "",
        move.res1 or "",
        move.res2 or "",
        move.n1 if move.n1 is not None else -1,
        move.n2 if move.n2 is not None else -1,
        move.reason or "",
        move.kind.value,
    )


def canonical_offer(
    offer: tuple[ParsedCommand, ...],
) -> tuple[tuple[ParsedCommand, ...], tuple[ParsedCommand, ...]]:
    """``(normalized offer, engine originals)``, both in the same stable order.

    Index i of the first tuple is what the observation encodes and what an
    action index names; index i of the second is what gets handed to
    ``driver.advance``.
    """
    rows = sorted(
        ((normalize_move(move), move) for move in offer),
        key=lambda row: sort_key(row[0]),
    )
    return tuple(row[0] for row in rows), tuple(row[1] for row in rows)
