"""Legal-move generation: ``legal_moves(state) -> tuple[ParsedCommand, ...]``.

Task 15, the final implementation task. A pure query over the frozen
engine (``state.py``/``apply.py``/every action module) -- no new
``HANDLERS`` entries, no state mutation. Consumed by agents (MCTS, LLM
tools) that need "what can I legally do right now" without hand-rolling
per-verb affordability/reachability checks themselves.

--------------------------------------------------------------------------
Design decision 1: pending-decision answers are additive, not a gate
--------------------------------------------------------------------------

An earlier revision of this module treated a queued ``leech``/
``cult_choice``/``gain_favor``/``gain_town``/``convert_w_to_p`` pending as
an *exclusive* gate -- "queue nonempty -> enumerate *only* the head
decision's answers" (the task brief's literal wording). Containment
testing against real games falsified that reading: ``4pLeague_S27_D1L1_G1``
row where cultists holds a live ``cult_choice`` pending (from an earlier
accepted leech) still submits an ordinary ``pass tile=BON7`` main-track
row in between -- and ``apply.py`` genuinely allows this (``pass`` only
checks ``faction == active_faction(state)``, never "does this faction have
an outstanding pending"; ``gain_cult``'s own unconditional membership in
``_ORDER_EXEMPT_VERBS`` means the pending's *answer* is exempt from turn
order, not that everything else is blocked until it's resolved). So every
one of :data:`_BLOCKING_PENDING_KINDS` (``leech``, ``cult_choice``,
``gain_favor``, ``gain_town``, ``convert_w_to_p``) is **additive**: its
answers are unioned into whatever else is legal for ``faction`` right now
(main-track ACTIONS-phase moves if active, the baseline otherwise),
never a replacement for them. ("Blocking" is kept as this set's name for
historical/diff-locality reasons -- what it actually marks is "these kinds
have a small, enumerable answer set worth generating", not "these kinds
exclude everything else.")

``state.pending`` still separates a second, genuinely different thing
(``leech.py``'s module docstring flags this distinction for
``cultist_leech_watch``; this generalizes it to every pending kind in play
as of Task 10):

- **One-shot marker pendings**: ``free_d``/``free_tp``/``free_tf``
  (ACTW/ACTS/ACTN) and ``bridge`` (ACT1/ACTE) are allowances that change
  how a *later* build/upgrade/transform/bridge command is priced -- they
  never had their own enumerable "answers" the way leech/gain_favor/etc.
  do (see ``legal_actions.py``'s module docstring for the full reasoning
  and the real corpus evidence, e.g. ``action ACTW. -FREE_D``, that a
  marker can be declined outright instead of used). These are folded into
  ``legal_actions.actions_phase_moves`` as pricing modifiers instead.
- **``cultist_leech_watch``** (Cultists' internal batch-tracking marker,
  ``leech.py``): never a player decision at all -- excluded entirely,
  never surfaced here, per the task brief's explicit callout.

--------------------------------------------------------------------------
Design decision 2: the exemption surface -- `legal_moves` vs.
`legal_moves_for`/`legal_moves_all`
--------------------------------------------------------------------------

``round_flow.active_faction``'s own docstring establishes that a queued
leech/cult_choice/gain_favor/gain_town/convert_w_to_p answer is legal for
*any* faction holding one, simultaneously with whichever faction is
structurally "active" (turn_order[active_index]) -- e.g. engineers can
answer a leech offer from nomads' build while mermaids is mid-turn.
``legal_moves(state)`` (the frozen, brief-mandated signature) stays
active-faction-only, for MCTS simplicity: one call, one faction's turn to
search over. The exemption surface -- every *other* live faction's own
outstanding blocking-decision answers -- is exposed separately:

- :func:`legal_moves_for(state, faction)`: the same enumeration logic,
  parameterized by faction. Identical to ``legal_moves(state)`` when
  ``faction == active_faction(state)``; for any other faction, returns
  that faction's own blocking-pending answers (if any) plus the
  order-exempt baseline every faction always has (``convert``/``wait``/
  ``done`` -- see Design decision 4 below) -- never main-track ACTIONS-phase
  verbs like build/upgrade, which stay gated to ``active_faction(state)``
  by ``apply()`` itself (``EngineError`` on "acted out of turn" otherwise).
- :func:`legal_moves_all(state) -> Mapping[str, tuple[ParsedCommand, ...]]`:
  every live (non-dropped) faction's current legal moves at once --
  the active faction's full set plus every other faction's exemption-only
  set (``{}`` for a faction with nothing outstanding).

--------------------------------------------------------------------------
Design decision 3: enumeration granularity
--------------------------------------------------------------------------

See ``legal_actions.py``'s module docstring for the ACTIONS-phase
granularity policy (dig/burn enumerated exactly; convert coarsened to
single-unit exchanges; transform enumerated over every budget-reachable
color). ``gain_town``'s ``cmd.n1`` (a *count* of simultaneous
same-tile-type pendings being resolved in one row, not a scaled amount --
``actions_build.handle_gain_town``'s own docstring) is generated here at
``n1=1`` only, one pending at a time -- a real ``+2TW1`` row (two
simultaneous same-tile picks) is a documented containment normalization
(decompose to two ``n1=1`` checks), the same policy ``convert`` uses for
amounts greater than one unit.

--------------------------------------------------------------------------
Design decision 4: the order-exempt baseline (``convert``/``wait``/``done``)
--------------------------------------------------------------------------

``apply.py``'s ``_ORDER_EXEMPT_VERBS`` marks ``convert`` and ``wait``
unconditionally exempt from the ``active_faction`` gate -- *any* live
faction may submit either at *any* time, any phase, active or not (that
set's own docstring: "a resource exchange is free and unconditionally
available whenever the game is in 'play' state, not gated by turn
rotation"; corpus rows bear this out for both -- e.g.
``4pLeague_S3_D3L3_G2`` row where engineers converts PW while swarmlings
is ``active_faction`` during ``Phase.INCOME``). So every
:func:`legal_moves_for` call computes this baseline once, unconditionally
for live factions (except ``Phase.FINISHED`` and dropped factions), and
unions it with any pending-decision answers to form the result -- an
additive model where both sets are available simultaneously. ``Phase.INCOME``/
``Phase.CLEANUP`` additionally allow a ``transform`` for any faction
sitting on an unspent ``spades_available`` balance -- ``apply.py``'s own
docstring cites the corpus mechanism (a cult-income SPADE grant forces an
immediate out-of-turn transform before the income batch finishes) --
``noop``/``convert`` cover the rest of what's observed during those two
phases. ``resign`` is excluded from the baseline for the same reason
``legal_actions.exempt_noop_moves`` excludes it (see that function's
docstring). ``done`` (unlike ``wait``) is *not* order-exempt in
``apply.py`` -- it only appears when ``faction`` may otherwise act
(``legal_actions.active_noop_moves``, folded in alongside the
phase-specific enumeration below, not the baseline).
"""

from __future__ import annotations

from collections.abc import Mapping

from bgai.data.ledger_parser import Kind, ParsedCommand
from bgai.engine.tm.factions_data import CULTS, FACTIONS
from bgai.engine.tm.legal_actions import (
    actions_phase_moves,
    active_noop_moves,
    convert_moves,
    exempt_noop_moves,
    transform_moves,
)
from bgai.engine.tm.legal_shared import cmd
from bgai.engine.tm.state import GameState, PendingDecision, Phase, active_faction
from bgai.engine.tm.tiles import FAVOR_TILES, TOWN_TILES

# See module docstring, Design decision 1.
_BLOCKING_PENDING_KINDS = frozenset(
    {"leech", "cult_choice", "gain_favor", "gain_town", "convert_w_to_p"}
)


# --------------------------------------------------------------------------
# Blocking pending-decision answers
# --------------------------------------------------------------------------


def _leech_answers(pending: PendingDecision) -> tuple[ParsedCommand, ...]:
    """``leech N`` / ``decline`` -- accepting the offer's cached amount is
    the canonical accept answer (``handle_leech`` recomputes the real cap
    fresh at accept time regardless of the requested ``n1``, so any
    positive ``n1`` naming the right offer resolves identically -- see
    ``leech.py``'s ``_find_leech_pending`` docstring; this is why the
    containment test treats ``leech``'s exact ``n1`` as non-load-bearing).
    """
    return (
        cmd("leech", n1=pending.amount, target=pending.source),
        cmd("decline", target=pending.source),
    )


def _cult_choice_answers(pending: PendingDecision) -> tuple[ParsedCommand, ...]:
    """Cultists' leech "taken" effect: pick one cult track to advance by
    ``pending.amount`` (always 1 -- ``leech.py``'s ``_CultistsHooks``).
    Parsed/generated as ``gain_cult`` (``Kind.BOOKKEEPING`` in the ledger
    grammar -- most ``+N<CULT>`` rows are automatic companions, not player
    picks -- so this is not part of the DECISION-kind containment check,
    but it is a genuine agent-facing choice and the task brief explicitly
    asks for it).
    """
    return tuple(cmd("gain_cult", kind=Kind.BOOKKEEPING, cult=c, n1=pending.amount) for c in CULTS)


def _gain_favor_answers(
    state: GameState, faction: str, pending: PendingDecision
) -> tuple[ParsedCommand, ...]:
    fs = state.factions[faction]
    return tuple(
        cmd("gain_favor", tile=t)
        for t in FAVOR_TILES
        if state.favors_pool.get(t, 0) > 0 and t not in fs.favors
    )


def _gain_town_answers(state: GameState, pending: PendingDecision) -> tuple[ParsedCommand, ...]:
    return tuple(
        cmd("gain_town", tile=t, n1=1) for t in TOWN_TILES if state.towns_pool.get(t, 0) > 0
    )


def _convert_w_to_p_answers(
    state: GameState, faction: str, pending: PendingDecision
) -> tuple[ParsedCommand, ...]:
    fs = state.factions[faction]
    if pending.amount >= 1 and fs.workers >= 1:
        return (cmd("convert", res1="W", n1=1, res2="P", n2=1),)
    return ()


def _pending_answers(
    state: GameState, faction: str, pending: PendingDecision
) -> tuple[ParsedCommand, ...]:
    if pending.kind == "leech":
        return _leech_answers(pending)
    if pending.kind == "cult_choice":
        return _cult_choice_answers(pending)
    if pending.kind == "gain_favor":
        return _gain_favor_answers(state, faction, pending)
    if pending.kind == "gain_town":
        return _gain_town_answers(state, pending)
    if pending.kind == "convert_w_to_p":
        return _convert_w_to_p_answers(state, faction, pending)
    return ()  # unreachable for kinds in _BLOCKING_PENDING_KINDS


# --------------------------------------------------------------------------
# Setup phases
# --------------------------------------------------------------------------


def _setup_dwelling_moves(state: GameState, faction: str) -> tuple[ParsedCommand, ...]:
    """``Phase.SETUP_DWELLINGS``: any unoccupied, already-home-colored hex
    -- ``handle_build``'s ``setup`` branch skips cost/reachability
    entirely and hard-errors on a color mismatch (``actions_build.py``).
    """
    fs = state.factions[faction]
    color = FACTIONS[faction].color
    d_track = FACTIONS[faction].buildings["D"]
    if len(fs.buildings["D"]) >= d_track.max_count:
        return ()
    return tuple(
        cmd("build", loc=hex_key)
        for hex_key, hx in state.hexes.items()
        if hx.building is None and hx.color == color
    )


def _setup_bonus_moves(state: GameState, faction: str) -> tuple[ParsedCommand, ...]:
    """``Phase.SETUP_BONUS``: pick any not-yet-held starting bonus tile
    (``handle_pass``'s ``_handle_setup_bonus_pick`` branch).
    """
    held = {o.bonus for o in state.factions.values() if o.bonus is not None}
    return tuple(cmd("pass", tile=tile) for tile in state.setup.bonus_tiles if tile not in held)


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


def _own_pending_answers(
    state: GameState, faction: str, seen: set[ParsedCommand]
) -> tuple[ParsedCommand, ...]:
    """Union of every one of ``faction``'s own outstanding
    :data:`_BLOCKING_PENDING_KINDS` answer sets, deduped against ``seen``
    (mutated in place) and each other -- module docstring, Design
    decision 1: more than one instance of the *same* kind can be
    simultaneously outstanding (two leech offers from two different
    builders, two gain_town pendings from two clusters that qualified at
    once), and every one is independently answerable right now, in any
    order.
    """
    answers: list[ParsedCommand] = []
    for pending in state.pending:
        if pending.faction != faction or pending.kind not in _BLOCKING_PENDING_KINDS:
            continue
        for answer in _pending_answers(state, faction, pending):
            if answer not in seen:
                seen.add(answer)
                answers.append(answer)
    return tuple(answers)


def legal_moves_for(state: GameState, faction: str) -> tuple[ParsedCommand, ...]:
    """Legal moves for an arbitrary faction (module docstring, Design
    decisions 1 and 2): the order-exempt baseline, plus any outstanding
    pending-decision answers, plus -- only when ``faction`` is
    ``active_faction(state)`` -- ordinary phase-specific main-track moves.
    Identical to ``legal_moves(state)`` when ``faction`` is
    ``active_faction(state)``.
    """
    if state.phase == Phase.FINISHED:
        return ()
    fs = state.factions.get(faction)
    if fs is None or fs.dropped:
        return ()

    # Order-exempt baseline (module docstring, Design decision 4):
    # convert/wait are legal for any live faction regardless of phase,
    # active-faction status, or an outstanding pending. `done` is a
    # separate, narrower case -- see `active_noop_moves`'s docstring.
    baseline = tuple(convert_moves(state, faction, fs)) + tuple(exempt_noop_moves())
    seen: set[ParsedCommand] = set(baseline)
    pending_answers = _own_pending_answers(state, faction, seen)

    if state.phase in (Phase.INCOME, Phase.CLEANUP):
        # Also order-exempt during these two phases specifically: every
        # verb (including `done`) is phase-exempt here, and a live spade
        # balance forces an immediate transform before the batch proceeds
        # (Design decision 4's citation).
        extra = transform_moves(state, faction, fs) if fs.spades_available > 0 else ()
        return baseline + pending_answers + tuple(active_noop_moves()) + tuple(extra)

    if faction != active_faction(state):
        return baseline + pending_answers
    if fs.passed:
        # Defensive only: round_flow._advance_actions never lets
        # active_faction(state) point at a passed faction during real
        # play, so this never fires on an engine-driven state -- kept for
        # a hand-constructed/malformed one (task brief's "passed faction
        # ... skipped" hand test).
        return baseline + pending_answers
    if state.phase == Phase.SETUP_DWELLINGS:
        return (
            baseline
            + pending_answers
            + tuple(active_noop_moves())
            + _setup_dwelling_moves(state, faction)
        )
    if state.phase == Phase.SETUP_BONUS:
        return (
            baseline
            + pending_answers
            + tuple(active_noop_moves())
            + _setup_bonus_moves(state, faction)
        )
    if state.phase == Phase.ACTIONS:
        return (
            baseline
            + pending_answers
            + tuple(active_noop_moves())
            + actions_phase_moves(state, faction)
        )
    return baseline + pending_answers + tuple(active_noop_moves())


def legal_moves(state: GameState) -> tuple[ParsedCommand, ...]:
    """Legal moves for ``active_faction(state)`` (frozen API, task brief).

    A passed faction, a dropped faction, or ``Phase.FINISHED`` all yield
    ``()``. See the module docstring for the pending-decision/marker
    split and the multi-faction exemption surface (:func:`legal_moves_for`
    / :func:`legal_moves_all`) this single-faction signature deliberately
    leaves out.
    """
    if state.phase == Phase.FINISHED:
        return ()
    return legal_moves_for(state, active_faction(state))


def legal_moves_all(state: GameState) -> Mapping[str, tuple[ParsedCommand, ...]]:
    """Every live faction's current legal moves at once (module docstring,
    Design decision 2) -- the active faction's full set, plus every other
    live faction's own exemption-only blocking-decision answers.
    """
    return {
        name: legal_moves_for(state, name) for name, fs in state.factions.items() if not fs.dropped
    }
