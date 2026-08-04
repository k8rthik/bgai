"""Leech offers: enqueue-on-build, accept/decline, and the Cultists hook.

Ported from the reference implementation (jsnell/terra-mystica, MIT):

- ``map.pm`` ``compute_leech`` (lines 450-471): given a builder and a hex,
  sum ``%building_strength`` (``towns.py`` ``building_power_value``) over
  every hex directly adjacent (``$map{$where}{adjacent}``, which already
  has bridge endpoints folded in by ``command_bridge`` -- this module uses
  ``connectivity.directly_adjacent`` for the same reason) that carries a
  building of a *different* color than the builder's, bucketed **by
  color** (not by faction). Returns ``()`` outright during setup
  (``return () if !$game{round}``, line 455) -- ``offers_for_build`` below
  mirrors that with an explicit ``state.round == 0`` guard.
- ``resources.pm`` ``note_leech`` (lines 405-442): walks
  ``factions_in_order_from($from_faction, 1)`` -- i.e. seat order starting
  *at* the builder and wrapping around (the builder's own color is never a
  key in ``compute_leech``'s result, so this is effectively "starting
  after the builder") -- and for every faction whose color has a nonzero
  entry, queues a ``type => 'leech'`` decision UNCONDITIONALLY (line 423),
  **regardless of `actual`** (line 418: ``actual = min(raw, P1*2+P2)``).
  Only the raw amount being present gates whether an offer exists at all;
  a faction whose bowls can't hold *any* of it (``actual == 0``) still
  gets an offer, it is just inert. This module ports that literally:
  ``offers_for_build`` does **not** skip zero-``gainable()`` factions --
  see the "zero-cap" note below for why the task brief's tentative
  "auto-skip" assumption is wrong.
- ``commands.pm`` ``command_leech`` (lines 419-513): ``actual_pw`` is first
  capped by the faction's remaining VP (``if pw > VP: pw = VP + 1``, so
  the VP payment ``actual_pw - 1`` never exceeds the VP on hand -- the
  "VP floor" from the task brief), then passed through ``gain_power``
  (``resources.pm`` lines 172-188), which is exactly
  ``power.py``'s ``Power.gain`` bowl-token walk (``2*bowl1 + bowl2`` is
  the max number of *steps* -- not the final bowl3 delta -- that can be
  "spent"; a bowl1 token still costs a VP-bearing step even though it
  only reaches bowl2, not bowl3). ``vp = actual_pw - 1``; VP is only
  docked ``if actual_pw > 0`` (line 487) -- a zero-effect accept is free.
- ``commands.pm`` ``command_decline`` (lines 515-556): a bare ``decline``
  (no amount/from) declines *every* outstanding leech offer for the
  faction; a specific ``decline N from X`` declines just that one.
- ``strict-leech`` option (``command_leech`` line 467, ``commands.pm``):
  gates a single narrow case -- a leech row that arrives *after* another
  command already "tainted" it (``leech_tainted``, set at lines 372-373
  when a ``convert`` happens while a leech decision is still outstanding,
  or at 440-441 on an amount/from-faction mismatch): without the option
  it is just an ``$ledger->warn``; with it, a hard `die`. This engine has
  no notion of a `convert` staling a leech decision (Task 7's
  ``handle_convert`` doesn't check for outstanding leech pendings), so
  ``strict_leech`` has **no observable effect here yet** -- flagged as a
  seam for whichever later task wires resource-conversion/leech ordering.
- Cultists timing (``leech_effect``, ``factions_data.py``): the "taken"
  effect (``command_leech`` lines 446-458) fires **unconditionally, the
  very first time** *any* offer from one build is accepted with
  ``actual > 0`` (deduped per build via ``leech_cult_gained``, keyed by
  the Perl ``leech_id`` counter) -- **immediately, mid-batch**, not gated
  by ``errata-cultist-power``, and not deferred until every offer from
  that build has answered. The "not_taken" effect
  (``cultist_maybe_gain_power``, lines 558-586) only fires once the *last*
  offer with ``actual > 0`` from that build has been explicitly declined
  (``leech_not_rejected`` reaches 0, i.e. every such offer was declined),
  and only under ``errata-cultist-power``.

  This module ports that timing directly (a prior revision deferred the
  "taken" effect to batch-end to match the task brief's simplified
  wording, but that diverges from a real corpus replay: a game can log
  the Cultists' ``+CULT`` row before every opponent has answered that
  build's leech offers -- fixed per code review). A single
  ``"cultist_leech_watch"`` bookkeeping ``PendingDecision`` per
  originating build (pushed by :func:`queue_leech` alongside the batch of
  offers, ``amount`` = count of nonzero-amount offers, ``source`` =
  builder name) still tracks *how many offers remain unanswered* (needed
  to detect "the last offer was just declined" for the "not_taken" case)
  and *whether the "taken" effect already fired this batch* (via
  ``options``, so a second accept in the same batch doesn't push a second
  ``cult_choice``) -- but the effect itself now fires through
  ``factions.hooks.HOOKS["cultists"].on_leech_resolved`` (see that hook's
  docstring), called exactly at the two Perl-faithful trigger points
  (first accept; last-offer-of-an-all-declined-batch), not once per raw
  offer resolution. This satisfies the brief's literal instruction
  ("Cultists: ``HOOKS["cultists"].on_leech_resolved``") which the prior
  revision left unwired.

**Zero-cap decision** (task brief: "capped by Power.gainable(); no offer
if cap is 0 (verify vs Perl)"): verified above against ``note_leech`` --
Perl does *not* skip zero-``gainable()`` offers, so neither does this
module. ``PendingDecision.amount`` here holds ``min(raw_building_power,
gainable())`` at offer-creation time -- Perl's "amount"/"actual" split
collapses to one number since this engine doesn't track a separate
"declared vs. capped" pair the way the Perl ``action_required`` record
does. ``handle_leech`` re-derives *both* the VP-floor cap and a **fresh**
``gainable()`` cap at accept time (Perl's ``gain_power`` always reads
live ``P1``/``P2`` bowls, not an offer-time snapshot -- see that
function's docstring).
"""

from __future__ import annotations

from dataclasses import replace

from bgai.data.ledger_parser import ParsedCommand
from bgai.engine.tm.apply import EngineError, pop_pending, push_pending, register_handler
from bgai.engine.tm.connectivity import directly_adjacent
from bgai.engine.tm.factions.hooks import HOOKS, FactionHooks, hooks_for
from bgai.engine.tm.factions_data import FACTIONS
from bgai.engine.tm.state import GameState, PendingDecision, with_faction
from bgai.engine.tm.towns import building_power_value

_WATCH_KIND = "cultist_leech_watch"
_FIRED_OPTION = "taken_effect_fired"


class _CultistsHooks(FactionHooks):
    """Cultists' ``leech_effect`` (``factions_data.py``): ``on_leech_resolved``
    is called by :func:`_resolve_cultist_watch` at exactly the two
    Perl-faithful trigger points (see module docstring) -- ``accepted=True``
    means "the first accepted offer of this build's batch just resolved"
    (``leech_effect["taken"]``, unconditional); ``accepted=False`` means
    "the last offer of an all-declined batch just resolved"
    (``leech_effect["not_taken"]``, gated by ``errata_cultist_power``).
    """

    def on_leech_resolved(self, state: GameState, faction: str, accepted: bool) -> GameState:
        if accepted:
            return push_pending(
                state, PendingDecision(faction=faction, kind="cult_choice", amount=1)
            )
        if not state.setup.options.errata_cultist_power:
            return state
        fs = state.factions[faction]
        return with_faction(state, faction, replace(fs, power=fs.power.gain(1)))


HOOKS["cultists"] = _CultistsHooks()


def offers_for_build(state: GameState, builder: str, hex_key: str) -> tuple[PendingDecision, ...]:
    """Leech offers triggered by ``builder`` building/upgrading at ``hex_key``.

    One offer per faction with a different-color building directly
    adjacent (incl. bridges) to ``hex_key``; ``amount`` = that faction's
    adjacent building power summed by color, capped by
    ``Power.gainable()``. Enqueued in seat order starting after
    ``builder`` (``factions_in_order_from``, see module docstring). See
    the module docstring for why zero-``gainable()`` factions still get an
    offer.
    """
    if state.round == 0:
        return ()

    builder_color = FACTIONS[builder].color
    raw_by_color: dict[str, int] = {}
    for neighbor in directly_adjacent(state, hex_key):
        hex_state = state.hexes[neighbor]
        if hex_state.building is None or hex_state.color == builder_color:
            continue
        raw_by_color[hex_state.color] = raw_by_color.get(hex_state.color, 0) + building_power_value(
            hex_state.building
        )

    if not raw_by_color:
        return ()

    start = state.turn_order.index(builder)
    seat_order = state.turn_order[start:] + state.turn_order[:start]

    offers = []
    for faction in seat_order:
        if faction == builder:
            continue
        raw = raw_by_color.get(FACTIONS[faction].color)
        if not raw:
            continue
        capped = min(raw, state.factions[faction].power.gainable())
        offers.append(PendingDecision(faction=faction, kind="leech", amount=capped, source=builder))
    return tuple(offers)


def queue_leech(
    state: GameState,
    builder: str,
    hex_key: str,
    offers: tuple[PendingDecision, ...] | None = None,
) -> GameState:
    """Push ``offers_for_build``'s offers, plus a Cultists watch marker.

    ``offers`` may be passed precomputed (``actions_build.py`` needs the
    same offers to decide the TP-upgrade neighbour discount, so it calls
    ``offers_for_build`` itself and hands the result in here to avoid
    recomputing).
    """
    if offers is None:
        offers = offers_for_build(state, builder, hex_key)
    new_state = push_pending(state, *offers)

    if FACTIONS[builder].leech_effect:
        nonzero = sum(1 for o in offers if o.amount > 0)
        if nonzero:
            watch = PendingDecision(
                faction=builder, kind=_WATCH_KIND, amount=nonzero, source=builder
            )
            new_state = push_pending(new_state, watch)
    return new_state


def _find_leech_pending(state: GameState, faction: str, cmd: ParsedCommand) -> int:
    """Locate the queued ``leech`` offer a ``leech``/``decline`` row
    answers. ``cmd.target`` (the ``from X`` clause), when present, fully
    disambiguates which offer -- ``queue_leech`` never pushes more than
    one offer per source per faction -- so ``cmd.n1`` is *not* also
    required to equal the offer's cached ``amount`` in that case: a
    ``leech N from X`` row's ``N`` is a *request*, capped down by
    ``handle_leech`` itself (``gainable()``/VP-floor/the offer's own
    amount, module docstring) if it exceeds what the faction can actually
    still take -- which is exactly what an offer's ``amount`` (capped at
    *offer-creation* time) can diverge from by the time it's answered
    (task-13 report, reference-game row 155: nomads' F3+G2 dwellings
    together raise 2 raw power against engineers' F4 build, but nomads'
    ``gainable()`` had already dropped to 1 by offer-creation time, so the
    queued offer's ``amount`` is 1 while the ledger row still reads
    ``leech 2 from engineers`` -- the corpus's own "greedy" request
    number, not a promise). Only a bare ``leech N`` with no ``from``
    clause (early-era logs) still needs ``amount`` to disambiguate between
    multiple simultaneous offers.
    """
    for i, p in enumerate(state.pending):
        if p.faction != faction or p.kind != "leech":
            continue
        if cmd.target is not None:
            if p.source != cmd.target:
                continue
        elif cmd.n1 is not None and p.amount != cmd.n1:
            continue
        return i
    raise EngineError(
        f"no queued leech offer for {faction} matching {cmd.raw!r}",
        state=state,
        faction=faction,
        cmd=cmd,
    )


def _resolve_cultist_watch(
    state: GameState, resolved: PendingDecision, *, accepted: bool
) -> GameState:
    """Update the Cultists watch marker for ``resolved``'s batch, if any,
    firing ``HOOKS[builder].on_leech_resolved`` at the two Perl-faithful
    trigger points (module docstring): immediately on the first accept
    (``accepted=True``, regardless of how many offers in the batch remain
    unanswered), and once on an all-declined batch's last offer
    (``accepted=False``, only if the "taken" effect never fired).

    No-op for offers with ``amount == 0`` (Perl: only ``actual > 0``
    offers participate in ``leech_not_rejected``/``leech_rejected``) and
    for builds whose builder has no ``leech_effect`` (no watch was ever
    pushed).
    """
    if resolved.amount <= 0:
        return state

    idx = next(
        (
            i
            for i, p in enumerate(state.pending)
            if p.kind == _WATCH_KIND and p.source == resolved.source
        ),
        None,
    )
    if idx is None:
        return state

    watch = state.pending[idx]
    builder = watch.faction
    remaining = watch.amount - 1
    already_fired = _FIRED_OPTION in watch.options

    new_state = state
    if accepted and not already_fired:
        new_state = hooks_for(builder).on_leech_resolved(new_state, builder, True)
        already_fired = True

    pending = new_state.pending
    # The watch's own index is unaffected by the hook call above: it only
    # ever appends (push_pending) or replaces a *different* faction's
    # resources, never removes/reorders entries ahead of `idx`.
    if remaining > 0:
        new_watch = replace(
            watch, amount=remaining, options=(_FIRED_OPTION,) if already_fired else ()
        )
        return replace(new_state, pending=pending[:idx] + (new_watch,) + pending[idx + 1 :])

    new_state = pop_pending(new_state, idx)
    if not already_fired:
        new_state = hooks_for(builder).on_leech_resolved(new_state, builder, False)
    return new_state


def handle_leech(state: GameState, faction: str, cmd: ParsedCommand) -> GameState:
    """``leech N[ from X]``: accept up to ``N`` power, pay ``N-1`` VP.

    ``N`` is capped three ways, matching ``command_leech``/``gain_power``
    (module docstring): by the offer's own ``amount`` (can't claim more
    than was on offer), by the faction's remaining VP
    (``min(requested, fs.vp + 1)``), and by the faction's **current**
    ``Power.gainable()`` -- recomputed fresh here rather than trusting
    ``pending.amount``'s offer-time snapshot, since Perl's ``gain_power``
    always reads live ``P1``/``P2`` bowls at accept time: if this faction
    accepted an earlier offer from the same batch (or otherwise spent
    power) since this offer was queued, its gainable capacity may now be
    lower than what was cached on the pending. A zero-effect accept
    (``actual == 0``) costs no VP.
    """
    idx = _find_leech_pending(state, faction, cmd)
    pending = state.pending[idx]
    fs = state.factions[faction]

    requested = cmd.n1 if cmd.n1 is not None else pending.amount
    actual = min(requested, pending.amount, fs.power.gainable(), fs.vp + 1)
    actual = max(actual, 0)

    new_fs = fs
    if actual > 0:
        new_fs = replace(fs, power=fs.power.gain(actual), vp=fs.vp - (actual - 1))

    new_state = with_faction(state, faction, new_fs)
    new_state = replace(new_state, pending=new_state.pending[:idx] + new_state.pending[idx + 1 :])
    return _resolve_cultist_watch(new_state, pending, accepted=True)


def _decline_one(state: GameState, faction: str, cmd: ParsedCommand) -> GameState:
    idx = _find_leech_pending(state, faction, cmd)
    pending = state.pending[idx]
    new_state = pop_pending(state, idx)
    return _resolve_cultist_watch(new_state, pending, accepted=False)


def handle_decline(state: GameState, faction: str, cmd: ParsedCommand) -> GameState:
    """``decline[ N from X]``: a bare ``decline`` rejects every outstanding
    offer for ``faction`` (``command_decline``, module docstring); a
    specific ``decline N from X`` rejects just that one.
    """
    if cmd.n1 is None and cmd.target is None:
        new_state = state
        while any(p.faction == faction and p.kind == "leech" for p in new_state.pending):
            new_state = _decline_one(new_state, faction, cmd)
        return new_state
    return _decline_one(state, faction, cmd)


register_handler("leech", handle_leech)
register_handler("decline", handle_decline)
