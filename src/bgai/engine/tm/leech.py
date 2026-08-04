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
  to detect "the last offer was just declined") and *whether the "taken"
  effect already fired this batch* (via ``options``, so a second accept
  in the same batch doesn't push a second ``cult_choice``) -- the "taken"
  effect itself fires through
  ``factions.hooks.HOOKS["cultists"].on_leech_resolved(..., accepted=True)``
  (see that hook's docstring), called at the first-accept trigger point,
  not once per raw offer resolution. This satisfies the brief's literal
  instruction ("Cultists: ``HOOKS["cultists"].on_leech_resolved``") which
  a prior revision left unwired.

- **"not_taken" is granted directly off the ledger's own bracket row, not
  inferred from the resolving decline.** ``add_row_for_effect``
  (``ledger.pm`` 119-131) is an *unbuffered* immediate push straight onto
  the ledger's row array, while the ordinary row a command produces is
  only pushed later, when that row's *buffered* collector flushes
  (``finish_row``, ``ledger.pm`` 78-112, called once at the end of
  processing a full submitted move). Since ``cultist_maybe_gain_power``
  (called synchronously *during* ``command_decline``'s own execution,
  commands.pm 537) uses the unbuffered path, its bracket row
  (``"[all opponents declined power]"``) always lands **one position
  before** the resolving ``decline`` row's own summary in final ledger
  order -- even though the decline logically, causally precedes it.
  Verified against the corpus (task-13 report follow-up,
  ``4pLeague_S10_D1L1_G5`` row 208): the reference's ``deltas.parquet``
  checkpoint at the bracket row already reflects the +1 PW, one full row
  before the ``decline`` command that (in this engine's original design)
  would have driven it. Inferring the gain from ``handle_decline`` alone
  therefore always lands it one row late.

  Fixed by having ``ledger_parser.py`` recognize this exact bracket text
  as its own verb (``cultist_leech_bonus``, not the generic
  ``annotation``) and granting ``FACTIONS[faction].leech_effect
  ["not_taken"]`` directly off *that* row (:func:`handle_cultist_leech_bonus`
  below) -- the row's own ``faction`` field is already the Cultists
  player, so no offer/pending lookup is even needed.
  ``_resolve_cultist_watch``'s ``accepted=False`` branch still walks the
  watch marker down to zero and pops it (state hygiene: no dangling
  ``cultist_leech_watch`` pending once a batch is fully answered) but no
  longer grants any resource itself -- ``HOOKS["cultists"]
  .on_leech_resolved(..., accepted=False)`` is now a no-op passthrough.

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
    is called by :func:`_resolve_cultist_watch` at the accept trigger point
    (see module docstring) -- ``accepted=True`` means "the first accepted
    offer of this build's batch just resolved" (``leech_effect["taken"]``,
    unconditional: pushes a ``cult_choice`` pending, resolved by a later
    explicit ``+CULT`` row). ``accepted=False`` ("the last offer of an
    all-declined batch just resolved") is a no-op here: ``leech_effect
    ["not_taken"]`` is granted directly off the ledger's own
    ``"[all opponents declined power]"`` bracket row instead
    (:func:`handle_cultist_leech_bonus`, module docstring) -- inferring it
    from the resolving decline would land it one ledger row late.
    """

    def on_leech_resolved(self, state: GameState, faction: str, accepted: bool) -> GameState:
        if accepted:
            return push_pending(
                state, PendingDecision(faction=faction, kind="cult_choice", amount=1)
            )
        return state


HOOKS["cultists"] = _CultistsHooks()


def _raw_leech_by_color(state: GameState, builder: str, hex_key: str) -> dict[str, int]:
    """``map.pm``'s ``compute_leech`` (module docstring): every adjacent
    hex's building power, bucketed by color, for a *different*-colored
    building than ``builder``'s -- purely geometric (board state only),
    independent of turn order, seat order, or any faction's ``dropped``
    status. ``()`` during setup (``return () if !$game{round}``, same as
    the caller-level ``state.round == 0`` guard below).
    """
    if state.round == 0:
        return {}
    builder_color = FACTIONS[builder].color
    raw_by_color: dict[str, int] = {}
    for neighbor in directly_adjacent(state, hex_key):
        hex_state = state.hexes[neighbor]
        if hex_state.building is None or hex_state.color == builder_color:
            continue
        raw_by_color[hex_state.color] = raw_by_color.get(hex_state.color, 0) + building_power_value(
            hex_state.building
        )
    return raw_by_color


def has_leechable_neighbor(state: GameState, builder: str, hex_key: str) -> bool:
    """Whether ``hex_key`` has *any* adjacent different-color building --
    ``commands.pm``'s ``command_upgrade`` TP-upgrade isolated-surcharge
    check (``if (!keys %this_leech)``, ~288, where ``%this_leech`` is
    ``note_leech``'s return, itself ``compute_leech``'s -- commands.pm
    280 near ``note_leech``). This is a **different** question than
    "does any offer exist" (``offers_for_build`` below): ``map.pm``'s
    ``compute_leech`` (~457-468) sums a color's building strength into
    ``%this_leech`` unconditionally, *before* it ever looks up which
    living faction currently holds that color
    (``grep {$_->{color} eq $map_color} factions_in_order(1)`` -- the
    ``no_dummy=1`` filter that excludes dropped factions, ~461-463) --
    that lookup's result only feeds the per-faction ``building_strength``
    override fallback, it never gates whether the color's entry gets
    added to ``%this_leech`` at all. So a build next to a *dropped*
    faction's still-standing building is correctly **not** isolated in
    real Perl, even though that dropped faction of course never receives
    an actual leech offer for it (a separate mechanism entirely --
    ``resources.pm``'s ``note_leech`` walks
    ``factions_in_order_from($from, 1)``, the same ``no_dummy`` filter,
    to decide *who* gets offered one).

    Task-14 corpus fix: an earlier revision used ``bool(offers_for_build(
    ...))`` for this check, conflating the two -- wrong the moment a
    dropped faction's color falls out of ``offers_for_build``'s own
    seat-order walk (``state.turn_order``, which shrinks to only
    currently-passing/live factions every round under
    ``variable-turn-order``, ``round_flow.end_of_round``'s own
    ``passed_order`` branch -- a dropped faction never explicitly passes
    again, so it simply stops appearing there). Corpus:
    ``4pLeague_S64_D1L1_G6`` row 348, nomads' ``upgrade E3 to TP``: E3 is
    directly adjacent to two of *dropped* alchemists' still-standing
    dwellings (E2, D2) -- the real ledger charges the un-isolated 3 C,
    not the doubled 6 C ``bool(offers_for_build(...))`` produced once
    alchemists (dropped at row 276) had already fallen out of
    ``turn_order``.
    """
    return bool(_raw_leech_by_color(state, builder, hex_key))


def offers_for_build(state: GameState, builder: str, hex_key: str) -> tuple[PendingDecision, ...]:
    """Leech offers triggered by ``builder`` building/upgrading at ``hex_key``.

    One offer per **live** (not ``FactionState.dropped``) faction with a
    different-color building directly adjacent (incl. bridges) to
    ``hex_key``; ``amount`` = that faction's adjacent building power
    summed by color, capped by ``Power.gainable()``. Enqueued in seat
    order starting after ``builder`` (``factions_in_order_from``, see
    module docstring) -- ``resources.pm``'s ``note_leech`` walks
    ``factions_in_order_from($from, 1)``, the ``no_dummy=1`` variant that
    excludes dropped factions (``acting.pm``'s own ``!dropped`` filter),
    so a dropped faction never receives an offer here even though its
    still-standing building's color can still make a build "not isolated"
    for cost purposes -- see ``has_leechable_neighbor`` above for that
    separate check. See the module docstring for why zero-``gainable()``
    factions still get an offer.
    """
    raw_by_color = _raw_leech_by_color(state, builder, hex_key)
    if not raw_by_color:
        return ()

    start = state.turn_order.index(builder)
    seat_order = state.turn_order[start:] + state.turn_order[:start]

    offers = []
    for faction in seat_order:
        if faction == builder or state.factions[faction].dropped:
            continue
        raw = raw_by_color.get(FACTIONS[faction].color)
        if not raw:
            continue
        capped = min(raw, state.factions[faction].power.gainable())
        offers.append(
            PendingDecision(
                faction=faction, kind="leech", amount=capped, source=builder, options=(hex_key,)
            )
        )
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
                faction=builder, kind=_WATCH_KIND, amount=nonzero, source=builder,
                options=(hex_key,),
            )
            new_state = push_pending(new_state, watch)
    return new_state


def _find_leech_pending(state: GameState, faction: str, cmd: ParsedCommand) -> int:
    """Locate the queued ``leech`` offer a ``leech``/``decline`` row
    answers. ``cmd.n1`` (the requested amount) is a *request*, not a
    promise -- ``handle_leech`` itself caps the actual gain down
    (``gainable()``/VP-floor/the offer's own amount, module docstring) if
    it exceeds what the faction can actually still take, which is exactly
    what an offer's ``amount`` (capped at *offer-creation* time) can
    diverge from by the time it's answered. So ``cmd.n1``/``cmd.target``
    only need to *disambiguate* which offer a row means, never match it
    exactly:

    - Exactly one queued offer for ``faction``: that's the answer,
      regardless of what ``cmd.n1``/``cmd.target`` say (task-13 report,
      reference-game row 282: nomads' only offer is capped to amount 1,
      but the bare ``leech 4`` row -- no ``from`` clause at all -- still
      names 4, the corpus's own "greedy" request number).
    - Multiple queued offers: ``cmd.target`` (the ``from X`` clause), when
      present, disambiguates by source -- ordinarily to exactly one
      offer, since ``queue_leech`` never pushes more than one offer per
      source per faction *from a single build* (reference-game row 155:
      nomads' F3+G2 dwellings raise 2 raw power against engineers'
      build, but nomads' ``gainable()`` had already dropped to 1 by
      offer-creation time, so the offer's cached ``amount`` is 1 while
      the row reads ``leech 2 from engineers``). With no ``target``
      either (early-era logs), ``cmd.n1`` disambiguates by amount
      instead.
    - Multiple queued offers *from the same source* (task-14 fix: the
      same faction can build/upgrade a second time -- a second
      ``queue_leech`` call -- before every offer from its first build is
      answered, so ``cmd.target`` alone can leave more than one
      candidate): ``cmd.n1`` breaks the tie by amount, same as the no-
      ``target`` case, falling back to the first ``target`` match only
      if no offer's amount matches either (corpus
      ``4pLeague_S20_D1L1_G7`` row 216-217: engineers has two
      simultaneous ``leech ... from cultists`` offers, from two
      different Cultists builds; ``cmd.n1`` is what actually tells them
      apart).
    """
    matches = [
        i for i, p in enumerate(state.pending) if p.faction == faction and p.kind == "leech"
    ]
    if len(matches) == 1:
        return matches[0]

    candidates = matches
    if cmd.target is not None:
        target_matches = [i for i in matches if state.pending[i].source == cmd.target]
        if target_matches:
            candidates = target_matches

    if len(candidates) > 1 and cmd.n1 is not None:
        amount_matches = [i for i in candidates if state.pending[i].amount == cmd.n1]
        if amount_matches:
            candidates = amount_matches

    if candidates:
        return candidates[0]
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

    Matched by *both* ``source`` (the builder) and the originating
    build's ``hex_key`` (task-14 fix): the same Cultists faction can
    build/upgrade more than once before every offer from an earlier
    build has been answered, queuing two ``_WATCH_KIND`` pendings with
    the identical ``source="cultists"`` at once. Matching by ``source``
    alone (an earlier revision) always resolved the *first* (oldest,
    still-queued) matching watch regardless of which build the
    now-resolved offer actually came from -- silently updating the wrong
    batch's counter and, once the older batch's own last offer arrived,
    leaving the newer batch's "taken" effect (and its own eventual
    ``cult_choice`` pending) never fired at all: corpus
    ``4pLeague_S20_D1L1_G7`` row 217, cultists' second ``+CULT`` answer
    has no pending to consume and hard-errors "acted out of turn" purely
    because the strict gate no longer sees it as a queued cult_choice
    answer. ``options`` now carries the originating ``hex_key`` as its
    first entry on both the offer and the watch (``offers_for_build``/
    ``queue_leech``), surviving the watch's own ``_FIRED_OPTION`` append
    below.
    """
    if resolved.amount <= 0:
        return state

    hex_key = resolved.options[0] if resolved.options else None

    idx = next(
        (
            i
            for i, p in enumerate(state.pending)
            if p.kind == _WATCH_KIND
            and p.source == resolved.source
            and (hex_key is None or hex_key in p.options)
        ),
        None,
    )
    if idx is None:
        return state

    watch = state.pending[idx]
    builder = watch.faction
    remaining = watch.amount - 1
    already_fired = _FIRED_OPTION in watch.options
    watch_hex_key = next((o for o in watch.options if o != _FIRED_OPTION), None)

    new_state = state
    if accepted and not already_fired:
        new_state = hooks_for(builder).on_leech_resolved(new_state, builder, True)
        already_fired = True

    pending = new_state.pending
    # The watch's own index is unaffected by the hook call above: it only
    # ever appends (push_pending) or replaces a *different* faction's
    # resources, never removes/reorders entries ahead of `idx`.
    if remaining > 0:
        new_options = (watch_hex_key,) if watch_hex_key else ()
        if already_fired:
            new_options = new_options + (_FIRED_OPTION,)
        new_watch = replace(watch, amount=remaining, options=new_options)
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


def handle_cultist_leech_bonus(state: GameState, faction: str, cmd: ParsedCommand) -> GameState:
    """``"[all opponents declined power]"`` (``ledger_parser.py``'s
    ``cultist_leech_bonus`` rule, module docstring): grant
    ``FACTIONS[faction].leech_effect["not_taken"]`` (``{"PW": 1}`` for
    Cultists) directly to the row's own acting faction. No offer/pending
    lookup needed -- the ledger row itself already names who gets it and
    the raw ledger only ever contains this bracket when
    ``errata-cultist-power`` produced it, but the option is still checked
    here for defense (``cultist_maybe_gain_power``, commands.pm 576).
    """
    if not state.setup.options.errata_cultist_power:
        raise EngineError(
            "cultist_leech_bonus row seen without errata_cultist_power set",
            state=state,
            faction=faction,
            cmd=cmd,
        )
    gain = FACTIONS[faction].leech_effect.get("not_taken", {})
    fs = state.factions[faction]
    for key, amount in gain.items():
        if not amount:
            continue
        if key == "PW":
            fs = replace(fs, power=fs.power.gain(amount))
        elif key == "VP":
            fs = replace(fs, vp=fs.vp + amount)
        else:
            raise ValueError(f"unhandled leech_effect.not_taken key {key!r} for {faction}")
    return with_faction(state, faction, fs)


register_handler("leech", handle_leech)
register_handler("decline", handle_decline)
register_handler("cultist_leech_bonus", handle_cultist_leech_bonus)
