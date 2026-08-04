"""Income phase, cleanup, turn advancement, and setup-to-round-1 flow.

This is the module that ties the whole round structure together: it is the
only module in this engine that reasons about *whose turn it is* and *what
phase we're in* as first-class concerns (every earlier task's handlers only
read/write a single faction's state for one command at a time).

--------------------------------------------------------------------------
STEP 1 (mandatory, empirical): the ``other_income``/``cult_income`` split
--------------------------------------------------------------------------

The ledger grammar (``ledger_parser.py``) recognizes three bare,
faction-carrying, argument-free verbs: ``other_income_for_faction``,
``cult_income_for_faction``, and ``all_income_for_faction`` (the last one
only under the ``merge-income-phases`` option -- 59 rows across the whole
corpus, vs. ~85k/~71k for the other two). None of the three carry any
resource/amount fields of their own -- the engine has to independently
know which of ``income.py``'s four ``faction_income`` categories
(base/buildings/bonus/favors) each row grants, and by what mechanism the
"cult" one is computed. This was settled by joining ``moves.parquet``
against ``deltas.parquet`` (which gives, per ledger row, the *resulting*
C/W/P/VP running totals and per-row deltas, plus a snapshot ``cult``
string "F/W/E/A" and ``pw`` bowl-string) across **three independent
games**:

- **Game 1**, ``4pLeague_S10_D1L1_G1`` (the same reference game the
  setup-order test below reproduces). Rows 42-45
  (``other_income_for_faction``, one per faction, seat order:
  engineers/darklings/nomads/mermaids) fire *immediately after* the
  SETUP_BONUS pass rows (37-40), i.e. at the very start of round 1's
  INCOME phase -- before any ACTIONS-phase row exists yet. Their C/W/P
  deltas are nonzero and vary per faction in a way that lines up with
  building/bonus-tile counts (e.g. darklings' row 43, C +6, W +3 --
  their board.pm buildings + BON3's ``{C: 6}`` income). Round 1's own
  SCORE tile (``score_tiles[0]``: cult WATER, req 4, income
  ``{SPADE: 1}``) plays no part here at all. Rows 94-97
  (``cult_income_for_faction``, all 4 factions) fire near the end of
  round 1's ACTIONS phase, right before round 2's
  ``other_income_for_faction`` rows (100-103) begin. Their C/W/P deltas
  are **all zero** -- consistent with round 1's SCORE tile granting
  SPADE, a resource ``deltas.parquet`` simply has no column for. The
  decisive cross-check is round 2's SCORE tile (``score_tiles[1]``: cult
  EARTH, req 1, income ``{C: 1}``) and its cleanup rows, 136-139:
  mermaids/engineers/nomads/darklings show C deltas of exactly 3/5/1/5,
  which match **exactly** ``floor(EARTH_position / 1) * 1`` read off each
  faction's post-row ``cult`` snapshot string (EARTH 3/5/1/5
  respectively) -- a bare integer-division scaling, not a flat award.
  Round 3's tile (``score_tiles[2]``: cult FIRE, req 2, income ``{W: 1}``)
  confirms it again at rows 179-182: darklings FIRE=2 -> W+1 (floor(2/2)),
  engineers FIRE=0 -> W+0, mermaids FIRE=0 -> W+0, nomads FIRE=4 -> W+2
  (floor(4/2)) -- every single delta matches, including the *zero* cases
  (the row is still emitted for a faction that earned nothing).
- **Game 2**, ``4pLeague_S3_D1L1_G1``, ``all_income_for_faction`` (under
  ``merge-income-phases``) rows 91-94: C/W/P deltas 3/4/1, 3/3/1, 7/5/1,
  3/6/1 for dwarves/giants/darklings/nomads -- ordinary building/bonus/
  favor-sized amounts, confirming a merged row is not scaled down or
  otherwise different from the sum of the two split rows.
- **Game 3**, ``4pLeague_S10_D1L1_G2``, isolates the *favor*-income
  contribution specifically (base/buildings/bonus alone were already
  covered by game 1). Darklings gain FAV9 (``income={"C": 3}``) at row
  103; by their next ``other_income_for_faction`` row (148) their board
  is D=3/TP=1/TE=1 (reconstructed from every intervening ``build``/
  ``upgrade`` row: 5 dwellings built total, 2 upgraded away to TP then
  one of those to TE) holding BON10 (``income={"PW": 3}``, taken at row
  119). Predicted C = TP's ``income["C"][1]`` (2) + FAV9's 3 = 5; W = D's
  ``income["W"][3]`` (4); P = TE's ``income["P"][1]`` (1). Row 148's
  actual deltas: C +5, W +4, P +1 -- an exact match on all three,
  confirming favor income flows through the same ``other_income_for_faction``
  bucket as base/building/bonus income (test:
  ``test_other_income_matches_a_third_independent_corpus_game`` in
  ``tests/test_round_flow.py`` reproduces this exact board/tile state).

**Finding**: ``other_income_for_faction`` = ``income.faction_income``'s
four categories (base + buildings + bonus + favors), granted once per
faction at the start of every round's INCOME phase, independent of any
scoring tile. ``cult_income_for_faction`` = the *current round's* SCORE
tile's ``cult_income`` (``tiles.ScoringTile.cult_income``), scaled by
``floor(faction's live position on that tile's .cult track / tile.req)``,
granted once per faction at CLEANUP (end of ACTIONS phase, using that
faction's cult position as it stands at that moment) -- **always** emitted,
even when the faction earns 0. ``all_income_for_faction`` (under
``merge-income-phases``) grants both components together in one row.

**On row ordering** (code review correction): a first pass at this
docstring characterized game 1's cult-income rows (94-97) as arriving "in
a different order -- not seat order" as if that were some anomaly unique
to cult income. That framing was misleading. Per the reference
implementation, *both* verbs are emitted the same way: ``command_income``
(``commands.pm`` 994-1016) iterates ``$game{acting}->factions_in_turn_order()``
for the ``'other'`` batch exactly as it does for the ``'cult'`` batch
(both call sites live in ``command_start_planning``, ``commands.pm``
1144-1154), and ``factions_in_turn_order`` (``acting.pm`` 203-210) is
itself just ``factions_in_order`` rotated to start after whichever
faction currently holds ``{start_player}`` -- which, under
``variable-turn-order`` (this reference game's own options -- see
``setup.py``'s ``GameOptions``), is exactly the *previous* round's pass
order, the same source ``end_of_round`` below feeds into next round's
``turn_order``. So there is no separate "cult order" mechanism to model;
the apparent seat-order-vs-not difference between the two batches in the
sampled rows simply reflects whatever ``factions_in_turn_order`` evaluated
to at each call site, not a distinct rule. This is why the finding above
does not encode any particular row order as load-bearing: every row names
its own faction explicitly, ``apply.py``'s gate exempts all three income
verbs, and ``handle_income_row`` reads only ``state``/``faction`` -- the
order rows arrive in is provably irrelevant to correctness here, whatever
produces it in a given replay.

A companion finding, needed for ``end_of_round``'s cult-reward plumbing:
which resource bucket does a SPADE-valued ``cult_income`` (round 1's own
tile, and ``SCORE2``/``SCORE6``/``SCORE8`` generally) land in? The corpus
gives no ledger evidence either way (``deltas.parquet`` has no SPADE
column and no companion ``+SPADE``/``dig``-style row ever follows a
``cult_income_for_faction`` row). By design, though,
``FactionState.spades_available`` is documented (``state.py``,
``actions_terraform.py``) as *the* single running spade balance that every
other spade source (``dig``, a Halflings SH grant, ACT5/ACT6/ACTG/BON1)
already feeds -- so a cult-tile SPADE reward is applied here the same way,
straight onto ``spades_available``. The brief's example scenario ("4 WATER
-> 1 SPADE", ``SCORE6``) is a direct test of this.

Also resolved without needing new plumbing: the task brief flagged
Mermaids' free stronghold shipping level (``factions_data.py``'s SH
``build_gain={"GAIN_SHIP": 1}``) as a Task-8 seam. It turns out
``actions_build.py``'s ``_apply_build_gain`` already applies it
immediately at SH-build time (bumping ``fs.shipping``, capped at
``max_level``) -- see ``actions_pass.py``'s module docstring for the full
citation. Nothing left to wire here.

--------------------------------------------------------------------------
Bonus-tile coin accumulation
--------------------------------------------------------------------------

Corpus cross-check (same reference game): a bonus tile not chosen by
anyone this round accrues +1 coin at cleanup (``GameState.bonus_coins``,
seeded to 0 for every tile at ``GameState.initial``). Row 83, nomads takes
BON7 (unclaimed since round 1) and gains exactly 1 C; row 298, nomads
takes BON4 (unclaimed for 4 rounds) gains exactly 4 C; row 300, engineers
takes BON9 (unclaimed 3 rounds) gains exactly 3 C. ``actions_pass.py``'s
``handle_pass`` pays this out and zeroes the taken tile's counter;
``end_of_round`` below increments every tile *not currently held by any
faction* by 1.

--------------------------------------------------------------------------
Turn advancement: the central design
--------------------------------------------------------------------------

``apply()`` (``apply.py``) is a per-*command* entry point -- one
``ParsedCommand`` at a time -- and ``ParsedCommand`` carries no ledger
row/seq boundary. A single ledger row (one player's whole turn) routinely
bundles several "main-track" verbs together when they're causally chained
(``dig 1. build A3. connect R1. gain_town TW1`` -- one Mermaids turn,
corpus row 208; ``action ACTW. build F5`` -- Witches' Ride is one turn,
per ``actions_power.py``'s own docstring), so "advance the turn after
every main-track verb" would over-advance. Conversely, decision/pending
*answers* from a faction other than the currently-active one (leech,
decline, the Cultists' cult-choice answer) are already gate-exempt in
``apply.py`` and must never move ``active_index`` at all, no matter how
many of them land between one active faction's turn and the next.

**Design decision**: turn advancement is *not* wired into ``apply()``
(which would require guessing at row boundaries from ``cmd.verb`` alone,
and — per the task brief's explicit constraint — must not change the
behavior of every existing test that calls a ``handle_*`` function
directly, bypassing ``apply()`` entirely). Instead this module exposes
``advance_turn(state) -> GameState`` as an explicit, phase-aware primitive
that a *row-grouping* caller (the replay harness, Task 12) invokes exactly
once per completed ledger row belonging to the currently-active main-track
faction -- never after a bare bookkeeping row (``setup``,
``*_income_for_faction``), never after another faction's leech/decline/
gain_cult answer row. See "Task 12 contract" below for the precise call
sequence this implies.

``advance_turn`` is phase-aware:

- ``Phase.SETUP_DWELLINGS``: steps through the precomputed snake order
  (``_setup_dwellings_order`` -- ``acting.pm`` setup_order: seat order
  once, reverse seat order once, then any Nomads-only third dwelling, then
  any Chaos-Magicians-only single dwelling last; ``start_setup`` installs
  this sequence as ``state.turn_order`` before any dwelling is built).
  Reaching the end transitions to ``Phase.SETUP_BONUS`` with
  ``turn_order`` replaced by *reverse* seat order (corpus rows 37-40:
  mermaids/nomads/darklings/engineers, the exact reverse of
  engineers/darklings/nomads/mermaids).
- ``Phase.SETUP_BONUS``: steps through that reverse order; reaching the
  end transitions to ``round=1, Phase.INCOME`` with ``turn_order``
  restored to real seat order.
- ``Phase.ACTIONS``: honors ``FactionState.extra_actions`` (Chaos
  Magicians' ACTC ticket -- ``state.py``'s own docstring: "spending it
  back down (one full turn per unit)" -- one unit is consumed and the
  *same* faction goes again, ``active_index`` unchanged) then, with no
  ticket left, steps ``active_index`` forward to the next faction in
  ``turn_order`` whose ``FactionState.passed`` is false. If every faction
  has passed, transitions straight to ``Phase.CLEANUP`` (no faction is
  "active" there any more -- cleanup rows are gate-exempt, see below).
- ``Phase.INCOME``/``Phase.CLEANUP``: a no-op. Income and cult-income rows
  apply to an explicit ``faction`` named by the row itself, in whatever
  order the corpus lists them -- both verbs share the same underlying
  Perl ordering mechanism and neither this engine's handler nor its
  correctness depend on row order at all (see Step 1 above, "On row
  ordering") -- ``apply.py``'s turn-order gate now exempts all three
  income verbs (plus the bookkeeping-only ``setup`` verb) for exactly
  this reason, so ``active_index``/``active_faction`` are irrelevant
  during these two phases. ``begin_actions``/``end_of_round`` below own
  their phase transitions instead.

--------------------------------------------------------------------------
Task 12 contract (the replay harness)
--------------------------------------------------------------------------

1. ``state = GameState.initial(setup)`` then immediately
   ``state = start_setup(state)`` -- installs the SETUP_DWELLINGS snake
   order as ``turn_order`` before any command is applied.
2. Apply every ``setup``/``build`` row for SETUP_DWELLINGS via ``apply()``
   in ledger order. Call ``advance_turn(state)`` after each ``build`` row
   only (never after a bare ``setup`` row -- that verb is gate-exempt and
   carries no turn-order weight).
3. Apply every ``pass`` row for SETUP_BONUS via ``apply()``; call
   ``advance_turn(state)`` after each one. The last call transitions to
   ``round=1, Phase.INCOME``.
4. Apply that round's ``other_income_for_faction``/``all_income_for_faction``
   rows via ``apply()`` (any order; gate-exempt) -- **do not** call
   ``advance_turn`` for these. Once every faction's income row for the
   round has landed, call ``begin_actions(state)`` to enter
   ``Phase.ACTIONS``.
5. Apply ACTIONS-phase rows (build/upgrade/action/advance/pass/send/dig/
   transform/bridge/connect and their leech/gain_favor/gain_town/gain_cult
   sub-decision answers) via ``apply()``. Call ``advance_turn(state)`` once
   per row that belongs to the currently-active main-track faction (i.e.
   the row that was gated by ``faction == active_faction(state)``) --
   never after a leech/decline/gain_cult answer row from a *different*
   faction. Once every faction has passed, ``advance_turn`` itself
   transitions to ``Phase.CLEANUP``.
6. Apply that round's ``cult_income_for_faction``/``all_income_for_faction``
   rows via ``apply()`` (any order; gate-exempt) -- no ``advance_turn``
   calls. Once every faction's cult-income row for the round has landed,
   call ``end_of_round(state)``: this grants bonus-tile coin
   accumulation, resets per-round balances, sets next round's
   ``turn_order`` (task-14 correction: under ``variable_turn_order``,
   this round's ``passed_order`` directly; otherwise seat order rotated
   to start at whoever passed *first* this round, which is **not** the
   same as ``passed_order`` -- that function's own docstring has the
   full citation trail and corpus counter-example), and transitions to
   either the next round's ``Phase.INCOME`` (loop back to step 4) or,
   after round 6, ``Phase.FINISHED`` with ``round`` left at 6.
7. ``Phase.FINISHED`` is a deliberate hand-off point: no final/area/
   resource-conversion scoring has been applied yet. ``round_flow.py``
   guarantees only that every faction's resources, board, and VP total
   reflect the end of round 6's cleanup. What happens next is Task 12's
   two-mode contract (``scoring.py``'s own module docstring has the full
   citation trail -- summarized here so this contract doesn't go stale):

   - **Replay** (this harness, reproducing a real ledger): the corpus's
     final-scoring block is itself made of ordinary ledger rows --
     ``score_vp`` (cult/network grants) and ``score_resources``
     (leftover-resource conversion) -- that arrive **after**
     ``end_of_round`` has already set ``Phase.FINISHED``. Keep applying
     them via ``apply()`` exactly like every other row (both verbs are
     gate-exempt in ``apply.py``, so ``active_faction``/``turn_order`` are
     irrelevant here too, same as income/cleanup); ``apply()`` dispatches
     them to ``scoring.handle_score_vp``/``scoring.handle_score_resources``,
     which validate each grant against the engine's own recomputation
     before applying it. So, contrary to an earlier draft of this
     contract, ``apply()`` calls are still expected once
     ``Phase.FINISHED`` -- just no *other* state-machine calls from this
     module (``advance_turn``/``begin_actions``/``end_of_round``): the
     round/turn machinery's job is done, only the scoring verbs remain.
   - **Simulation** (an agent playing out a game with no ledger to
     replay against): call ``scoring.final_scoring(state)`` once instead
     -- a pure computation that derives and applies the same cult/network/
     resource grants in one shot, with no ``apply()``/ledger rows
     involved at all.

   These two modes are mutually exclusive on a given state (replaying the
   rows and then also calling ``final_scoring``, or vice versa, would
   double-apply every VP/resource delta) -- see ``scoring.py``'s module
   docstring for the full contract and the double-apply hazard.
"""

from __future__ import annotations

from dataclasses import replace

from bgai.data.ledger_parser import ParsedCommand
from bgai.engine.tm.apply import EngineError, register_handler
from bgai.engine.tm.factions.hooks import hooks_for
from bgai.engine.tm.factions_data import CULTS, FACTIONS
from bgai.engine.tm.income import faction_income
from bgai.engine.tm.setup import GameSetup
from bgai.engine.tm.state import FactionState, GameState, Phase, with_faction

# --------------------------------------------------------------------------
# Income phase
# --------------------------------------------------------------------------

_INCOME_RESOURCE_ATTR: dict[str, str] = {"C": "coins", "W": "workers"}


def _apply_spade_income_bonus(state: GameState, faction: str, fs: FactionState, spades: int) -> FactionState:
    """``actions_build.py``'s/``actions_terraform.py``'s/
    ``actions_power.py``'s identically-named-in-spirit helper
    (``_apply_spade_gain_bonus``/``_apply_extra_dig_gain``), a fifth
    private copy for the same no-cross-coupling rationale those modules
    already document: ``hooks_for(faction).extra_dig_gain`` (Halflings'
    unconditional +1 VP/spade, Alchemists' SH-gated +2 PW/spade) fires
    for *any* positive SPADE delta, cult-income included, not just
    ``dig``'s (``resources.pm`` 313-392's generic per-unit gain loop --
    those other modules' own docstrings cite this). Missing it here was a
    real engine bug: reference-corpus game ``4pLeague_S10_D1L1_G2`` row
    142, halflings' round-2 cult income grants 1 spade (AIR position 4,
    req 4) and should score the unconditional +1 VP alongside it -- task-13
    report.
    """
    if spades <= 0:
        return fs
    extra = hooks_for(faction).extra_dig_gain(state, faction)
    for res, per_unit in extra.items():
        total = per_unit * spades
        if not total:
            continue
        if res == "VP":
            fs = replace(fs, vp=fs.vp + total)
        elif res == "PW":
            fs = replace(fs, power=fs.power.gain(total))
        else:
            raise ValueError(f"unhandled extra_dig_gain key {res!r} for {faction}")
    return fs


def _apply_income_resource(
    state: GameState, faction: str, fs: FactionState, resource: str, amount: int
) -> FactionState:
    if not amount:
        return fs
    if resource == "PW":
        return replace(fs, power=fs.power.gain(amount))
    if resource == "P":
        return replace(fs, priests=min(fs.priests + amount, fs.priest_pool))
    if resource == "SPADE":
        fs = replace(fs, spades_available=fs.spades_available + amount)
        return _apply_spade_income_bonus(state, faction, fs, amount)
    attr = _INCOME_RESOURCE_ATTR.get(resource)
    if attr is None:
        raise ValueError(f"unhandled income resource {resource!r}")
    return replace(fs, **{attr: getattr(fs, attr) + amount})


def _grant_other_income(state: GameState, faction: str) -> GameState:
    """``other_income_for_faction``: ``income.faction_income``'s four
    categories, summed and applied (Step 1 finding above)."""
    fs = state.factions[faction]
    categories = faction_income(fs, FACTIONS[faction])
    totals: dict[str, int] = {}
    for bucket in categories.values():
        for resource, amount in bucket.items():
            totals[resource] = totals.get(resource, 0) + amount
    for resource, amount in totals.items():
        fs = _apply_income_resource(state, faction, fs, resource, amount)
    return with_faction(state, faction, fs)


def _cult_p_position(state: GameState, faction: str) -> int:
    """``$faction->{CULT_P}`` (``commands.pm`` ``command_send`` line 334):
    a running count, incremented once every time one of ``faction``'s
    ``send`` commands lands on a previously-*empty* priest slot on any
    cult track (never decremented -- landing on an already-occupied step
    just advances the track, no ``CULT_P`` bump, per that same ``command_
    send`` branch). This engine has no separate running counter for it,
    but ``state.priest_slots`` (``apply.py``'s ``handle_send``) already
    records, per track, *which* faction occupies each priest-only step --
    the live count is exactly how many of those slots, across all 4
    tracks, are currently this faction's (task-14 fix, corpus
    ``4pLeague_S12_D1L1_G1`` rows 150/151: engineers/alchemists each have
    exactly 1 occupied slot and are owed the ``temple-scoring-tile``'s
    2 C/unit; nomads/witches have 0 and correctly get nothing).
    """
    return sum(1 for cult in CULTS for occupant in state.priest_slots[cult] if occupant == faction)


def _grant_cult_income(state: GameState, faction: str) -> GameState:
    """``cult_income_for_faction``: the current round's SCORE tile's
    ``cult_income``, scaled by ``floor(position / req)`` (Step 1 finding
    above). ``tile.cult == "CULT_P"`` (the optional ``temple-scoring-tile``
    SCORE9) is not a real cult track -- ``state.cults`` has no such key --
    its "position" is ``_cult_p_position`` instead (task-14 fix; an earlier
    revision treated this as a documented no-grant no-op, which the
    corpus disproves).
    """
    tile = state.setup.score_tiles[state.round - 1]
    if not tile.cult_income:
        return state
    if tile.cult == "CULT_P":
        position = _cult_p_position(state, faction)
    elif tile.cult in CULTS:
        position = state.cults[faction][tile.cult]
    else:
        return state
    units = position // tile.req
    if units <= 0:
        return state
    fs = state.factions[faction]
    for resource, per_unit in tile.cult_income:
        fs = _apply_income_resource(state, faction, fs, resource, per_unit * units)
    return with_faction(state, faction, fs)


def grant_missing_cult_income(state: GameState, faction: str) -> GameState:
    """Public wrapper around :func:`_grant_cult_income`, for a caller
    outside this module (``replay.py``'s ``_grant_missing_cult_income_
    retroactively``) that needs to apply a faction's cult-income grant
    without a ``cult_income_for_faction`` ledger row to drive it through
    ``handle_income_row``'s normal dispatch -- a handful of corpus games
    never emit one for a specific faction/round at all (this module's
    own Step-1 finding docstring already established this as a genuine,
    accepted ledger gap, not a parsing bug), and the grant still needs to
    land once the batch's ``end_of_round`` fires regardless. Identical to
    calling ``handle_income_row`` with a synthetic ``cult_income_for_
    faction`` command; exists as its own name so callers don't need to
    construct a throwaway ``ParsedCommand`` just to invoke it.
    """
    return _grant_cult_income(state, faction)


def handle_income_row(state: GameState, faction: str, cmd: ParsedCommand) -> GameState:
    """``other_income_for_faction``/``cult_income_for_faction``/
    ``all_income_for_faction`` (module docstring, Step 1)."""
    if cmd.verb == "other_income_for_faction":
        return _grant_other_income(state, faction)
    if cmd.verb == "cult_income_for_faction":
        return _grant_cult_income(state, faction)
    if cmd.verb == "all_income_for_faction":
        return _grant_cult_income(_grant_other_income(state, faction), faction)
    raise EngineError(f"unknown income verb {cmd.verb!r}", state=state, faction=faction, cmd=cmd)


register_handler("other_income_for_faction", handle_income_row)
register_handler("cult_income_for_faction", handle_income_row)
register_handler("all_income_for_faction", handle_income_row)


def _start_full_move_reset(fs: FactionState) -> FactionState:
    """``acting.pm``'s ``start_full_move`` (~227-238): every time a faction
    starts a brand-new full action, Perl resets several transient markers
    -- ``delete $faction->{TELEPORT_TO}`` and ``delete $faction->{cult_
    blocked}`` among them (this engine's ``teleported_hex``/
    ``cult_blocked``; ``allowed_sub_actions``/``allowed_build_locations``/
    ``require_home_terrain_tf``/``disable_spade_decline`` have no engine
    equivalent to reset). A faction capped at 9 on some cult track for
    lack of a key does **not** carry that memory into a later turn -- a
    key gained in a *different*, later turn has nothing queued to retry
    (task-14 fix; corpus evidence: ``4pLeague_S10_D3L1_G4`` row 329 blocks
    chaosmagicians' FIRE at 9, and gaining a key from an unrelated
    ``gain_town`` 7 rows/one full turn-cycle later at row 336 does *not*
    retroactively bump it to 10 -- contrast ``_retry_blocked_cults``'s own
    *same-row* reference case, ``4pLeague_S10_D1L1_G6`` row 315, where the
    block and the retrying key land in the same turn). No-op (returns
    ``fs`` unchanged) when neither field needs clearing, so callers can
    use this unconditionally without manufacturing spurious no-op
    ``replace`` calls.
    """
    if fs.teleported_hex is None and not fs.cult_blocked:
        return fs
    return replace(fs, teleported_hex=None, cult_blocked=frozenset())


def _first_eligible_index(state: GameState, turn_order: tuple[str, ...], start: int = 0) -> int:
    """First index ``>= start`` in ``turn_order`` whose faction is neither
    ``passed`` nor ``FactionState.dropped`` -- falls back to ``start`` if
    every remaining entry is ineligible (defensive; every real game keeps
    at least one live faction through round 6). Used wherever
    ``active_index`` is (re)seeded to a fresh value rather than stepped
    forward one faction at a time from an already-valid position
    (``_advance_actions`` below already skips dropped/passed factions when
    stepping, but a freshly *seeded* ``active_index`` -- ``begin_actions``'
    round-start ``turn_order[0]`` -- was never validated at all before this
    task-14 fix: a faction dropped mid-``SETUP_BONUS`` could still land as
    round 1's ``active_faction`` simply for sitting at seat index 0,
    corpus ``4pLeague_S21_D3L1_G4`` row 47).
    """
    for idx in range(start, len(turn_order)):
        fs = state.factions[turn_order[idx]]
        if not fs.passed and not fs.dropped:
            return idx
    return start


def begin_actions(state: GameState) -> GameState:
    """``Phase.INCOME`` -> ``Phase.ACTIONS`` once every income row for the
    round has been applied (Task 12 contract, step 4). Resets every
    faction via ``_start_full_move_reset`` -- ``_advance_actions`` below
    does the same for whichever faction becomes newly active *during* the
    round, but the round's very first active faction (``turn_order[0]``)
    never goes through that path, so this is the other of the two reset
    points Perl's own ``start_full_move`` collapses into one call.
    """
    if state.phase != Phase.INCOME:
        raise ValueError(f"begin_actions called outside Phase.INCOME (got {state.phase})")
    new_factions = {name: _start_full_move_reset(fs) for name, fs in state.factions.items()}
    active_index = _first_eligible_index(state, state.turn_order)
    return replace(state, phase=Phase.ACTIONS, active_index=active_index, factions=new_factions)


# --------------------------------------------------------------------------
# Cleanup / end of round
# --------------------------------------------------------------------------


def _bumped_bonus_coins(state: GameState) -> dict[str, int]:
    """+1 coin on every bonus tile no faction currently holds
    (``commands.pm`` ``command_start``, lines 899-904: ``for (keys
    %{$game{pool}}) { next if !/^BON/; next if !$game{pool}{$_};
    $game{bonus_coins}{$_}{C}++; }``). ``command_start`` fires on *every*
    round transition, including the very first (``$game{round}++`` from 0
    to 1) -- this is why ``_advance_setup_bonus`` below calls this helper
    too, not just ``end_of_round`` (module docstring's own "Bonus-tile
    coin accumulation" section already cites this exact mechanic; task-13
    report, reference-game row 83 -- nomads takes BON7 for 1 C at row 83,
    still round 1's own ACTIONS phase, which only checks out if BON7
    already carried 1 accumulated coin from the SETUP_BONUS -> round-1
    transition, since round 1's own cleanup hasn't happened yet by then).
    """
    held_bonus = {fs.bonus for fs in state.factions.values() if fs.bonus is not None}
    new_bonus_coins = dict(state.bonus_coins)
    for tile in state.setup.bonus_tiles:
        if tile not in held_bonus:
            new_bonus_coins[tile] = new_bonus_coins.get(tile, 0) + 1
    return new_bonus_coins


def end_of_round(state: GameState) -> GameState:
    """``Phase.CLEANUP`` -> next round's ``Phase.INCOME`` (or
    ``Phase.FINISHED`` after round 6) -- module docstring, Task 12
    contract step 6: bonus-tile coin accumulation, per-round balance
    resets, and next round's turn order.

    ``spades_available`` is deliberately **not** reset here (task-13 fix:
    an earlier revision zeroed it alongside ``actions_used``/
    ``extra_actions``/``passed``, with no Perl citation backing that
    reset -- ``commands.pm`` never touches ``$faction->{SPADE}`` at
    ``command_start``, only ever inside ``command_dig``/``command_
    transform``/the ``-SPADE`` branch). Reference-game row 96-98 proves
    it must survive: round 1's own WATER->SPADE cult-income tile grants
    darklings 1 spade at cleanup (row 96, still round 1's tile), and row
    98 -- *after* this function has already run (``round`` is 2, still
    ``Phase.INCOME``) -- spends that exact spade on a ``transform``. A
    reset here would zero it out before darklings ever gets to spend it.
    """
    if state.phase != Phase.CLEANUP:
        raise ValueError(f"end_of_round called outside Phase.CLEANUP (got {state.phase})")

    new_bonus_coins = _bumped_bonus_coins(state)

    new_factions = {
        name: replace(fs, actions_used=frozenset(), extra_actions=0, passed=False)
        for name, fs in state.factions.items()
    }

    # Task-14 correction: next round's turn_order always starts with
    # whichever faction passed *first* this round (acting.pm's
    # factions_in_turn_order, 203-210: rotates raw_factions_in_order so
    # whichever faction holds {start_player} -- set on any first-to-pass,
    # command_pass ~780-785, unconditionally -- goes first). variable-
    # turn-order only gates whether every pass *also* reorders
    # raw_factions_in_order itself (moving the passer to the array's end,
    # commands.pm ~788-792); applied across a round's 4 passes this
    # converges to plain chronological passed_order. Without the option,
    # raw_factions_in_order is never touched by passing, so only *which*
    # seat starts is chronological -- the other 3 keep their original
    # seat *positions*, not their pass *timestamps*, which can differ
    # whenever a faction passes early relative to its seat position.
    # Corpus proof (an earlier revision used unrotated seat order here,
    # never checked against a real non-variable-turn-order game -- also
    # wrong, differently): ``4pLeague_S1_D1L1_G1``, round 2's pass order
    # is (engineers, cultists, darklings, witches), but round 3 actually
    # starts (engineers, cultists, witches, darklings) -- seat order
    # (darklings, engineers, cultists, witches) rotated to the first
    # passer, not passed_order itself. ``state.passed_order`` is never
    # empty here (every faction must pass before ``end_of_round`` runs),
    # so ``[0]`` is safe.
    if state.setup.options.variable_turn_order:
        new_turn_order = state.passed_order
    else:
        first_to_pass = state.passed_order[0]
        seats = state.setup.factions
        pivot = seats.index(first_to_pass)
        new_turn_order = seats[pivot:] + seats[:pivot]

    is_final_round = state.round >= 6
    new_round = state.round if is_final_round else state.round + 1
    new_phase = Phase.FINISHED if is_final_round else Phase.INCOME

    return replace(
        state,
        factions=new_factions,
        power_actions_taken=frozenset(),
        bonus_coins=new_bonus_coins,
        turn_order=new_turn_order,
        passed_order=(),
        active_index=0,
        round=new_round,
        phase=new_phase,
    )


# --------------------------------------------------------------------------
# Turn advancement
# --------------------------------------------------------------------------


def _setup_dwellings_order(setup: GameSetup) -> tuple[str, ...]:
    """``acting.pm`` setup_order (module docstring): seat order once,
    reverse seat order once (both skipping any single-dwelling faction --
    Chaos Magicians), then any 3-dwelling faction's extra turn (Nomads),
    then any single-dwelling faction's turn, last.
    """
    seats = setup.factions
    single_d = [f for f in seats if FACTIONS[f].start_dwellings == 1]
    triple_d = [f for f in seats if FACTIONS[f].start_dwellings == 3]
    forward = [f for f in seats if FACTIONS[f].start_dwellings != 1]
    reverse = list(reversed(forward))
    return tuple(forward + reverse + triple_d + single_d)


def start_setup(state: GameState) -> GameState:
    """Install the SETUP_DWELLINGS snake order as ``turn_order`` -- call
    once, immediately after ``GameState.initial(setup)``, before applying
    any command (Task 12 contract, step 1).
    """
    return replace(state, turn_order=_setup_dwellings_order(state.setup), active_index=0)


def _first_live_setup_index(state: GameState, turn_order: tuple[str, ...], start: int) -> int | None:
    """First index ``>= start`` in ``turn_order`` whose faction is not
    (yet) ``FactionState.dropped`` -- ``None`` if every remaining entry
    belongs to a dropped faction. ``commands.pm``'s ``drop-faction``
    handler removes *every remaining* ``setup_order`` entry for the
    dropped faction outright the moment the drop happens (~1597-1599:
    ``$game{acting}->setup_order([grep {$_->[0] ne $f} @{...setup_order()}])``),
    not just the very next one -- a faction that drops before its own
    setup turn ever arrives loses *both* its dwelling-snake slots
    (forward and reverse) and its bonus-tile pick outright, in one shot.
    Corpus: ``4pLeague_S45_D3L4_G1``, darklings drops at row 30, exactly
    when its own forward-order dwelling turn would start (row 29 is
    cultists' forward pick) -- row 31 (engineers) is the very next row,
    and darklings never gets *either* dwelling pick, confirmed by the
    reverse-order pass a few rows later skipping straight from
    swarmlings to swarmlings again (its own forward+reverse back to
    back) with no darklings row between.
    """
    for idx in range(start, len(turn_order)):
        if not state.factions[turn_order[idx]].dropped:
            return idx
    return None


def _advance_setup_dwellings(state: GameState) -> GameState:
    next_index = _first_live_setup_index(state, state.turn_order, state.active_index + 1)
    if next_index is not None:
        return replace(state, active_index=next_index)
    reverse_seats = tuple(reversed(state.setup.factions))
    # A faction already dropped by the time SETUP_BONUS starts (this
    # corpus has no example past round 0's own dwelling snake, but
    # ``_first_live_setup_index``'s own docstring citation applies here
    # identically) never gets a bonus-tile pick either -- same
    # ``_first_live_setup_index`` skip, starting from index 0.
    bonus_index = _first_live_setup_index(state, reverse_seats, 0)
    return replace(
        state, phase=Phase.SETUP_BONUS, turn_order=reverse_seats, active_index=bonus_index or 0
    )


def _advance_setup_bonus(state: GameState) -> GameState:
    next_index = _first_live_setup_index(state, state.turn_order, state.active_index + 1)
    if next_index is not None:
        return replace(state, active_index=next_index)
    # command_start's bonus-coin bump fires here too (_bumped_bonus_coins'
    # own docstring) -- this is round 1's "$game{round}++", from 0 to 1.
    return replace(
        state,
        phase=Phase.INCOME,
        round=1,
        turn_order=state.setup.factions,
        active_index=0,
        bonus_coins=_bumped_bonus_coins(state),
    )


def _advance_actions(state: GameState) -> GameState:
    faction = state.turn_order[state.active_index]
    fs = state.factions[faction]
    if fs.extra_actions > 0:
        # A fresh full action for the *same* faction (ACTC ticket) is
        # still a new ``start_full_move`` in Perl terms -- reset here too
        # (``_start_full_move_reset`` docstring, task-14 fix).
        new_fs = replace(_start_full_move_reset(fs), extra_actions=fs.extra_actions - 1)
        return with_faction(state, faction, new_fs)

    n = len(state.turn_order)
    for step in range(1, n + 1):
        idx = (state.active_index + step) % n
        next_faction = state.turn_order[idx]
        next_fs = state.factions[next_faction]
        # A dropped faction (``FactionState.dropped`` docstring) is
        # permanently excluded from turn order, exactly like
        # ``acting.pm``'s own ``!$_->{dropped}`` filter -- checked
        # alongside ``passed`` (which resets every round; ``dropped``
        # never does) rather than folded into it, so the two stay
        # independently readable at every call site.
        if not next_fs.passed and not next_fs.dropped:
            state = with_faction(state, next_faction, _start_full_move_reset(next_fs))
            return replace(state, active_index=idx)
    return replace(state, phase=Phase.CLEANUP)


def advance_turn(state: GameState) -> GameState:
    """Central, phase-aware turn-advancement primitive (module docstring).
    Must be called by the row-grouping caller once per completed *full
    action* belonging to the currently-active main-track faction -- see
    ``is_turn_boundary`` below for when a ledger row bundles more than one
    of those. Never call for income/cleanup/bare-``setup`` rows, never for
    another faction's leech/decline/gain_cult answer row. A no-op during
    ``Phase.INCOME``/``Phase.CLEANUP`` (module docstring).
    """
    if state.phase == Phase.SETUP_DWELLINGS:
        return _advance_setup_dwellings(state)
    if state.phase == Phase.SETUP_BONUS:
        return _advance_setup_bonus(state)
    if state.phase == Phase.ACTIONS:
        return _advance_actions(state)
    return state


# --------------------------------------------------------------------------
# Row-to-turn-boundary disambiguation (ACTC compound-turn bundling)
# --------------------------------------------------------------------------

_ALWAYS_FRESH_ACTION_VERBS = frozenset({"action", "pass", "advance", "send"})
# ``connect`` never costs its own action -- it is always an automatic
# consequence of whatever full action just formed or completed the
# qualifying river-adjacent cluster (a build/upgrade directly, corpus row
# 208; or a milestone favor-tile grant from an unrelated action rescanning
# per FAV5's "Hack" immediate-rescan rule, task-13 fix #13 -- corpus row
# 380, ``send FIRE. connect R20. gain_town TW8``, where ``send``ing a
# priest crosses a cult milestone that grants FAV5). So, like ``transform``,
# it never itself starts a fresh action, regardless of what precedes it or
# how many ``connect`` commands land in a row (corpus pattern
# ``connect r1. connect r20.``, two simultaneous qualifying clusters).
_NEVER_FRESH_ACTION_VERBS = frozenset({"transform", "connect"})
# "build" checks both "free_d" (ACTW's genuine free building) and "free_tf"
# (ACTN's free-transform marker -- ``actions_build.py``'s ``handle_build``
# consumes it internally for its own implicit transform-if-needed dispatch,
# task-13 fix #9, with no separate ``transform`` verb ever appearing in the
# ledger for that case: corpus pattern ``action ACTN; build ...``).
_MARKER_KINDS_FOR_VERB: dict[str, tuple[str, ...]] = {
    "build": ("free_d", "free_tf"),
    "upgrade": ("free_tp",),
    "bridge": ("bridge",),
}
# A ``build``/``upgrade`` immediately after a ``connect`` (corpus:
# ``connect R20. gain_town TW1. ... upgrade B3 to TP``, one Mermaids turn
# -- the ledger doesn't always order the pair build/upgrade-then-connect)
# is still a continuation of the same compound turn ``connect`` itself was
# part of, even though ``connect`` never opens/closes a boundary of its
# own (it's in ``_NEVER_FRESH_ACTION_VERBS`` above, not tracked here).
# ``build``/``upgrade`` do *not* get each other added here: a bare second
# ``build`` after ACTC (corpus pattern ``action ACTC; build; build``) must
# stay fresh.
_CONTINUATION_PREV_VERBS: dict[str, frozenset[str]] = {
    "build": frozenset({"transform", "dig", "connect"}),
    "upgrade": frozenset({"transform", "dig", "connect"}),
    "dig": frozenset(),
    "bridge": frozenset({"transform", "dig"}),
}


def never_starts_action(verb: str) -> bool:
    """Whether *verb* can never, by itself, open a fresh full action -- it
    always waits to be folded into whichever full action precedes or (if
    none is open yet) follows it in the same row. ``replay.py``'s
    ``_apply_row_commands`` uses this to avoid eagerly treating a row-
    leading ``transform``/``connect`` as the start of a chargeable segment
    it would then (wrongly) close before the row's real fresh command (e.g.
    a deferred ``pass``) ever lands -- see ``is_turn_boundary``'s docstring
    for the corpus evidence (``connect R10. gain_town TW8. pass``, one
    Mermaids turn where ``pass``, not ``connect``, is the actual action).
    """
    return verb in _NEVER_FRESH_ACTION_VERBS


def is_turn_boundary(
    cmd: ParsedCommand, state_before: GameState, faction: str, prev_verb: str | None = None
) -> bool:
    """Whether *this* main-track command, about to be applied on top of
    ``state_before``, **starts** a genuinely independent full action (as
    opposed to continuing the compound full action already in progress).
    The row-grouping caller (Task 12 contract) owes an ``advance_turn``
    call once a run of such continuations ends -- i.e. right *before*
    applying the next command classified as fresh, or at the end of the
    row (``replay.py``'s ``_apply_row_commands`` -- this function only
    classifies, it never decides *when* to call ``advance_turn``, since
    that requires closing out a full action only once its own
    continuations have also landed).

    Ordinarily a whole ledger row is exactly one full action (module
    docstring's "every real player turn contains exactly one main-track
    verb" finding, step 5) -- but Chaos Magicians' ACTC (``gain =>
    {GAIN_ACTION => 2}``, ``resources.pm`` ~289, wired onto
    ``FactionState.extra_actions`` by ``actions_power.py``) can grant 2
    *extra* full actions that get submitted -- and therefore ledgered -- as
    part of the *same* row as the ACTC action itself whenever no other
    faction's move interleaves. Corpus example: row 308,
    ``action ACTC; action BON2; gain_cult n1=1; pass BON3`` is 3 Perl-level
    full actions (ACTC itself, BON2, the final pass) bundled into one
    ledger row -- needing 3 ``advance_turn`` calls, not 1, each fired only
    once its own action's commands are done. Before this predicate
    existed, the replay harness called ``advance_turn`` once per row
    regardless, which under-advanced ``active_index``/``extra_actions`` on
    every such row and produced the corpus's single most common hard error
    ("faction acted out of turn (active is 'chaosmagicians')") -- 12+ of
    the first 100 corpus games.

    Perl's own discriminator is ``require_subaction($faction, $type,
    ...)`` (``acting.pm`` ~250): a command spends the faction's
    ``allowed_actions`` counter (this engine's ``extra_actions``/the base
    1-per-turn budget) unless it is answering a queued
    ``allowed_sub_actions`` grant instead -- one-shot permission bought by
    an *earlier* command in the same compound turn (a spade grant from
    ``dig``/ACT5/ACT6/BON1, or a marker from ACT1/ACTE/ACTW/ACTS/ACTN).
    This engine doesn't model ``allowed_sub_actions`` as its own structure,
    but the concrete things it grants -- ``FactionState.spades_available``,
    a ``PendingDecision`` of kind ``free_d``/``free_tp``/``bridge``, and
    (for a spade balance a ``transform``/``dig`` has *just* zeroed out by
    spending it, rather than one still sitting unspent) simply having
    ``transform``/``dig`` be the immediately preceding main-track verb --
    are exactly the state this predicate inspects:

    - ``transform`` never starts a fresh action (real TM: never a complete
      turn by itself, always in service of a build funded by the dig/
      action that granted its spades).
    - ``action``/``pass``/``advance``/``send`` always start a fresh action
      -- no marker or spade balance ever substitutes for one of these
      (checked against every sampled ACTC-bundling pattern in the corpus,
      e.g. ``action; send; send`` and ``action; action; gain_cult; pass``,
      each needing one call per verb).
    - ``build``/``upgrade``/``dig``/``connect``/``bridge`` continue the
      action in progress (not fresh) when ``prev_verb`` (the immediately
      preceding main-track verb applied in this row, ignoring non-main-
      track commands like ``gain_cult``/``burn`` in between) is
      ``transform`` or ``dig`` -- the "just spent the spade that funded
      this build" case a bare ``state_before.spades_available`` check
      alone can't see, since the spending transform/dig already zeroed the
      balance -- **or** when ``state_before`` still shows an unspent
      ``spades_available`` balance (an ACT5/ACT6/BON1 direct grant not yet
      (fully) consumed) or a matching one-shot marker pending for this
      faction. Otherwise they start a fresh action (corpus pattern
      ``action ACTC; build; build``: two independent builds, no dig/
      spade-granting action between them).
    """
    verb = cmd.verb
    if verb in _NEVER_FRESH_ACTION_VERBS:
        return False
    if verb in _ALWAYS_FRESH_ACTION_VERBS:
        return True
    if prev_verb in _CONTINUATION_PREV_VERBS.get(verb, frozenset()):
        return False
    fs = state_before.factions.get(faction)
    if fs is not None and fs.spades_available > 0:
        return False
    kinds = _MARKER_KINDS_FOR_VERB.get(verb, ())
    return not any(p.faction == faction and p.kind in kinds for p in state_before.pending)
