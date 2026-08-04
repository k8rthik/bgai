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
   ``turn_order`` (``variable_turn_order`` -> ``passed_order``; else seat
   order, unchanged), and transitions to either the next round's
   ``Phase.INCOME`` (loop back to step 4) or, after round 6,
   ``Phase.FINISHED`` with ``round`` left at 6.
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
from bgai.engine.tm.factions_data import CULTS, FACTIONS
from bgai.engine.tm.income import faction_income
from bgai.engine.tm.setup import GameSetup
from bgai.engine.tm.state import FactionState, GameState, Phase, with_faction

# --------------------------------------------------------------------------
# Income phase
# --------------------------------------------------------------------------

_INCOME_RESOURCE_ATTR: dict[str, str] = {"C": "coins", "W": "workers"}


def _apply_income_resource(fs: FactionState, resource: str, amount: int) -> FactionState:
    if not amount:
        return fs
    if resource == "PW":
        return replace(fs, power=fs.power.gain(amount))
    if resource == "P":
        return replace(fs, priests=min(fs.priests + amount, fs.priest_pool))
    if resource == "SPADE":
        return replace(fs, spades_available=fs.spades_available + amount)
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
        fs = _apply_income_resource(fs, resource, amount)
    return with_faction(state, faction, fs)


def _grant_cult_income(state: GameState, faction: str) -> GameState:
    """``cult_income_for_faction``: the current round's SCORE tile's
    ``cult_income``, scaled by ``floor(position / req)`` (Step 1 finding
    above). ``tile.cult == "CULT_P"`` (the optional ``temple-scoring-tile``
    SCORE9) is not a real cult track -- ``state.cults`` has no such key --
    and this engine has no priest-on-temple scoring mechanic elsewhere
    either, so it is a documented no-grant no-op here rather than a
    ``KeyError``.
    """
    tile = state.setup.score_tiles[state.round - 1]
    if tile.cult not in CULTS or not tile.cult_income:
        return state
    position = state.cults[faction][tile.cult]
    units = position // tile.req
    if units <= 0:
        return state
    fs = state.factions[faction]
    for resource, per_unit in tile.cult_income:
        fs = _apply_income_resource(fs, resource, per_unit * units)
    return with_faction(state, faction, fs)


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


def begin_actions(state: GameState) -> GameState:
    """``Phase.INCOME`` -> ``Phase.ACTIONS`` once every income row for the
    round has been applied (Task 12 contract, step 4)."""
    if state.phase != Phase.INCOME:
        raise ValueError(f"begin_actions called outside Phase.INCOME (got {state.phase})")
    return replace(state, phase=Phase.ACTIONS, active_index=0)


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

    if state.setup.options.variable_turn_order:
        new_turn_order = state.passed_order
    else:
        new_turn_order = state.setup.factions

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


def _advance_setup_dwellings(state: GameState) -> GameState:
    next_index = state.active_index + 1
    if next_index < len(state.turn_order):
        return replace(state, active_index=next_index)
    reverse_seats = tuple(reversed(state.setup.factions))
    return replace(state, phase=Phase.SETUP_BONUS, turn_order=reverse_seats, active_index=0)


def _advance_setup_bonus(state: GameState) -> GameState:
    next_index = state.active_index + 1
    if next_index < len(state.turn_order):
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
        new_fs = replace(fs, extra_actions=fs.extra_actions - 1)
        return with_faction(state, faction, new_fs)

    n = len(state.turn_order)
    for step in range(1, n + 1):
        idx = (state.active_index + step) % n
        if not state.factions[state.turn_order[idx]].passed:
            return replace(state, active_index=idx)
    return replace(state, phase=Phase.CLEANUP)


def advance_turn(state: GameState) -> GameState:
    """Central, phase-aware turn-advancement primitive (module docstring).
    Must be called by the row-grouping caller exactly once per completed
    ledger row belonging to the currently-active main-track faction --
    never for income/cleanup/bare-``setup`` rows, never for another
    faction's leech/decline/gain_cult answer row. A no-op during
    ``Phase.INCOME``/``Phase.CLEANUP`` (module docstring).
    """
    if state.phase == Phase.SETUP_DWELLINGS:
        return _advance_setup_dwellings(state)
    if state.phase == Phase.SETUP_BONUS:
        return _advance_setup_bonus(state)
    if state.phase == Phase.ACTIONS:
        return _advance_actions(state)
    return state
