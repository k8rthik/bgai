"""The engine's single entry point: `apply(state, faction, cmd) -> GameState`.

Dispatches on `cmd.verb` to handler functions registered in the frozen
`HANDLERS` table (see `register_handler`). This task implements the "simple"
verbs in full; later tasks register more handlers (build/upgrade/transform/
leech/...) into the same table without touching the dispatch mechanism.

Perl references (jsnell/terra-mystica `commands.pm`/`cults.pm`) are cited
per handler docstring below (`handle_convert`, `handle_send`).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from bgai.data.ledger_parser import ParsedCommand
from bgai.engine.tm.cults import PRIEST_SLOT_STEPS, CultAdvance, advance
from bgai.engine.tm.factions_data import BASE_EXCHANGE_RATES, FACTIONS
from bgai.engine.tm.state import (
    FactionState,
    GameState,
    PendingDecision,
    Phase,
    active_faction,
    with_faction,
)


class EngineError(Exception):
    """Raised when `apply` can't process a command; carries game context
    (round/phase/faction/raw command) so callers needn't re-derive it.
    """

    def __init__(self, message: str, *, state: GameState, faction: str, cmd: ParsedCommand) -> None:
        context = (
            f"{message} [round={state.round} phase={state.phase.name} "
            f"faction={faction} cmd={cmd.raw!r}]"
        )
        super().__init__(context)
        self.game_round = state.round
        self.phase = state.phase
        self.faction = faction
        self.cmd = cmd


Handler = Callable[[GameState, str, ParsedCommand], GameState]

HANDLERS: dict[str, Handler] = {}


def register_handler(verb: str, handler: Handler) -> None:
    """Register (or replace) the handler for `verb` (direct `HANDLERS[verb]
    = handler` mutation works too; this is just the documented spelling).
    """
    HANDLERS[verb] = handler


# Verbs that may be applied even when `faction` is not `active_faction(state)`.
# `setup` (SETUP_DWELLINGS's per-faction acknowledgment rows, seat order,
# ahead of the snake-order dwelling builds) and the three income-phase
# verbs (Task 11's `round_flow.py`: each row's own `faction` field is the
# grantee, independent of turn_order/active_index -- see that module's
# docstring for the empirical evidence that cleanup's cult-income rows in
# particular arrive in a different order each round) are bookkeeping
# anchors, not turns, so they never need the active_faction() gate. `score_vp`
# and `score_resources` (Task 12's `scoring.py`) join them for the same
# reason: round 6's final-scoring rows name their own faction and arrive
# grouped by cult/network ranking (nobody scores on a level they don't
# hold), not in `turn_order` rotation -- see `scoring.py`'s module
# docstring citation of the corpus row order.
_ORDER_EXEMPT_VERBS = frozenset(
    {
        "wait",
        "annotation",
        "setup",
        "other_income_for_faction",
        "cult_income_for_faction",
        "all_income_for_faction",
        "score_vp",
        "score_resources",
    }
)
_LEECH_ANSWER_VERBS = frozenset({"leech", "decline"})
# `+CULT` answering an outstanding `cult_choice` pending (Cultists'
# leech_effect "taken" cult step, pushed mid-batch by leech.py -- see that
# module's docstring): the Cultists' answer can legitimately sit behind
# still-outstanding sibling leech offers from the same build in the pending
# queue, so this needs the same anywhere-in-queue exemption as leech/decline,
# not just an active_faction()-is-literally-me check.
_CULT_CHOICE_ANSWER_VERBS = frozenset({"gain_cult"})


def _has_queued_pending_of_kind(state: GameState, faction: str, kind: str) -> bool:
    """Any queued pending of `kind` for `faction`, anywhere in the queue.
    Strict `acting.pm` semantics (only the faction's own head entry) is a
    later task's; this only asks "is there one at all".
    """
    return any(p.faction == faction and p.kind == kind for p in state.pending)


def apply(state: GameState, faction: str, cmd: ParsedCommand) -> GameState:
    """The engine's single entry point: dispatch `cmd` for `faction`."""
    exempt = (
        cmd.verb in _ORDER_EXEMPT_VERBS
        # Phase.INCOME/Phase.CLEANUP have no "active" turn-order faction at
        # all (round_flow.py's own docstring) -- every row during these two
        # phases names its own grantee/actor directly, regardless of verb.
        # This was previously only granted to the three income verbs
        # themselves, but the same reasoning covers a genuine main-track
        # verb here too: a cult-income grant that includes SPADE (round 1's
        # own WATER->SPADE tile, e.g.) forces that faction to immediately
        # `transform` before the "other income" batch runs for anyone
        # (Perl's `command_start_planning`, commands.pm 1124-1160: cult
        # income is granted per-faction, and a resulting live SPADE pauses
        # that faction right there -- `$faction->{SPADE}` gates the next
        # step -- before `command_start`'s round++ and the "other" batch
        # proceed). Reference-game row 98 is exactly this: darklings
        # `transform H7 to black`, round=2 already but still Phase.INCOME,
        # sandwiched between round 1's cult_income rows (94-97) and round
        # 2's other_income rows (100-103) -- task-13 report.
        or state.phase in (Phase.INCOME, Phase.CLEANUP)
        or (
            cmd.verb in _LEECH_ANSWER_VERBS and _has_queued_pending_of_kind(state, faction, "leech")
        )
        or (
            cmd.verb in _CULT_CHOICE_ANSWER_VERBS
            and _has_queued_pending_of_kind(state, faction, "cult_choice")
        )
    )
    if not exempt and faction != active_faction(state):
        raise EngineError(
            f"faction acted out of turn (active is {active_faction(state)!r})",
            state=state,
            faction=faction,
            cmd=cmd,
        )

    handler = HANDLERS.get(cmd.verb)
    if handler is None:
        raise EngineError(
            f"no handler registered for verb {cmd.verb!r}", state=state, faction=faction, cmd=cmd
        )
    return handler(state, faction, cmd)


def push_pending(state: GameState, *decisions: PendingDecision) -> GameState:
    """FIFO-append `decisions` to the back of the pending queue."""
    return replace(state, pending=state.pending + tuple(decisions))


def pop_pending(state: GameState, index: int = 0) -> GameState:
    """Remove and drop the pending decision at `index` (default: the head)."""
    pending = list(state.pending)
    pending.pop(index)
    return replace(state, pending=tuple(pending))


# --------------------------------------------------------------------------
# Resource bookkeeping shared by several handlers.
# --------------------------------------------------------------------------

_RESOURCE_ATTR: dict[str, str] = {"C": "coins", "W": "workers", "P": "priests", "VP": "vp"}


def _get_resource(fs: FactionState, res: str) -> int:
    """Current amount of `res` held by `fs` (PW = usable/bowl3 power)."""
    if res == "PW":
        return fs.power.usable
    return getattr(fs, _RESOURCE_ATTR[res])


def _with_resource_delta(
    state: GameState, faction: str, fs: FactionState, res: str, delta: int, cmd: ParsedCommand
) -> FactionState:
    """Apply `delta` units of `res` to `fs`; raises `EngineError` (not a raw
    `ValueError`) if that would take a resource negative.
    """
    if res == "PW":
        try:
            power = fs.power.spend(-delta) if delta < 0 else fs.power.gain(delta)
        except ValueError as exc:
            # Power.spend raises ValueError on insufficient bowl3; Power.gain
            # only rejects n < 0, which callers here never pass.
            raise EngineError(str(exc), state=state, faction=faction, cmd=cmd) from exc
        return replace(fs, power=power)

    current = _get_resource(fs, res)
    new_value = current + delta
    if new_value < 0:
        raise EngineError(
            f"{faction} cannot afford {-delta} {res} (has {current})",
            state=state,
            faction=faction,
            cmd=cmd,
        )
    return replace(fs, **{_RESOURCE_ATTR[res]: new_value})


def _exchange_rate(faction: str, res1: str, res2: str) -> int | None:
    """Effective from-per-to exchange rate: faction overrides win over
    `BASE_EXCHANGE_RATES` per (from, to) pair (`Game/Factions.pm` lines 83-90).
    """
    overrides = FACTIONS[faction].exchange_rate_overrides
    if res1 in overrides and res2 in overrides[res1]:
        return overrides[res1][res2]
    return BASE_EXCHANGE_RATES.get(res1, {}).get(res2)


def _find_pending_index(state: GameState, faction: str, kind: str) -> int | None:
    for i, p in enumerate(state.pending):
        if p.faction == faction and p.kind == kind:
            return i
    return None


def _consume_pending_amount(
    pending: tuple[PendingDecision, ...], index: int, used: int
) -> tuple[PendingDecision, ...]:
    """Reduce the `amount` on `pending[index]` by `used`, dropping the entry
    entirely once it reaches 0.
    """
    entry = pending[index]
    remaining = entry.amount - used
    if remaining > 0:
        return pending[:index] + (replace(entry, amount=remaining),) + pending[index + 1 :]
    return pending[:index] + pending[index + 1 :]


def _advance_track(
    state: GameState, faction: str, fs: FactionState, cult: str, steps: int
) -> CultAdvance:
    return advance(
        state.cults[faction][cult],
        steps,
        keys_available=fs.keys,
        track_open=state.cult_10[cult] is None,
    )


def _apply_cult_advance(
    state: GameState, faction: str, fs: FactionState, cult: str, result: CultAdvance
) -> tuple[GameState, FactionState]:
    """Fold a `cults.advance` result into `state`/`fs`: track position,
    `cult_10` ownership + key spend, `cult_blocked`, and threshold power.
    Shared by `handle_gain_cult` and `handle_send`; callers still owe
    `with_faction` plus any handler-specific `fs` fields (priests, etc).
    """
    new_cults = {f: dict(v) for f, v in state.cults.items()}
    new_cults[faction][cult] = result.new_value

    new_cult_10 = dict(state.cult_10)
    new_keys = fs.keys
    new_cult_blocked = fs.cult_blocked
    if result.key_spent:
        new_keys -= 1
        new_cult_10[cult] = faction
    if result.blocked_at_9:
        new_cult_blocked = fs.cult_blocked | {cult}

    new_state = replace(state, cults=new_cults, cult_10=new_cult_10)
    new_fs = replace(
        fs, power=fs.power.gain(result.power_gained), keys=new_keys, cult_blocked=new_cult_blocked
    )
    return new_state, new_fs


# --------------------------------------------------------------------------
# Handlers
# --------------------------------------------------------------------------


def handle_convert(state: GameState, faction: str, cmd: ParsedCommand) -> GameState:
    """`convert N res1 to M res2`. Darklings' SH grants a one-shot "up to 3
    W to P at 1:1" allowance (`commands.pm` `command_convert` lines 383-387,
    counter seeded to 3 by `resources.pm` line 52). No base/override rate
    covers W->P otherwise, so this models the allowance as a queued
    `PendingDecision(faction, kind="convert_w_to_p", amount=N)` -- Task 8
    pushes it on SH build; this handler only consumes it, decrementing
    `amount` per call and popping at 0.
    """
    assert cmd.res1 is not None and cmd.res2 is not None
    assert cmd.n1 is not None and cmd.n2 is not None

    rate = _exchange_rate(faction, cmd.res1, cmd.res2)
    w_to_p_pending_index: int | None = None
    if rate is None and cmd.res1 == "W" and cmd.res2 == "P":
        w_to_p_pending_index = _find_pending_index(state, faction, "convert_w_to_p")
        if w_to_p_pending_index is not None:
            rate = 1

    if rate is None:
        raise EngineError(
            f"no exchange rate from {cmd.res1} to {cmd.res2}", state=state, faction=faction, cmd=cmd
        )
    wanted_from = cmd.n2 * rate
    if wanted_from != cmd.n1:
        raise EngineError(
            f"conversion to {cmd.n2} {cmd.res2} requires {wanted_from} {cmd.res1}, not {cmd.n1}",
            state=state,
            faction=faction,
            cmd=cmd,
        )

    if w_to_p_pending_index is not None:
        allowance = state.pending[w_to_p_pending_index].amount
        if cmd.n2 > allowance:
            raise EngineError(
                f"{faction} may convert at most {allowance} more W to P this SH allowance, "
                f"not {cmd.n2}",
                state=state,
                faction=faction,
                cmd=cmd,
            )

    fs = state.factions[faction]
    fs = _with_resource_delta(state, faction, fs, cmd.res1, -cmd.n1, cmd)
    fs = _with_resource_delta(state, faction, fs, cmd.res2, cmd.n2, cmd)
    new_state = with_faction(state, faction, fs)

    if w_to_p_pending_index is not None:
        new_state = replace(
            new_state,
            pending=_consume_pending_amount(state.pending, w_to_p_pending_index, cmd.n2),
        )

    return new_state


def handle_burn(state: GameState, faction: str, cmd: ParsedCommand) -> GameState:
    assert cmd.n1 is not None
    fs = state.factions[faction]
    try:
        power = fs.power.burn(cmd.n1)
    except ValueError as exc:
        raise EngineError(str(exc), state=state, faction=faction, cmd=cmd) from exc
    return with_faction(state, faction, replace(fs, power=power))


def handle_noop(state: GameState, faction: str, cmd: ParsedCommand) -> GameState:
    """`wait`/`done`/`resign`/`annotation`/`setup`: no state effect here --
    turn advancement/elimination/phase anchoring is later tasks' round
    machinery; this task only guarantees these verbs dispatch cleanly.
    """
    return state


def handle_lose_resource(state: GameState, faction: str, cmd: ParsedCommand) -> GameState:
    assert cmd.res1 is not None and cmd.n1 is not None
    fs = state.factions[faction]
    fs = _with_resource_delta(state, faction, fs, cmd.res1, -cmd.n1, cmd)
    return with_faction(state, faction, fs)


def handle_lose_marker(state: GameState, faction: str, cmd: ParsedCommand) -> GameState:
    """No-op placeholder: `-FREE_D`/`-FREE_TP`/`-FREE_TF`/`-BRIDGE` retire a
    one-shot marker from a build/special action. Task 10's
    `actions_power.handle_lose_marker` (module-bottom import below)
    replaces this entry with the real implementation once markers are
    tracked as queued `PendingDecision`s -- this body never runs once that
    import has happened.
    """
    return state


def handle_convert_marker(state: GameState, faction: str, cmd: ParsedCommand) -> GameState:
    """No-op bookkeeping row for `[+-]NCONVERT_X_TO_Y` (e.g. Darklings SH's
    `-3CONVERT_W_TO_P`): a receipt, not an instruction. The real W/P move
    happens through `handle_convert`'s `convert 3w to 3p`, gated by the
    `convert_w_to_p` pending it consumes (see that docstring; Task 8 pushes
    the pending on SH build). Deferral: if replay shows a bare
    `convert_marker` with no companion `convert`/pending, this will need to
    move W/P itself.
    """
    return state


def handle_gain_cult(state: GameState, faction: str, cmd: ParsedCommand) -> GameState:
    """`+N<CULT>`: advance `faction`'s track by N (default 1). If `faction`
    has a `cult_choice` pending queued *anywhere* (not just the head --
    Cultists' leech-effect choice, pushed mid-batch by `leech.py`, can
    legitimately sit behind still-outstanding sibling leech offers from
    the same build), this row is its answer and pops that specific entry
    (Cultists' leech choice, ACTA/BON2/FAV-action steps, town-tile cult
    gains).
    """
    assert cmd.cult is not None
    cult = cmd.cult
    steps = cmd.n1 if cmd.n1 is not None else 1
    fs = state.factions[faction]

    result = _advance_track(state, faction, fs, cult, steps)
    new_state, new_fs = _apply_cult_advance(state, faction, fs, cult, result)

    cult_choice_index = _find_pending_index(state, faction, "cult_choice")
    if cult_choice_index is not None:
        new_state = pop_pending(new_state, cult_choice_index)

    return with_faction(new_state, faction, new_fs)


def handle_lose_cult(state: GameState, faction: str, cmd: ParsedCommand) -> GameState:
    """Plain retreat: no power refund for thresholds already earned."""
    assert cmd.cult is not None
    cult = cmd.cult
    steps = cmd.n1 if cmd.n1 is not None else 1
    old_value = state.cults[faction][cult]
    new_value = max(0, old_value - steps)

    new_cults = {f: dict(v) for f, v in state.cults.items()}
    new_cults[faction][cult] = new_value
    return replace(state, cults=new_cults)


def handle_send(state: GameState, faction: str, cmd: ParsedCommand) -> GameState:
    """`send p to CULT[ for N]` (`commands.pm` `command_send` lines 313-350):
    first open slot wins (matching step `N` if given), costing 1
    `priest_pool`. If every slot is full the track still advances 1 step
    but no slot/`priest_pool` is touched -- the priest is always spent from
    hand either way (line 349's `adjust_resource` is unconditional).
    """
    assert cmd.cult is not None
    cult = cmd.cult
    fs = state.factions[faction]
    if fs.priests < 1:
        raise EngineError(f"{faction} has no priest to send", state=state, faction=faction, cmd=cmd)

    slots = state.priest_slots[cult]
    slot_index: int | None = None
    for i, occupant in enumerate(slots):
        if occupant is not None:
            continue
        if cmd.n1 is not None and PRIEST_SLOT_STEPS[i] != cmd.n1:
            continue
        slot_index = i
        break

    steps = PRIEST_SLOT_STEPS[slot_index] if slot_index is not None else 1
    if cmd.n1 is not None and steps != cmd.n1:
        raise EngineError(
            f"no open {cmd.n1}-step spot on {cult} track", state=state, faction=faction, cmd=cmd
        )

    result = _advance_track(state, faction, fs, cult, steps)

    new_priest_slots = dict(state.priest_slots)
    if slot_index is not None:
        updated = list(slots)
        updated[slot_index] = faction
        new_priest_slots[cult] = tuple(updated)

    new_priest_pool = fs.priest_pool - 1 if slot_index is not None else fs.priest_pool
    if new_priest_pool < 0:
        raise EngineError(
            f"{faction} has no priest_pool budget left", state=state, faction=faction, cmd=cmd
        )

    new_state, new_fs = _apply_cult_advance(state, faction, fs, cult, result)
    new_state = replace(new_state, priest_slots=new_priest_slots)
    new_fs = replace(new_fs, priests=fs.priests - 1, priest_pool=new_priest_pool)
    return with_faction(new_state, faction, new_fs)


register_handler("convert", handle_convert)
register_handler("burn", handle_burn)
register_handler("wait", handle_noop)
register_handler("done", handle_noop)
register_handler("resign", handle_noop)
register_handler("annotation", handle_noop)
register_handler("setup", handle_noop)
register_handler("send", handle_send)
register_handler("gain_cult", handle_gain_cult)
register_handler("lose_cult", handle_lose_cult)
register_handler("lose_resource", handle_lose_resource)
register_handler("lose_marker", handle_lose_marker)
register_handler("convert_marker", handle_convert_marker)


# --------------------------------------------------------------------------
# Side-effect imports: modules that register additional verbs into
# HANDLERS on import (Task 8's build/upgrade/bridge/favor/town/leech/
# decline; Task 9's dig/transform/lose_spade; Task 10's action/lose_marker;
# Task 11's pass/advance/connect and the three income verbs).
# Placed at the bottom, after every symbol those modules import from here
# (EngineError, push_pending, pop_pending, register_handler, state helpers)
# is already defined, so this is not a circular import:
# `actions_build`/`leech`/`actions_terraform`/`actions_power`/`actions_pass`/
# `round_flow` import *from* this module at their own top, and by the time
# Python reaches these lines this module's own top-to-bottom execution has
# already bound everything they need. `actions_terraform`'s real
# `lose_spade` handler is registered here too, replacing the placeholder
# entry this module never installs anymore (see `handle_lose_resource`'s
# neighbours above -- the no-op stub used to live here, Task 9 owns it).
# `actions_power`'s real `lose_marker` handler similarly replaces
# `handle_lose_marker` above (Task 10) -- import order among these six
# doesn't matter for correctness (only `actions_power` ever registers
# "lose_marker"; none of the others touch it), so they are kept
# alphabetical for the import-sorter.
from bgai.engine.tm import actions_build as _actions_build  # noqa: E402,F401
from bgai.engine.tm import actions_pass as _actions_pass  # noqa: E402,F401
from bgai.engine.tm import actions_power as _actions_power  # noqa: E402,F401
from bgai.engine.tm import actions_terraform as _actions_terraform  # noqa: E402,F401
from bgai.engine.tm import leech as _leech  # noqa: E402,F401
from bgai.engine.tm import round_flow as _round_flow  # noqa: E402,F401
from bgai.engine.tm import scoring as _scoring  # noqa: E402,F401
