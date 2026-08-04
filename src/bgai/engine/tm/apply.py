"""The engine's single entry point: `apply(state, faction, cmd) -> GameState`.

Dispatches on `cmd.verb` to handler functions registered in the frozen
`HANDLERS` table (see `register_handler`). This task implements the "simple"
verbs in full; later tasks register more handlers (build/upgrade/transform/
leech/...) into the same table without touching the dispatch mechanism.

Perl references (jsnell/terra-mystica):
- `command_convert` (`commands.pm` lines 352-399): merges
  `factions_data.BASE_EXCHANGE_RATES` with a faction's
  `exchange_rate_overrides` (`Game/Factions.pm` `initialize_faction`, lines
  83-90 -- overrides win per `from`/`to` pair, base rates fill the rest),
  then requires `to_count * rate[from][to] == from_count` exactly
  (line 392-395) before moving resources.
- `command_send` (`commands.pm` lines 313-350): tries cult-track slots
  `<CULT>1..4` in order (`cults.pm` `setup_cults`: slot 1 gains 3 steps,
  slots 2-4 gain 2 steps each -- `PRIEST_SLOT_STEPS`). The first *open*
  slot is taken (or, if an explicit step amount was requested, the first
  open slot whose step count matches); taking a slot marks it permanently
  occupied and decrements the faction's `MAX_P` (our `priest_pool`) by 1.
  If every slot is occupied, `$gain` is left at its initial `{ $cult => 1
  }` default: the track still advances 1 step, but no slot is claimed and
  `priest_pool` is *not* touched -- confirmed by reading the loop: it only
  mutates `$spot`/`MAX_P`/`CULT_P` inside the `if (!$spot->{building})`
  branch, which is never entered when nothing is open. `adjust_resource
  $faction, "P", -1` (line 349) runs unconditionally in both cases: the
  priest is always spent from hand, whether or not it ends up on the
  board.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from bgai.data.ledger_parser import ParsedCommand
from bgai.engine.tm.cults import PRIEST_SLOT_STEPS, advance
from bgai.engine.tm.factions_data import BASE_EXCHANGE_RATES, FACTIONS
from bgai.engine.tm.state import (
    FactionState,
    GameState,
    PendingDecision,
    active_faction,
    with_faction,
)


class EngineError(Exception):
    """Raised when `apply` cannot process a command; carries game context
    (round/phase/faction/raw command text) so callers can report failures
    without having to re-derive where in the replay they happened.
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
    """Register (or replace) the handler for `verb`. Direct `HANDLERS[verb] =
    handler` mutation works too -- this is just the documented spelling.
    """
    HANDLERS[verb] = handler


# Verbs that may be applied even when `faction` is not `active_faction(state)`.
_ORDER_EXEMPT_VERBS = frozenset({"wait", "annotation"})
_LEECH_ANSWER_VERBS = frozenset({"leech", "decline"})


def _has_queued_leech_for(state: GameState, faction: str) -> bool:
    """Whether any queued pending decision anywhere in the queue is a leech
    offer for `faction`. Snellman's strict-leech option restricts this to
    the faction's *own* first queued offer (`acting.pm`); for now (this
    task) any of that faction's queued leech offers may be answered
    out-of-turn -- Task 8 tightens this to strict-leech semantics.
    """
    return any(p.faction == faction and p.kind == "leech" for p in state.pending)


def apply(state: GameState, faction: str, cmd: ParsedCommand) -> GameState:
    """The engine's single entry point: dispatch `cmd` for `faction`."""
    exempt = cmd.verb in _ORDER_EXEMPT_VERBS or (
        cmd.verb in _LEECH_ANSWER_VERBS and _has_queued_leech_for(state, faction)
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
    if res == "PW":
        return fs.power.usable
    return getattr(fs, _RESOURCE_ATTR[res])


def _with_resource_delta(
    state: GameState, faction: str, fs: FactionState, res: str, delta: int, cmd: ParsedCommand
) -> FactionState:
    """Apply `delta` units of `res` to `fs`, raising `EngineError` if that
    would take a resource negative (system-boundary validation -- Perl lets
    `adjust_resource` go negative and relies on upstream legality checks
    that don't exist yet in this engine; we fail fast instead).
    """
    if res == "PW":
        power = fs.power.spend(-delta) if delta < 0 else fs.power.gain(delta)
        return replace(fs, power=power)

    attr = _RESOURCE_ATTR[res]
    new_value = getattr(fs, attr) + delta
    if new_value < 0:
        raise EngineError(
            f"{faction} cannot afford {-delta} {res} (has {getattr(fs, attr)})",
            state=state,
            faction=faction,
            cmd=cmd,
        )
    return replace(fs, **{attr: new_value})


def _exchange_rate(faction: str, res1: str, res2: str) -> int | None:
    """Effective from-per-to exchange rate: faction overrides win over
    `BASE_EXCHANGE_RATES` per (from, to) pair (`Game/Factions.pm` lines 83-90).
    """
    overrides = FACTIONS[faction].exchange_rate_overrides
    if res1 in overrides and res2 in overrides[res1]:
        return overrides[res1][res2]
    return BASE_EXCHANGE_RATES.get(res1, {}).get(res2)


# --------------------------------------------------------------------------
# Handlers
# --------------------------------------------------------------------------


def handle_convert(state: GameState, faction: str, cmd: ParsedCommand) -> GameState:
    assert cmd.res1 is not None and cmd.res2 is not None
    assert cmd.n1 is not None and cmd.n2 is not None
    rate = _exchange_rate(faction, cmd.res1, cmd.res2)
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
    fs = state.factions[faction]
    fs = _with_resource_delta(state, faction, fs, cmd.res1, -cmd.n1, cmd)
    fs = _with_resource_delta(state, faction, fs, cmd.res2, cmd.n2, cmd)
    return with_faction(state, faction, fs)


def handle_burn(state: GameState, faction: str, cmd: ParsedCommand) -> GameState:
    assert cmd.n1 is not None
    fs = state.factions[faction]
    try:
        power = fs.power.burn(cmd.n1)
    except ValueError as exc:
        raise EngineError(str(exc), state=state, faction=faction, cmd=cmd) from exc
    return with_faction(state, faction, replace(fs, power=power))


def handle_noop(state: GameState, faction: str, cmd: ParsedCommand) -> GameState:
    """`wait`/`done`/`resign`/`annotation`/`setup`: no state effect here.
    Turn advancement (`done`), faction elimination (`resign`), and phase
    anchoring (`setup`) belong to the round/turn-order machinery of a later
    task; this task only guarantees these verbs dispatch cleanly.
    """
    return state


def handle_lose_resource(state: GameState, faction: str, cmd: ParsedCommand) -> GameState:
    assert cmd.res1 is not None and cmd.n1 is not None
    fs = state.factions[faction]
    fs = _with_resource_delta(state, faction, fs, cmd.res1, -cmd.n1, cmd)
    return with_faction(state, faction, fs)


def handle_lose_spade(state: GameState, faction: str, cmd: ParsedCommand) -> GameState:
    """No-op stub: spade/terrain balance bookkeeping is Task 9's
    (`terraform.py` owns spade accounting); this task only registers the
    verb so `apply` doesn't reject `-Nspade` rows as unknown.
    """
    return state


def handle_lose_marker(state: GameState, faction: str, cmd: ParsedCommand) -> GameState:
    """No-op stub: `-FREE_D`/`-FREE_TP`/`-FREE_TF`/`-BRIDGE` retire a
    one-shot marker granted by a build/special action. Tracking those
    markers on `FactionState` is deferred to whichever later task
    (Task 9/10) introduces the build/special-action handlers that grant
    them; this task only registers the verb.
    """
    return state


def handle_convert_marker(state: GameState, faction: str, cmd: ParsedCommand) -> GameState:
    """No-op bookkeeping row for `[+-]NCONVERT_X_TO_Y` (e.g. Darklings SH's
    `-3CONVERT_W_TO_P`). Per `commands.pm` `command_convert` lines 383-387,
    the faction's `CONVERT_W_TO_P` counter only *enables* a temporarily
    better W->P exchange rate for a normal `convert` command (which does
    the real W/P movement via `handle_convert`); this row records that the
    one-shot allowance was granted/spent, not an independent resource
    move. Documented deferral: if corpus replay later shows a bare
    `convert_marker` row with no companion `convert` row in the same turn,
    this handler will need to perform the W/P move itself.
    """
    return state


def handle_gain_cult(state: GameState, faction: str, cmd: ParsedCommand) -> GameState:
    assert cmd.cult is not None
    cult = cmd.cult
    steps = cmd.n1 if cmd.n1 is not None else 1
    fs = state.factions[faction]

    result = advance(
        state.cults[faction][cult],
        steps,
        keys_available=fs.keys,
        track_open=state.cult_10[cult] is None,
    )

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

    new_fs = replace(
        fs, power=fs.power.gain(result.power_gained), keys=new_keys, cult_blocked=new_cult_blocked
    )

    new_pending = state.pending
    if new_pending and new_pending[0].faction == faction and new_pending[0].kind == "cult_choice":
        new_pending = new_pending[1:]

    new_state = replace(state, cults=new_cults, cult_10=new_cult_10, pending=new_pending)
    return with_faction(new_state, faction, new_fs)


def handle_lose_cult(state: GameState, faction: str, cmd: ParsedCommand) -> GameState:
    """Plain retreat: no power refund for losing thresholds already earned
    (matches the base game's cult-track rules -- power gained on the way up
    is never clawed back on the way down).
    """
    assert cmd.cult is not None
    cult = cmd.cult
    steps = cmd.n1 if cmd.n1 is not None else 1
    old_value = state.cults[faction][cult]
    new_value = max(0, old_value - steps)

    new_cults = {f: dict(v) for f, v in state.cults.items()}
    new_cults[faction][cult] = new_value
    return replace(state, cults=new_cults)


def handle_send(state: GameState, faction: str, cmd: ParsedCommand) -> GameState:
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

    result = advance(
        state.cults[faction][cult],
        steps,
        keys_available=fs.keys,
        track_open=state.cult_10[cult] is None,
    )

    new_priest_slots = dict(state.priest_slots)
    if slot_index is not None:
        updated = list(slots)
        updated[slot_index] = faction
        new_priest_slots[cult] = tuple(updated)

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

    new_priest_pool = fs.priest_pool - 1 if slot_index is not None else fs.priest_pool
    if new_priest_pool < 0:
        raise EngineError(
            f"{faction} has no priest_pool budget left", state=state, faction=faction, cmd=cmd
        )

    new_fs = replace(
        fs,
        priests=fs.priests - 1,
        priest_pool=new_priest_pool,
        power=fs.power.gain(result.power_gained),
        keys=new_keys,
        cult_blocked=new_cult_blocked,
    )

    new_state = replace(state, cults=new_cults, priest_slots=new_priest_slots, cult_10=new_cult_10)
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
register_handler("lose_spade", handle_lose_spade)
register_handler("lose_marker", handle_lose_marker)
register_handler("convert_marker", handle_convert_marker)
