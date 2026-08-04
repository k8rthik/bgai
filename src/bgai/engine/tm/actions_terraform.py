"""``dig``/``transform``/``lose_spade`` handlers.

Ported from the reference implementation (jsnell/terra-mystica, MIT),
``src/commands.pm``:

- ``command_dig`` (670-698): pays ``$faction->{dig}{cost}[level]`` and
  gains ``$faction->{dig}{gain}[level]`` (both indexed by the faction's
  *current* dig-track level -- ``factions_data.DigTrack.cost``/``.dig_gain``
  here), ``$amount`` times. When ``dig_gain`` is empty/falsy for that level
  (every base faction except Darklings), the spade grant is a plain
  ``adjust_resource($faction, 'SPADE', $amount)`` (690); when it's set
  (Darklings' single level: ``{SPADE=>1, VP=>2}``), the whole gain dict is
  applied once per unit via ``gain()`` (693) instead -- net effect either
  way is ``$amount`` spades added to the balance, Darklings' dig additionally
  scoring 2 VP per spade and never advancing the (max-level-0) track.
- ``command_transform`` (588-668) + ``map.pm`` ``transform_cost``
  (554-618)/``transform_colors``(505-535): spade cost is
  ``abs(color_difference(current, target))`` the short way around the
  7-color wheel (``terraform.spade_distance`` here), forced to exactly 2
  whenever nonzero for Giants regardless of true distance (map.pm 610-614:
  ``if ($faction->{name} eq 'giants' and $color_difference != 0) {
  $color_difference = 2; }``); an unspecified target color defaults to the
  faction's own color (``transform_colors``, 505-535 -- our engine has no
  ``secondary_color``/ice mechanic, so the full bidirectional
  ``transform_colors_on_cycle`` wheel-walk that F&I factions need collapses
  to "home" for every base faction). ``pay $faction, $transform_cost``
  (631) is the ``spades_available`` debit; a request that can't be paid
  raises here exactly like any other insufficient-resource case
  (``resources.pm`` ``adjust_resource``'s ``$faction->{$type} < 0`` die,
  400-402).
- ``resources.pm`` ``adjust_resource`` (250-403): for any *positive* delta
  to a plain resource (including ``SPADE``), the generic branch (313-333)
  loops ``for (1..$delta)`` calling ``maybe_score_current_score_tile`` and
  ``maybe_gain_faction_special`` with mode ``'gain'`` once per unit gained
  (389-391) -- since a negative ``$delta``'s ``1..$delta`` range is empty
  in Perl, this **only fires on gain, never on spend**. This settles the
  brief's open question: Halflings' ``special => { mode => gain, SPADE =>
  {VP=>1} }`` and Alchemists' ``special => { SPADE => {PW=>2}, mode => gain,
  enable_if => {SH=>1} }`` (``Game/Factions/Halflings.pm``/``Alchemists.pm``)
  both fire **once per spade gained** (dig, or a Halflings SH's 3-spade
  grant), never per spade spent on a transform -- ``command_transform``'s
  own per-spade-spent loop (634-640) calls ``maybe_gain_faction_special``
  with mode ``'spend'``, which is a no-op for every base faction (no base
  faction's ``special`` uses ``mode => 'spend'``; grepped the whole
  ``src/Game/Factions/*.pm`` tree -- zero hits).
- ``commands.pm`` line 98 + resources.pm: a bare ``-SPADE``/``-Nspade``
  row (``lose_spade`` here) is a **plain decrement** of the balance by N
  (``adjust_resource($faction, 'SPADE', -N)``), not an unconditional
  "zero it out" -- confirmed against the crawled corpus (task-9 report):
  a Darklings row ``"dig 3. transform G7 to black. -2SPADE"`` discards
  exactly 2 of the 3 just-dug spades after spending 1 on the transform,
  and every ``-Nspade``/``-SPADE`` row's N always matches the raw text's
  digit (default 1 when bare). ``handle_lose_spade`` mirrors
  ``apply.py``'s ``handle_lose_resource`` for this reason: subtract N,
  ``EngineError`` if that would go negative.

Giants' 202-for-202 real ``transform ... to red`` rows in the crawled
corpus (task-9 report) empirically back the brief's stricter contract over
the letter of ``command_transform`` (which technically lets *any* faction,
Giants included, name an explicit non-home target -- ``validate_transform_color``
only rejects ice/volcano, 495-503): this module raises ``EngineError`` on an
explicit non-home ``to color`` for Giants rather than silently allowing it.

``FactionState.spades_available`` (state.py) is the single running balance
``dig``/``transform``/``lose_spade`` all read and write -- a still-pending
``halflings_spades`` grant (pushed by ``actions_build.py`` on Halflings' SH
build, an amount=3 marker rather than an immediate balance bump) is folded
into it the first time either handler needs the balance, at which point the
Halflings VP-per-spade fires for the whole grant at once (closest available
analog to Perl's "granted and scored the instant the SH goes up" timing,
since this engine defers the literal ``SPADE`` resource bump to a pending
rather than mutating ``fs`` from inside ``actions_build.handle_upgrade``).
ACT5/ACT6/BON1 spade grants (Task 10) are the seam this leaves open: as
long as whatever adds to that balance also goes through
``FactionState.spades_available``, ``transform``/``lose_spade`` here don't
need to change at all.
"""

from __future__ import annotations

from dataclasses import replace

from bgai.data.ledger_parser import ParsedCommand
from bgai.engine.tm.apply import EngineError, pop_pending, register_handler
from bgai.engine.tm.board import RIVER
from bgai.engine.tm.connectivity import reachable
from bgai.engine.tm.factions.hooks import HOOKS, FactionHooks, hooks_for
from bgai.engine.tm.factions_data import FACTIONS
from bgai.engine.tm.state import FactionState, GameState, with_faction
from bgai.engine.tm.terraform import spade_distance

_DIG_COST_RES: dict[str, str] = {"W": "workers", "C": "coins", "P": "priests"}

# map.pm alias_color (438-444): the ledger/UI accepts "grey" as a synonym
# for the wheel's "gray".
_COLOR_ALIASES: dict[str, str] = {"grey": "gray"}


# --------------------------------------------------------------------------
# Faction hooks
# --------------------------------------------------------------------------


class _GiantsHooks(FactionHooks):
    """map.pm 505-514/610-614: every transform targets home terrain."""

    def spade_transform_target(self, state: GameState, faction: str, loc: str, color: str) -> str:
        return FACTIONS[faction].color


class _SpadeSpecialGainHooks(FactionHooks):
    """Per-spade-gained bonus sourced from ``FactionData.special_gain['SPADE']``,
    gated by ``special_requires_sh`` -- Alchemists' ``{PW: 2}`` needs their
    stronghold built, Halflings' ``{VP: 1}`` is unconditional (see module
    docstring's ``maybe_gain_faction_special`` citation for why this is a
    *gain*-time hook, not a spend-time one). Shared by both factions rather
    than duplicated per-faction subclasses since the logic is identical
    data lookup + one gate check.
    """

    def extra_dig_gain(self, state: GameState, faction: str) -> dict[str, int]:
        data = FACTIONS[faction]
        special = data.special_gain.get("SPADE")
        if not special:
            return {}
        if data.special_requires_sh and not state.factions[faction].buildings.get("SH"):
            return {}
        return dict(special)


HOOKS["giants"] = _GiantsHooks()
HOOKS["halflings"] = _SpadeSpecialGainHooks()
HOOKS["alchemists"] = _SpadeSpecialGainHooks()


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------


def _alias_color(color: str) -> str:
    return _COLOR_ALIASES.get(color, color)


def _find_pending_index(state: GameState, faction: str, kind: str) -> int | None:
    for i, p in enumerate(state.pending):
        if p.faction == faction and p.kind == kind:
            return i
    return None


def _pay_resources(
    state: GameState, faction: str, fs: FactionState, cost: dict[str, int], cmd: ParsedCommand
) -> FactionState:
    for res, amount in cost.items():
        if not amount:
            continue
        attr = _DIG_COST_RES[res]
        new_value = getattr(fs, attr) - amount
        if new_value < 0:
            raise EngineError(
                f"{faction} cannot afford {amount} {res} (has {getattr(fs, attr)})",
                state=state,
                faction=faction,
                cmd=cmd,
            )
        fs = replace(fs, **{attr: new_value})
    return fs


def _apply_extra_dig_gain(
    state: GameState, faction: str, fs: FactionState, spade_count: int
) -> FactionState:
    """Fold ``hooks_for(faction).extra_dig_gain`` into ``fs``, scaled by
    ``spade_count`` spades just gained (dig, or a ``halflings_spades``
    pending absorption -- see module docstring). ``state`` is read for the
    Alchemists SH gate only; callers must pass the state *before* this
    gain (buildings never change here, so any snapshot works).
    """
    if spade_count <= 0:
        return fs
    extra = hooks_for(faction).extra_dig_gain(state, faction)
    for res, per_unit in extra.items():
        total = per_unit * spade_count
        if not total:
            continue
        if res == "VP":
            fs = replace(fs, vp=fs.vp + total)
        elif res == "PW":
            fs = replace(fs, power=fs.power.gain(total))
        else:
            raise ValueError(f"unhandled extra_dig_gain key {res!r} for {faction}")
    return fs


def _absorb_halflings_pending(
    state: GameState, faction: str, fs: FactionState
) -> tuple[GameState, FactionState]:
    """Fold a still-outstanding ``halflings_spades`` pending (pushed by
    ``actions_build.py`` on Halflings' SH build) into ``fs.spades_available``,
    firing the Halflings VP-per-spade bonus for the whole grant at once
    (module docstring). No-op if no such pending is queued, or for any
    other faction (the pending's ``kind`` is Halflings-only).
    """
    idx = _find_pending_index(state, faction, "halflings_spades")
    if idx is None:
        return state, fs
    amount = state.pending[idx].amount
    new_state = pop_pending(state, idx)
    fs = replace(fs, spades_available=fs.spades_available + amount)
    fs = _apply_extra_dig_gain(state, faction, fs, amount)
    return new_state, fs


# --------------------------------------------------------------------------
# Handlers
# --------------------------------------------------------------------------


def handle_dig(state: GameState, faction: str, cmd: ParsedCommand) -> GameState:
    """``dig N[ LOC]``: pay N times the current dig level's cost, gain N
    spades (module docstring). ``cmd.loc`` (the rare "DIG N LOC" form) is
    a location hint the crawled corpus always follows with its own
    separate explicit ``transform LOC ...`` row -- ignored here as
    purely informational (task-9 report).
    """
    assert cmd.n1 is not None
    amount = cmd.n1
    fs = state.factions[faction]
    dig_track = FACTIONS[faction].dig
    level = fs.dig_level
    if level >= len(dig_track.cost):
        raise EngineError(
            f"{faction} has no dig cost defined for level {level}",
            state=state,
            faction=faction,
            cmd=cmd,
        )

    cost = {res: amt * amount for res, amt in dig_track.cost[level].items()}
    fs = _pay_resources(state, faction, fs, cost, cmd)

    gain = dig_track.dig_gain[level] if level < len(dig_track.dig_gain) else {}
    spades = 0
    if gain:
        for key, per_unit in gain.items():
            total = per_unit * amount
            if key == "SPADE":
                spades = total
            elif key == "VP":
                fs = replace(fs, vp=fs.vp + total)
            else:
                raise ValueError(f"unhandled dig_gain key {key!r} for {faction}")
    else:
        spades = amount

    fs = replace(fs, spades_available=fs.spades_available + spades)
    fs = _apply_extra_dig_gain(state, faction, fs, spades)
    return with_faction(state, faction, fs)


def handle_transform(state: GameState, faction: str, cmd: ParsedCommand) -> GameState:
    """``transform HEX[ to COLOR]``: spend ``spades_available`` to recolor
    an unoccupied, non-river, reachable hex (module docstring).
    """
    assert cmd.loc is not None
    hex_key = cmd.loc
    if hex_key not in state.hexes:
        raise EngineError(f"unknown hex {hex_key!r}", state=state, faction=faction, cmd=cmd)

    hex_state = state.hexes[hex_key]
    if hex_state.building is not None:
        raise EngineError(
            f"{hex_key} already contains a {hex_state.building}",
            state=state,
            faction=faction,
            cmd=cmd,
        )
    if hex_state.color == RIVER:
        raise EngineError(
            f"{hex_key} is a river hex, cannot be transformed",
            state=state,
            faction=faction,
            cmd=cmd,
        )
    if hex_key not in reachable(state, faction):
        raise EngineError(
            f"{hex_key} is not reachable by {faction}", state=state, faction=faction, cmd=cmd
        )

    home_color = FACTIONS[faction].color
    requested_color = _alias_color(cmd.color) if cmd.color is not None else None
    proposed_color = requested_color if requested_color is not None else home_color
    effective_color = hooks_for(faction).spade_transform_target(
        state, faction, hex_key, proposed_color
    )

    if requested_color is not None and effective_color != requested_color:
        raise EngineError(
            f"{faction} must transform to {effective_color}, not {requested_color}",
            state=state,
            faction=faction,
            cmd=cmd,
        )
    if hex_state.color == effective_color:
        raise EngineError(
            f"{hex_key} is already {effective_color}", state=state, faction=faction, cmd=cmd
        )

    # Giants always pay exactly 2 spades for any nonzero-distance transform
    # (map.pm 610-614); every other faction pays the true wheel distance.
    cost = 2 if faction == "giants" else spade_distance(hex_state.color, effective_color)

    fs = state.factions[faction]
    new_state, fs = _absorb_halflings_pending(state, faction, fs)

    if cost > fs.spades_available:
        raise EngineError(
            f"{faction} has {fs.spades_available} spades available, needs {cost} to "
            f"transform {hex_key}",
            state=new_state,
            faction=faction,
            cmd=cmd,
        )
    fs = replace(fs, spades_available=fs.spades_available - cost)

    new_hexes = dict(new_state.hexes)
    new_hexes[hex_key] = replace(hex_state, color=effective_color)
    return replace(with_faction(new_state, faction, fs), hexes=new_hexes)


def handle_lose_spade(state: GameState, faction: str, cmd: ParsedCommand) -> GameState:
    """``-Nspade``/``-SPADE``: discard N (default 1) spades from the
    balance -- a plain decrement, not a hard reset to 0 (module docstring).
    """
    n = cmd.n1 if cmd.n1 is not None else 1
    fs = state.factions[faction]
    new_state, fs = _absorb_halflings_pending(state, faction, fs)

    new_value = fs.spades_available - n
    if new_value < 0:
        raise EngineError(
            f"{faction} cannot discard {n} spades (has {fs.spades_available})",
            state=new_state,
            faction=faction,
            cmd=cmd,
        )
    fs = replace(fs, spades_available=new_value)
    return with_faction(new_state, faction, fs)


register_handler("dig", handle_dig)
register_handler("transform", handle_transform)
register_handler("lose_spade", handle_lose_spade)
