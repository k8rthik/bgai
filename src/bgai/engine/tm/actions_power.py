"""``action ACTx``/``BONx``/``FAVx`` handler: the power wheel plus every
faction/bonus/favor special action.

Ported from the reference implementation (jsnell/terra-mystica, MIT):

- ``Game/Constants.pm`` ``%actions`` (lines 55-105): ACT1-6's ``cost``/
  ``gain`` (``tiles.POWER_ACTIONS`` here), and the faction special actions
  ACTA/ACTC/ACTE/ACTG/ACTN/ACTS/ACTW (``factions_data.FACTION_SPECIAL_ACTIONS``).
  BON1/BON2/FAV6's ``special_action`` (``tiles.py``) are the same ``%actions``
  entries, merged onto those tiles by ``init_tiles`` (``tiles.py``'s own
  docstring).
- ``commands.pm`` ``command_action`` (lines 834-885): looked up via
  ``$faction->{$action}`` unless the action is ACT1-6 (die "No $action space
  available" otherwise -- ``_check_special_action_available`` below); pays
  ``$ar->{cost}`` (``Power.spend`` for ACT1-6's ``PW`` cost, plain W/C/P via
  ``_pay_simple`` for ACTE's ``{W=>2}`` -- the only nonzero faction-special
  cost); ``gain $faction, $ar->{gain}, $name`` (``_apply_action_gain``);
  blocks the action space (``$map{$action}{blocked} = 1 unless
  $ar->{dont_block}``, line 857) keyed by ``$action`` **after** line 853-855's
  ``$action .= "/$faction_name" if $action !~ /^ACT/``. Read literally,
  that regex only appends the per-faction suffix to the **non**-``ACT``-
  prefixed ids -- BON1/BON2/FAV6 -- so those alone get a genuinely
  per-faction blocking key; every ``ACT``-prefixed id, ACT1-6 *and*
  ACTA/ACTC/ACTE/ACTG/ACTN/ACTS/ACTW alike, keeps the bare id and so
  shares Perl's literal blocking *mechanism* with ACT1-6. That said, each
  faction-special id (ACTA/ACTC/ACTG/ACTN/ACTS/ACTW) is only ever grantable
  to exactly one faction per game (each base faction's stronghold grants a
  distinct, unique id -- ``factions_data.py``'s ``FACTIONS`` table has no
  two factions sharing an SH ``build_gain`` ACT-key), so a bare-id
  "global" key and a per-faction key are **observably identical** for
  those six ids: nobody else can ever reach the ``$faction->{$action}``
  gate to contend for the same key regardless of which scheme is used.
  This port therefore tracks ACT1-6 on ``GameState.power_actions_taken``
  (a literal, observable global, since multiple factions genuinely *can*
  contend for the same ACT1-6 tile) and ACTA/ACTC/ACTG/ACTN/ACTS/ACTW on
  ``FactionState.actions_used`` (per-faction storage for a fact that
  happens to be per-faction-exclusive already) -- purely a data-modeling
  convenience, not a behavioral divergence from Perl. BON1/BON2/FAV6 also
  use ``actions_used``, this time matching Perl's per-faction key exactly
  (a bonus/favor tile *can* pass between factions across rounds, or exist
  as several favor copies held by different factions at once, so
  per-faction blocking is both correct and load-bearing there). ACTE
  alone sets ``dont_block => 1`` (Constants.pm line 67) so it is **never**
  blocked at all -- reusable every turn, gated only by ``BRIDGE_COUNT``
  (``handle_bridge`` in ``actions_build.py``) same as any other bridge
  placement. Every ``{blocked}`` flag, global-keyed and per-faction-keyed
  alike, is reset uniformly at the start of each round by ``commands.pm``
  ``command_start`` (line 900, ``$map{$_}{blocked} = 0 for keys %map``) --
  the reset itself is Task 11's, not this task's; this module only sets
  the flags, never clears them.
- ``resources.pm`` ``adjust_resource``'s generic branch (lines ~327-331):
  ``if (exists $faction->{"MAX_$type"}) { ... if ($faction->{$type} >
  $max) { $faction->{$type} = $max } }`` -- a gain that would exceed a
  tracked max is **silently clamped**, never an error. ACT2's ``P => 1``
  gain is the only base-power-action gain this can ever bind on; our
  ``priest_pool`` field (``state.py``) *is* Perl's live ``MAX_P`` (started
  at ``MAX_PRIESTS`` = 7, permanently decremented by ``command_send``
  (``commands.pm`` line 333, ``$faction->{MAX_P}--``) every time a priest
  commits to a *new* cult-track slot -- exactly what ``apply.py``'s
  ``handle_send`` already decrements ``priest_pool`` for). So ACT2 clamps
  ``fs.priests`` to ``fs.priest_pool``, it never raises
  ``EngineError`` for "too many priests".
- ``resources.pm`` ``adjust_resource`` (lines 313-333, cross-referenced by
  ``actions_terraform.py``'s/``actions_build.py``'s own docstrings): any
  positive ``SPADE`` delta -- ACT5/ACT6/ACTG/BON1 included, not just
  ``dig``'s -- fires a faction's ``special_gain['SPADE']`` hook once per
  spade (Alchemists PW, Halflings VP). ``_apply_spade_gain_bonus`` here is
  a third private copy of the same fold ``actions_build.py``'s
  ``_apply_spade_gain_bonus``/``actions_terraform.py``'s
  ``_apply_extra_dig_gain`` already carry -- kept local rather than
  imported, per those modules' own stated rationale (no cross-module
  coupling between build/terraform/power for one four-line fold).
- **ACTG (Giants, 2 free spades)**: ``FACTION_SPECIAL_ACTIONS["ACTG"]``'s
  gain is a bare ``{"SPADE": 2}`` -- *not* a ``FREE_TF``-style marker
  (Constants.pm line 70: only ACTN carries ``FREE_TF``/
  ``TF_NEED_HEX_ADJACENCY``). It is handled by the exact same "SPADE"
  branch as ACT5/ACT6/dig: it lands on ``fs.spades_available``, and a
  following ``transform``/``build`` row spends it through
  ``actions_terraform.handle_transform`` unchanged. "Can ACTG's 2 spades be
  split across two transforms?" -- no, and this needs no extra code: Giants'
  ``spade_transform_cost`` hook (``actions_terraform._GiantsHooks``,
  Task 9) already charges a **flat 2** for *any* color change regardless of
  wheel distance, so a single non-home-color hex always costs exactly 0 or
  2 -- there is no cheaper partial transform to split into.
- **ACTN (Nomads Sandstorm)**: gain is ``{"FREE_TF": 1,
  "TF_NEED_HEX_ADJACENCY": 1}`` (Constants.pm line 74-75) -- these two
  pseudo-resources are only ever granted together for a base faction, so
  this port folds them into one marker, ``PendingDecision(kind="free_tf")``,
  consumed by ``actions_terraform.handle_transform`` (see that module's
  updated docstring/diff): forces the target to home color
  (``map.pm`` 511-514, "FREE_TF short-circuits straight to home"), requires
  direct **plain board** adjacency to one of the faction's own building
  hexes instead of the usual ``reachable()`` (``map.pm`` 641-651,
  ``TF_NEED_HEX_ADJACENCY`` -- ``next if $map{$where}{bridge}{$from}``
  explicitly *excludes* a ``$from`` reached only via a bridge before
  checking plain ``{adjacent}``, then ``die "Direct non-bridge adjacency
  required for transforming\n" if !$ok`` -- bridge-only adjacency does
  **not** satisfy this check, unlike ``reachable()``/leech, which do fold
  bridges in), and costs 0 spades (``map.pm`` 635-636:
  ``$cost->{FREE_TF} += 1`` *instead of* ``$cost->{SPADE} += ...``) rather
  than touching ``spades_available``. Real corpus rows for ACTN show both
  an explicit
  ``transform X to color`` row before the ``build`` (when the target needs
  recoloring) and a bare ``build X`` with no transform text at all (when X
  already happens to be home-colored, e.g. from an earlier turn's dig) --
  both replay correctly against this design without special-casing, since a
  no-op-color ``build`` never touches the ``free_tf`` pending at all (it is
  simply left outstanding, matching Perl's own "unused FREE_TF" incomplete-
  turn warning, ``acting.pm`` line 620 -- a Task 11 concern, not this
  module's).
- **ACTW (Witches' Ride)**: gain is ``{"FREE_D": 1, "TELEPORT_NO_TF": 1}``.
  Read against ``commands.pm`` ``command_build`` (161-164 grants the free
  build; 207-210: if ``TELEPORT_NO_TF`` is set and the target hex still
  needs a color change, it **dies** -- "Transforming terrain forbidden
  during this action") this proves the popular shorthand "build on ANY
  hex" is wrong: ``build_color_ok`` (``map.pm`` 657-664) is never bypassed,
  so the target must already be the faction's home color (green, for
  Witches) -- what ``TELEPORT_NO_TF`` actually removes is the *reachability*
  check (``check_reachable``, only called in the ``else`` branch that
  ``TELEPORT_NO_TF`` short-circuits around) and the cost. This corrects
  ``actions_build.py``'s Task-8 seam comment, which assumed a color bypass;
  folded into one marker, ``PendingDecision(kind="free_d")`` -- see
  ``actions_build.handle_build``'s updated docstring/diff. ``note_leech``
  (``commands.pm`` line 231) is unconditional in ``command_build``, so a
  Witches' Ride build still queues leech offers exactly like any other
  build -- confirmed against the corpus's ``action ACTW. build X`` rows,
  which are always followed by ordinary ``Leech N from witches`` rows from
  adjacent opponents when one exists.
- **ACTS (Swarmlings)**: gain is ``{"FREE_TP": 1}``. ``command_upgrade``
  (``commands.pm`` 279-293) computes ``%this_leech = note_leech $faction,
  $where`` *before* the free/cost branch, so ``FREE_TP`` -- folded into
  ``PendingDecision(kind="free_tp")``, consumed by
  ``actions_build.handle_upgrade`` -- makes the D->TP upgrade free but does
  **not** suppress its leech offers, matching the corpus's
  ``action ACTS. Upgrade X to TP`` rows, which trigger ordinary leech
  exactly like a paid upgrade.
- **ACTA (Auren)**: gain is ``{"CULT": 2, "CULTS_ON_SAME_TRACK": 1}``.
  Every real corpus row spends the whole grant in one combined
  ``+2<CULT>`` (e.g. ``action ACTA. +2FIRE``) -- ``apply.py``'s existing
  ``handle_gain_cult`` already advances a track by ``cmd.n1`` steps
  unconditionally, with no marker to satisfy and no turn-order gate (the
  ``+2<CULT>`` row is the *same* faction, immediately after, in the same
  ledger row -- ``active_faction`` never changes). So ACTA needs **no**
  pending at all: this handler only pays (free), marks ``ACTA`` used, and
  relies on the pre-existing ``gain_cult`` handler for the actual track
  advance. ``CULT``/``CULTS_ON_SAME_TRACK`` are therefore documented no-ops
  in ``_apply_action_gain`` -- ``command_adjust_resources``'s same-track
  enforcement (``commands.pm`` 62-64) only matters for a *split* ``+1CULT.
  +1CULT`` sequence, which the crawled corpus never uses for ACTA; BON2 and
  FAV6's bare ``CULT: 1`` gains follow the identical "companion row already
  exists, no pending needed" reasoning.
- **ACTC (Chaos Magicians)**: gain is ``{"GAIN_ACTION": 2}``
  (``resources.pm`` line ~289, ``$faction->{allowed_actions} += $delta``).
  This module only records the grant onto the new
  ``FactionState.extra_actions`` field (see ``state.py``'s docstring for
  the full contract) -- spending it back down across a real second turn is
  round/turn-order machinery that belongs to Task 11, not this one. The
  ``strict-chaosmagician-sh`` option only gates ``command_pass``
  (``commands.pm`` 771-773: clamp ``allowed_actions`` to 1 before passing;
  796: forbid passing twice in one round) -- entirely pass-command
  behavior, a documented no-op here (this module never sees a ``pass``
  verb). ``actions_pass.py``'s ``handle_pass`` is where that clamp
  actually lives (task-14 fix: an earlier revision left it as a no-op
  there too, which the 10-game sample never disproved since none of
  those games ever pass with a live ``extra_actions`` balance still
  banked).

``-FREE_D``/``-FREE_TP``/``-FREE_TF``/``-BRIDGE`` (``lose_marker``,
parsed with ``cmd.reason``) are the corpus's "declined the grant" rows --
real examples: ``action ACTW. -FREE_D``, ``action ACTS. -FREE_TP``,
``action ACTN. -FREE_TF``, ``action ACT1. -BRIDGE``. ``handle_lose_marker``
here replaces ``apply.py``'s no-op stub (module-bottom import, same
override pattern ``actions_terraform.py`` used for ``lose_spade``): it
looks up the matching pending by marker->kind and pops it unconsumed, no
resource effect.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

from bgai.data.ledger_parser import ParsedCommand
from bgai.engine.tm.apply import EngineError, pop_pending, push_pending, register_handler
from bgai.engine.tm.factions.hooks import hooks_for
from bgai.engine.tm.factions_data import FACTION_SPECIAL_ACTIONS, FACTIONS
from bgai.engine.tm.state import FactionState, GameState, PendingDecision, with_faction
from bgai.engine.tm.tiles import BONUS_TILES, FAVOR_TILES, POWER_ACTIONS, scored_vp

_COST_RES: dict[str, str] = {"W": "workers", "C": "coins", "P": "priests"}

# FactionState.actions_used tracks ACTA/ACTC/ACTG/ACTN/ACTS/ACTW and
# BON1/BON2/FAV6 per-faction (module docstring: literally correct for the
# non-ACT BON/FAV ids per commands.pm 853-855's "/$faction_name" suffix;
# an observably-equivalent modeling choice for the ACT-prefixed faction
# specials, which Perl's own regex keys the same bare way as ACT1-6 but
# which are single-faction-exclusive in practice anyway). ACTE alone opts
# out of blocking entirely (Constants.pm "dont_block => 1", line 67).
_UNBLOCKED_SPECIAL_ACTIONS = frozenset({"ACTE"})

# Marker gain keys that fold into a one-shot PendingDecision, consumed by a
# later build/upgrade/transform/bridge row (module docstring).
_MARKER_GAIN_KIND: dict[str, str] = {
    "BRIDGE": "bridge",
    "FREE_D": "free_d",
    "FREE_TP": "free_tp",
    "FREE_TF": "free_tf",
}

# Gain keys that are documented no-ops here: the companion effect either
# already has its own handler (CULT -- a following `+N<cult>` row, see
# module docstring) or is folded into another marker's semantics
# (CULTS_ON_SAME_TRACK rides along with ACTA's CULT; TELEPORT_NO_TF rides
# along with ACTW's FREE_D; TF_NEED_HEX_ADJACENCY rides along with ACTN's
# FREE_TF, enforced directly by actions_terraform.handle_transform).
_NO_OP_GAIN_KEYS = frozenset(
    {"CULT", "CULTS_ON_SAME_TRACK", "TELEPORT_NO_TF", "TF_NEED_HEX_ADJACENCY"}
)


def _find_pending_index(state: GameState, faction: str, kind: str) -> int | None:
    for i, p in enumerate(state.pending):
        if p.faction == faction and p.kind == kind:
            return i
    return None


def _pay_simple(
    state: GameState, faction: str, fs: FactionState, cost: Mapping[str, int], cmd: ParsedCommand
) -> FactionState:
    """Plain W/C/P debit (ACT1-6's ``PW`` cost goes through ``Power.spend``
    instead, in ``_pay_power``; every faction-special cost besides ACTE's
    ``{W: 2}`` is empty).
    """
    for res, amount in cost.items():
        if not amount:
            continue
        attr = _COST_RES[res]
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


def _current_score_tile_vp(state: GameState, type_: str, mode: str) -> int:
    """``actions_build.py``'s/``actions_terraform.py``'s identically-named
    helper, a fourth private copy by the same no-cross-coupling rationale
    (see ``tiles.scored_vp``'s docstring). ACT5/ACT6/BON1 never fire
    during setup, but the guard is kept for parity.
    """
    if state.round < 1:
        return 0
    tile = state.setup.score_tiles[state.round - 1]
    return scored_vp(tile, type_, mode)


def _apply_spade_gain_bonus(
    state: GameState, faction: str, fs: FactionState, spades: int
) -> FactionState:
    """Fold ``hooks_for(faction).extra_dig_gain`` into ``fs``, scaled by
    ``spades`` just gained (ACT5/ACT6/ACTG/BON1 -- module docstring; a
    third private copy alongside ``actions_build.py``'s/
    ``actions_terraform.py``'s, by the same no-cross-coupling rationale
    those modules already document).
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


def _apply_action_gain(
    state: GameState, faction: str, gain: Mapping[str, int], cmd: ParsedCommand
) -> GameState:
    """Fold one ``%actions`` ``gain`` dict into state (module docstring)."""
    fs = state.factions[faction]
    pendings: list[PendingDecision] = []
    for key, amount in gain.items():
        if not amount:
            continue
        if key in _MARKER_GAIN_KIND:
            pendings.append(
                PendingDecision(faction=faction, kind=_MARKER_GAIN_KIND[key], amount=amount)
            )
        elif key == "P":
            fs = replace(fs, priests=min(fs.priests + amount, fs.priest_pool))
        elif key == "W":
            fs = replace(fs, workers=fs.workers + amount)
        elif key == "C":
            fs = replace(fs, coins=fs.coins + amount)
        elif key == "SPADE":
            fs = replace(fs, spades_available=fs.spades_available + amount)
            fs = _apply_spade_gain_bonus(state, faction, fs, amount)
            fs = replace(fs, vp=fs.vp + amount * _current_score_tile_vp(state, "SPADE", "gain"))
        elif key == "GAIN_ACTION":
            fs = replace(fs, extra_actions=fs.extra_actions + amount)
        elif key in _NO_OP_GAIN_KEYS:
            continue
        else:
            raise ValueError(f"unhandled action gain key {key!r} for {faction}")
    new_state = with_faction(state, faction, fs)
    return push_pending(new_state, *pendings) if pendings else new_state


def _pay_power(
    state: GameState, faction: str, fs: FactionState, cost_power: int, cmd: ParsedCommand
) -> FactionState:
    try:
        power = fs.power.spend(cost_power)
    except ValueError as exc:
        raise EngineError(str(exc), state=state, faction=faction, cmd=cmd) from exc
    return replace(fs, power=power)


def _handle_power_wheel_action(
    state: GameState, faction: str, cmd: ParsedCommand, tile: str
) -> GameState:
    action = POWER_ACTIONS[tile]
    if tile in state.power_actions_taken:
        raise EngineError(
            f"power action space {tile} is blocked this round",
            state=state,
            faction=faction,
            cmd=cmd,
        )
    fs = _pay_power(state, faction, state.factions[faction], action.cost_power, cmd)
    new_state = with_faction(state, faction, fs)
    new_state = replace(new_state, power_actions_taken=state.power_actions_taken | {tile})
    return _apply_action_gain(new_state, faction, action.gain, cmd)


def _check_special_action_available(
    state: GameState, faction: str, tile: str, cmd: ParsedCommand
) -> None:
    """``command_action`` (834-838): ``$faction->{$action}`` must be truthy
    -- granted either unconditionally (``special_actions``, ACTE) or by a
    built stronghold whose ``build_gain`` mentions this tile id.
    """
    data = FACTIONS[faction]
    granted_by_sh = any(tile in gain for gain in data.buildings["SH"].build_gain)
    granted_always = tile in data.special_actions
    if not granted_by_sh and not granted_always:
        raise EngineError(
            f"{faction} has no {tile} space available", state=state, faction=faction, cmd=cmd
        )
    if granted_by_sh and not state.factions[faction].buildings.get("SH"):
        raise EngineError(
            f"{faction} must build a stronghold before using {tile}",
            state=state,
            faction=faction,
            cmd=cmd,
        )


def _handle_faction_special_action(
    state: GameState, faction: str, cmd: ParsedCommand, tile: str
) -> GameState:
    _check_special_action_available(state, faction, tile, cmd)

    blocks = tile not in _UNBLOCKED_SPECIAL_ACTIONS
    fs = state.factions[faction]
    if blocks and tile in fs.actions_used:
        raise EngineError(
            f"{faction} already used {tile} this round", state=state, faction=faction, cmd=cmd
        )

    spec = FACTION_SPECIAL_ACTIONS[tile]
    fs = _pay_simple(state, faction, fs, spec["cost"], cmd)
    if blocks:
        fs = replace(fs, actions_used=fs.actions_used | {tile})
    new_state = with_faction(state, faction, fs)
    return _apply_action_gain(new_state, faction, spec["gain"], cmd)


def _handle_bonus_or_favor_action(
    state: GameState,
    faction: str,
    cmd: ParsedCommand,
    tile: str,
    held: bool,
    gain: Mapping[str, int] | None,
) -> GameState:
    if gain is None:
        raise EngineError(f"{tile} has no special action", state=state, faction=faction, cmd=cmd)
    if not held:
        raise EngineError(f"{faction} does not hold {tile}", state=state, faction=faction, cmd=cmd)

    fs = state.factions[faction]
    if tile in fs.actions_used:
        raise EngineError(
            f"{faction} already used {tile} this round", state=state, faction=faction, cmd=cmd
        )
    fs = replace(fs, actions_used=fs.actions_used | {tile})
    new_state = with_faction(state, faction, fs)
    return _apply_action_gain(new_state, faction, gain, cmd)


def handle_action(state: GameState, faction: str, cmd: ParsedCommand) -> GameState:
    """``action ACTx``/``action BONx``/``action FAVx``: dispatch on
    ``cmd.tile`` (module docstring).
    """
    assert cmd.tile is not None
    tile = cmd.tile

    if tile in POWER_ACTIONS:
        return _handle_power_wheel_action(state, faction, cmd, tile)
    if tile in FACTION_SPECIAL_ACTIONS:
        return _handle_faction_special_action(state, faction, cmd, tile)
    if tile in ("BON1", "BON2"):
        bon = BONUS_TILES[tile]
        return _handle_bonus_or_favor_action(
            state, faction, cmd, tile, state.factions[faction].bonus == tile, bon.special_action
        )
    if tile == "FAV6":
        fav = FAVOR_TILES[tile]
        return _handle_bonus_or_favor_action(
            state, faction, cmd, tile, tile in state.factions[faction].favors, fav.special_action
        )

    raise EngineError(f"unknown action {tile!r}", state=state, faction=faction, cmd=cmd)


def handle_lose_marker(state: GameState, faction: str, cmd: ParsedCommand) -> GameState:
    """``-FREE_D``/``-FREE_TP``/``-FREE_TF``/``-BRIDGE``: pop the matching
    outstanding marker pending unconsumed -- overrides ``apply.py``'s
    no-op stub (module docstring, module-bottom import). ``cmd.reason`` is
    the same marker name ``_MARKER_GAIN_KIND`` maps to a pending kind.
    """
    assert cmd.reason is not None
    kind = _MARKER_GAIN_KIND.get(cmd.reason)
    if kind is None:
        raise EngineError(f"unknown marker {cmd.reason!r}", state=state, faction=faction, cmd=cmd)

    idx = _find_pending_index(state, faction, kind)
    if idx is None:
        raise EngineError(
            f"no outstanding {cmd.reason} marker for {faction}",
            state=state,
            faction=faction,
            cmd=cmd,
        )
    return pop_pending(state, idx)


register_handler("action", handle_action)
register_handler("lose_marker", handle_lose_marker)
