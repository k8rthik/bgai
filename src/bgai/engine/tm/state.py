"""Immutable game-state model: `GameState`/`FactionState` + tiny read helpers.

Pure construction only -- `GameState.initial`/`FactionState.initial` build
the pre-round-1 snapshot from a :class:`~bgai.engine.tm.setup.GameSetup`.
No move legality, no `apply`: that belongs to a later task. Field names and
shapes are frozen API per the task-3 brief; later tasks add fields via
migration rather than repurposing these.

Starting VP is 20 for every faction (deltas-oracle setup rows). Priest pool
starts at ``MAX_PRIESTS`` for every faction regardless of starting priests
in hand, per the brief's explicit ``FactionState.initial`` contract.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import Enum

from bgai.engine.tm.board import base_board
from bgai.engine.tm.cults import PRIEST_SLOT_STEPS
from bgai.engine.tm.factions_data import CULTS, FACTIONS, MAX_PRIESTS, FactionData
from bgai.engine.tm.power import Power
from bgai.engine.tm.setup import GameSetup
from bgai.engine.tm.tiles import FAVOR_POOL_COUNTS, TILE_OPTIONS, TOWN_POOL_COUNTS


class Phase(Enum):
    SETUP_DWELLINGS = "SETUP_DWELLINGS"
    SETUP_BONUS = "SETUP_BONUS"
    INCOME = "INCOME"
    ACTIONS = "ACTIONS"
    CLEANUP = "CLEANUP"
    FINISHED = "FINISHED"


@dataclass(frozen=True)
class HexState:
    color: str
    building: str | None
    owner: str | None


@dataclass(frozen=True)
class PendingDecision:
    """A forced sub-decision queued ahead of the turn-order successor.

    ``kind`` values used by later tasks: "leech", "gain_favor", "gain_town",
    "cult_choice", "convert_w_to_p", "bonus_choice", "spade_use", "bridge",
    "free_d", "free_tp", "free_tf". (Task 9's initial "halflings_spades"
    kind was retired -- a Halflings SH's spade grant is applied immediately
    by ``actions_build.py`` instead, see that module's docstring; kept out
    of this list so it doesn't get reused. "bridge"/"free_d"/"free_tp"/
    "free_tf" are Task 10's one-shot markers from ACT1/ACTE, ACTW, ACTS,
    and ACTN respectively -- see ``actions_power.py``'s module docstring.)
    """

    faction: str
    kind: str
    amount: int = 0
    source: str | None = None
    options: tuple[str, ...] = ()


@dataclass(frozen=True)
class FactionState:
    name: str
    coins: int
    workers: int
    priests: int
    priest_pool: int
    power: Power
    vp: int
    shipping: int
    dig_level: int
    teleport_level: int
    buildings: Mapping[str, frozenset[str]]
    favors: tuple[str, ...]
    bonus: str | None
    towns: tuple[str, ...]
    keys: int
    passed: bool
    actions_used: frozenset[str]
    cult_blocked: frozenset[str]
    bridges_built: int = 0
    """Count of bridges placed so far (``factions_data.BRIDGE_COUNT`` cap
    of 3); added by Task 8 -- a migration on top of the task-3 frozen
    shape, per this module's own docstring ("later tasks add fields via
    migration rather than repurposing these"). ``state.bridges`` is a
    flat, faction-agnostic set of hex-pairs, so nothing else on
    `GameState`/`FactionState` could answer "how many bridges has X built".
    """
    spades_available: int = 0
    """Transient spade balance (``$faction->{SPADE}`` in ``resources.pm``):
    incremented by ``dig N`` and by a Halflings SH's immediate 3-spade
    grant (``actions_build.py``), decremented by ``transform`` and by
    ``lose_spade``; added by Task 9 -- another migration on top of the
    task-3 frozen shape, per this module's docstring. ACT5/ACT6/BON1 spade
    grants (Task 10) feed the same balance -- ``actions_terraform.py``
    treats it as the single source of truth regardless of source.
    """
    extra_actions: int = 0
    """Chaos Magicians' ACTC "double turn" ticket -- ``$faction->{allowed_actions}``
    in ``resources.pm``/``acting.pm`` (``adjust_resource``'s ``GAIN_ACTION``
    branch, line ~289: ``$faction->{allowed_actions} += $delta``). ACTC's
    ``gain => { GAIN_ACTION => 2 }`` (Constants.pm ``%actions``) sets this to
    2; ``actions_power.py`` only sets the counter, it never consumes it --
    Task 11's turn-advancement flow owns spending it back down (one full
    turn per unit) before returning control to the next faction in
    ``turn_order``. Added by Task 10 -- another migration on top of the
    task-3 frozen shape, per this module's docstring.
    """
    teleported_hex: str | None = None
    """``$faction->{TELEPORT_TO}`` (``map.pm`` ``check_reachable`` lines
    244-250 / ``commands.pm`` ``command_transform`` line 629): the hex, if
    any, this faction has *this turn* already paid a
    ``connectivity.teleport_crossing`` fee (Dwarves' tunnel / Fakirs'
    carpet flight) to reach. A second command touching that *same* hex
    later in the *same* turn is free (Perl's own ``check_reachable``
    returns ``({}, {})`` when ``TELEPORT_TO eq $where``); a command
    touching a *different* hex the same turn is illegal in Perl ("Can't
    use tunnel / carpet flight multiple times in one turn") and never
    appears in this corpus, so this engine doesn't reject it, just doesn't
    treat it as already-paid. Cleared to ``None`` every time a faction
    starts a fresh full action (``start_full_move``'s own ``delete
    $faction->{TELEPORT_TO}``) -- ``round_flow.py``'s ``_advance_actions``/
    ``begin_actions`` own that reset (module docstring, task-14 fix).
    Corpus evidence for the *lack* of any longer-lived memory: game
    ``4pLeague_S10_D3L2_G6``, dwarves ``dig 1. transform A12`` at row 344
    pays the fee once; ``build A12`` -- on the very same, by-then-already-
    gray hex -- 3 turns later at row 350 (darklings/cultists/
    chaosmagicians all acted in between, so ``TELEPORT_TO`` was cleared
    and reset at least twice) pays the *same* fee again, in full. Added by
    Task 14 -- another migration on top of the task-3 frozen shape, per
    this module's docstring.
    """

    @classmethod
    def initial(cls, data: FactionData) -> FactionState:
        return cls(
            name=data.name,
            coins=data.coins,
            workers=data.workers,
            priests=data.priests,
            priest_pool=MAX_PRIESTS,
            power=Power(*data.power),
            vp=20,
            shipping=data.shipping.level,
            dig_level=data.dig.level,
            teleport_level=0,
            buildings={building: frozenset() for building in data.buildings},
            favors=(),
            bonus=None,
            towns=(),
            keys=0,
            passed=False,
            actions_used=frozenset(),
            cult_blocked=frozenset(),
            bridges_built=0,
            spades_available=0,
            extra_actions=0,
        )


def _initial_hexes() -> dict[str, HexState]:
    board = base_board()
    return {
        key: HexState(color=hex_.color, building=None, owner=None)
        for key, hex_ in board.hexes.items()
    }


def _initial_towns_pool(setup: GameSetup) -> dict[str, int]:
    return {
        town: count
        for town, count in TOWN_POOL_COUNTS.items()
        if TILE_OPTIONS.get(town) != "mini-expansion-1" or setup.options.mini_expansion_1
    }


@dataclass(frozen=True)
class GameState:
    setup: GameSetup
    round: int
    phase: Phase
    turn_order: tuple[str, ...]
    passed_order: tuple[str, ...]
    active_index: int
    pending: tuple[PendingDecision, ...]
    hexes: Mapping[str, HexState]
    bridges: frozenset[frozenset[str]]
    factions: Mapping[str, FactionState]
    cults: Mapping[str, Mapping[str, int]]
    cult_10: Mapping[str, str | None]
    priest_slots: Mapping[str, tuple[str | None, ...]]
    favors_pool: Mapping[str, int]
    towns_pool: Mapping[str, int]
    bonus_coins: Mapping[str, int]
    power_actions_taken: frozenset[str]
    founded_towns: Mapping[str, tuple[frozenset[str], ...]]

    @classmethod
    def initial(cls, setup: GameSetup) -> GameState:
        factions = {name: FactionState.initial(FACTIONS[name]) for name in setup.factions}
        cults = {name: dict(FACTIONS[name].cults) for name in setup.factions}
        empty_priest_slots = tuple(None for _ in PRIEST_SLOT_STEPS)
        return cls(
            setup=setup,
            round=0,
            phase=Phase.SETUP_DWELLINGS,
            turn_order=setup.factions,
            passed_order=(),
            # ``handle_pass`` (``actions_pass.py``) appends the real
            # per-faction pass order fresh over round 1's own ACTIONS
            # phase; ``end_of_round`` reads it once (under
            # ``variable_turn_order``) then resets it to `()` for the next
            # round -- seeding this with `setup.factions` (an earlier
            # revision's choice, no Perl citation) instead of empty meant
            # round 1's passed_order carried 4 *extra*, unwanted leading
            # entries into round 2's turn_order (task-13 report,
            # reference-game row 105: round 2's `turn_order` came out
            # length-8, "seat order + real round-1 pass order" concatenated,
            # so `turn_order[0]` was `engineers` instead of `nomads`, the
            # faction that actually passed first in round 1).
            active_index=0,
            pending=(),
            hexes=_initial_hexes(),
            bridges=frozenset(),
            factions=factions,
            cults=cults,
            cult_10={cult: None for cult in CULTS},
            priest_slots={cult: empty_priest_slots for cult in CULTS},
            favors_pool=dict(FAVOR_POOL_COUNTS),
            towns_pool=_initial_towns_pool(setup),
            bonus_coins={bon: 0 for bon in setup.bonus_tiles},
            power_actions_taken=frozenset(),
            founded_towns={name: () for name in setup.factions},
        )


def active_faction(state: GameState) -> str:
    """The faction taking the current full turn: ``turn_order[active_index]``.

    Ported from ``acting.pm``'s ``active_faction`` attribute (module
    docstring there: "Which faction is currently acting (a full action,
    not just an async decision on resources)") -- a plain slot updated
    only by explicit turn-advancement, deliberately independent of
    ``action_required`` (this engine's ``state.pending``). An earlier
    revision of this function returned ``pending[0].faction`` whenever
    ``pending`` was non-empty, on the theory that an outstanding decision
    (chiefly a leech offer) should gate every other faction's turn until
    answered. Replaying the reference game disproved that (task-13
    report, row 59): nomads' row-58 build queues leech offers for
    engineers and darklings, yet row 59 is mermaids taking an entirely
    ordinary ``upgrade`` turn with both offers still outstanding, and
    engineers doesn't answer its offer until row 60 -- *after* mermaids'
    turn. Perl's own dispatcher (``commands.pm`` ``command``,
    ``$assert_active_faction``) confirms this structurally: ``leech``/
    ``decline``/``+CULT`` route through the faction-only ``$assert_faction``
    check, never ``$assert_active_faction`` -- only genuine main-track
    verbs (build/upgrade/send/convert/burn/dig/bridge/connect/pass/action)
    are gated against ``active_faction`` at all, so a pending leech/
    cult-choice decision was never meant to block anyone's main-track
    turn, this faction's own included. ``apply.py``'s leech/decline/
    gain_cult exemptions already have their own independent
    ``state.pending``-membership checks (``_has_queued_pending_of_kind``)
    for exactly this reason -- they never depended on this function's
    former pending-preference behavior.
    """
    return state.turn_order[state.active_index]


def with_faction(state: GameState, name: str, fs: FactionState) -> GameState:
    """Return a new GameState with `name`'s FactionState replaced by `fs`."""
    new_factions = dict(state.factions)
    new_factions[name] = fs
    return replace(state, factions=new_factions)


def cult_string(state: GameState, faction: str) -> str:
    """Cult-track positions for `faction` as "F/W/E/A", e.g. "0/1/1/0"."""
    positions = state.cults[faction]
    return "/".join(str(positions[cult]) for cult in CULTS)
