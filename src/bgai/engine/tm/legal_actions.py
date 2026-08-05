"""ACTIONS-phase main-action enumeration -- the bulk of Task 15's
``legal_moves``, split out of ``legal.py`` per the task brief ("split
enumeration helpers by phase if not" under 400 lines).

Every helper here is a pure query: given ``state``/``faction``/``fs``, it
returns the list of currently-legal :class:`ParsedCommand` instances for
one action family (builds, upgrades, digs, ...), reusing the exact
affordability/reachability/adjacency primitives the real handlers
(``actions_build.py``/``actions_terraform.py``/``actions_power.py``/
``actions_pass.py``) already validate against -- so a generated command is
guaranteed to ``apply()`` cleanly (Step 2's generative-smoke contract)
as long as state doesn't change between the query and the ``apply()``
call.

**Marker pendings are not "decisions" -- they modify this enumeration.**
``free_d``/``free_tp``/``free_tf`` (ACTW/ACTS/ACTN) and ``bridge``
(ACT1/ACTE) are one-shot allowances, not a forced next-answer the way
``leech``/``gain_favor``/``gain_town``/``cult_choice``/``convert_w_to_p``
are (``legal.py``'s ``_BLOCKING_PENDING_KINDS`` docstring has the full
reasoning) -- a faction holding e.g. a live ``free_d`` marker can still
take any other ordinary ACTIONS-phase action; the marker only changes
*how* a matching build/upgrade/transform/bridge command is priced, and
``marker_decline_moves`` below always offers the corresponding
``lose_marker`` "decline the grant" option alongside it (real corpus
rows: ``action ACTW. -FREE_D`` etc., ``actions_power.py``'s module
docstring).

**Granularity choices** (documented once here, applies to every helper):
- ``dig``/``burn``: enumerated at *every* affordable exact amount
  (``dig 1`` .. ``dig N``, ``burn 1`` .. ``burn M``) since each is a
  single atomic ledger verb with a well-defined per-unit cost -- unlike
  ``convert``, there is no compounding of *different* per-unit costs to
  flatten, so enumerating the exact amount costs nothing extra and keeps
  containment a plain membership check instead of a decomposition.
- ``convert``: only the single-unit exchange (``n2=1``) for every
  affordable ``(res1, res2)`` rate this faction has -- a real ledger row
  requesting more than one unit (``convert 6C to 2VP``) is, by design,
  *not* separately enumerated; ``legal.py``'s module docstring documents
  the containment test's decomposition policy for this.
- ``transform``: enumerates every color reachable within the current
  ``spades_available`` budget (via ``hooks_for(faction)
  .spade_transform_cost``), not just the "closer to home" default a bare
  ``transform HEX`` (no ``to COLOR`` clause) would resolve to -- a
  superset that still contains whatever explicit color a real ledger row
  names, since ``handle_transform`` honors any requested color whose cost
  fits the budget (module docstring's per-helper citation below).
"""

from __future__ import annotations

from bgai.data.ledger_parser import ParsedCommand
from bgai.engine.tm.actions_build import _bridgable_pairs
from bgai.engine.tm.board import RIVER, base_board
from bgai.engine.tm.connectivity import reachable, teleport_crossing
from bgai.engine.tm.cults import PRIEST_SLOT_STEPS
from bgai.engine.tm.factions.hooks import hooks_for
from bgai.engine.tm.factions_data import (
    BASE_EXCHANGE_RATES,
    BRIDGE_COUNT,
    COLOR_WHEEL,
    CULTS,
    FACTION_SPECIAL_ACTIONS,
    FACTIONS,
)
from bgai.engine.tm.leech import has_leechable_neighbor
from bgai.engine.tm.legal_shared import affordable, cmd, own_pending_index, resource_amount
from bgai.engine.tm.state import FactionState, GameState
from bgai.engine.tm.tiles import BONUS_TILES, POWER_ACTIONS
from bgai.engine.tm.towns import river_town_candidate

# actions_build.py's private upgrade-target->source map, duplicated here
# (same no-cross-coupling rationale that module's own near-duplicate
# helpers document) rather than imported.
_UPGRADE_FROM: dict[str, str] = {"TP": "D", "TE": "TP", "SH": "TP", "SA": "TE"}


def _directly_adjacent_plain(fs: FactionState, hex_key: str) -> bool:
    """``actions_terraform.py``'s/``actions_build.py``'s identically-named
    helper (plain board adjacency only, bridges excluded -- ACTN's
    ``TF_NEED_HEX_ADJACENCY``), duplicated locally per those modules' own
    stated rationale.
    """
    own = frozenset().union(*fs.buildings.values()) if fs.buildings else frozenset()
    board = base_board()
    return any(hex_key in board.adjacent.get(b, frozenset()) for b in own)


def _can_afford_build(
    state: GameState, faction: str, fs: FactionState, hex_key: str, base_cost: dict[str, int]
) -> bool:
    """Whether ``fs`` can pay ``base_cost`` plus, if ``hex_key`` is only
    reachable via this faction's ``TeleportTrack``, that crossing's own
    fee (``connectivity.teleport_crossing`` -- ``handle_build``'s own
    teleport-fee branch, ``actions_build.py``).
    """
    total = dict(base_cost)
    if hex_key != fs.teleported_hex:
        tp_cost, _vp = teleport_crossing(state, faction, hex_key)
        for res, amt in tp_cost.items():
            total[res] = total.get(res, 0) + amt
    return affordable(fs, total)


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------


def build_moves(state: GameState, faction: str, fs: FactionState) -> list[ParsedCommand]:
    color = FACTIONS[faction].color
    d_track = FACTIONS[faction].buildings["D"]
    if len(fs.buildings["D"]) >= d_track.max_count:
        return []

    # ACTW's FREE_D marker: any empty, already-home-colored hex, no
    # reachability check, no cost (handle_build's free_d_index branch
    # skips both entirely -- actions_build.py's module docstring).
    if own_pending_index(state, faction, "free_d") is not None:
        return [
            cmd("build", loc=hex_key)
            for hex_key, hx in state.hexes.items()
            if hx.building is None and hx.color == color
        ]

    free_tf = own_pending_index(state, faction, "free_tf") is not None
    moves: list[ParsedCommand] = []
    for hex_key in reachable(state, faction):
        hx = state.hexes[hex_key]
        if hx.building is not None or hx.color == RIVER:
            continue
        if hx.color == color:
            if _can_afford_build(state, faction, fs, hex_key, d_track.cost):
                moves.append(cmd("build", loc=hex_key))
            continue
        # ACTN's FREE_TF marker folds into a build's own implicit
        # transform when the target is directly (plain-board) adjacent to
        # an existing building -- actions_build.py's handle_build docstring,
        # corpus pattern "action ACTN. build F2" (task-13 report row 174).
        # The marker waives the transform, not the dwelling cost
        # (handle_build still charges it -- arena fuzz finding,
        # test_legal_soundness).
        if (
            free_tf
            and _directly_adjacent_plain(fs, hex_key)
            and _can_afford_build(state, faction, fs, hex_key, d_track.cost)
        ):
            moves.append(cmd("build", loc=hex_key))
            continue
        tf_cost = hooks_for(faction).spade_transform_cost(state, faction, hx.color, color)
        if tf_cost <= fs.spades_available and _can_afford_build(
            state, faction, fs, hex_key, d_track.cost
        ):
            moves.append(cmd("build", loc=hex_key))
    return moves


# --------------------------------------------------------------------------
# upgrade
# --------------------------------------------------------------------------


def upgrade_moves(state: GameState, faction: str, fs: FactionState) -> list[ParsedCommand]:
    moves: list[ParsedCommand] = []
    for new_type, old_type in _UPGRADE_FROM.items():
        track = FACTIONS[faction].buildings[new_type]
        if len(fs.buildings[new_type]) >= track.max_count:
            continue
        free_tp = new_type == "TP" and own_pending_index(state, faction, "free_tp") is not None
        for hex_key in fs.buildings[old_type]:
            if free_tp:
                moves.append(cmd("upgrade", loc=hex_key, building=new_type))
                continue
            cost = dict(track.cost)
            if new_type == "TP" and not has_leechable_neighbor(state, faction, hex_key):
                cost["C"] = cost.get("C", 0) * 2
            if affordable(fs, cost):
                moves.append(cmd("upgrade", loc=hex_key, building=new_type))
    return moves


# --------------------------------------------------------------------------
# dig / transform
# --------------------------------------------------------------------------


def dig_moves(state: GameState, faction: str, fs: FactionState) -> list[ParsedCommand]:
    dig_track = FACTIONS[faction].dig
    level = fs.dig_level
    if level >= len(dig_track.cost):
        return []
    unit_cost = dig_track.cost[level]
    limits = [resource_amount(fs, res) // amt for res, amt in unit_cost.items() if amt > 0]
    max_n = min(limits) if limits else 0
    return [cmd("dig", n1=n) for n in range(1, max_n + 1)]


def transform_moves(state: GameState, faction: str, fs: FactionState) -> list[ParsedCommand]:
    color = FACTIONS[faction].color

    if own_pending_index(state, faction, "free_tf") is not None:
        board = base_board()
        own = frozenset().union(*fs.buildings.values()) if fs.buildings else frozenset()
        candidates: set[str] = set()
        for b in own:
            candidates |= board.adjacent.get(b, frozenset())
        return [
            cmd("transform", loc=hex_key, color=color)
            for hex_key in candidates
            if state.hexes[hex_key].building is None
            and state.hexes[hex_key].color not in (RIVER, color)
        ]

    if fs.spades_available <= 0:
        return []
    moves: list[ParsedCommand] = []
    for hex_key in reachable(state, faction):
        hx = state.hexes[hex_key]
        if hx.building is not None or hx.color == RIVER:
            continue
        if hex_key != fs.teleported_hex:
            tp_cost, _vp = teleport_crossing(state, faction, hex_key)
            if tp_cost and not affordable(fs, tp_cost):
                continue
        for target in COLOR_WHEEL:
            if target == hx.color:
                continue
            # handle_transform rejects any requested color the faction's
            # target hook overrides (Giants: always home terrain), so only
            # offer targets the hook leaves unchanged -- identity for every
            # faction but Giants (arena fuzz finding, test_legal_soundness).
            if hooks_for(faction).spade_transform_target(state, faction, hex_key, target) != target:
                continue
            cost = hooks_for(faction).spade_transform_cost(state, faction, hx.color, target)
            if 0 < cost <= fs.spades_available:
                moves.append(cmd("transform", loc=hex_key, color=target))
    return moves


# --------------------------------------------------------------------------
# power wheel / faction special / BON1-BON2/FAV6 actions
# --------------------------------------------------------------------------


def power_wheel_moves(state: GameState, faction: str, fs: FactionState) -> list[ParsedCommand]:
    return [
        cmd("action", tile=tile)
        for tile, action in POWER_ACTIONS.items()
        if tile not in state.power_actions_taken and fs.power.usable >= action.cost_power
    ]


def faction_special_moves(state: GameState, faction: str, fs: FactionState) -> list[ParsedCommand]:
    data = FACTIONS[faction]
    moves: list[ParsedCommand] = []
    for tile, spec in FACTION_SPECIAL_ACTIONS.items():
        granted_by_sh = any(tile in gain for gain in data.buildings["SH"].build_gain)
        granted_always = tile in data.special_actions
        if not granted_by_sh and not granted_always:
            continue
        if granted_by_sh and not fs.buildings.get("SH"):
            continue
        if tile != "ACTE" and tile in fs.actions_used:  # ACTE never blocks (dont_block)
            continue
        if not affordable(fs, spec["cost"]):
            continue
        moves.append(cmd("action", tile=tile))
    return moves


def bonus_favor_action_moves(
    state: GameState, faction: str, fs: FactionState
) -> list[ParsedCommand]:
    moves: list[ParsedCommand] = []
    if (
        fs.bonus in ("BON1", "BON2")
        and fs.bonus not in fs.actions_used
        and BONUS_TILES[fs.bonus].special_action is not None
    ):
        moves.append(cmd("action", tile=fs.bonus))
    if "FAV6" in fs.favors and "FAV6" not in fs.actions_used:
        moves.append(cmd("action", tile="FAV6"))
    return moves


# --------------------------------------------------------------------------
# send / advance
# --------------------------------------------------------------------------


def send_moves(state: GameState, faction: str, fs: FactionState) -> list[ParsedCommand]:
    if fs.priests < 1:
        return []
    moves: list[ParsedCommand] = []
    for cult in CULTS:
        slots = state.priest_slots[cult]
        open_steps = sorted({PRIEST_SLOT_STEPS[i] for i, occ in enumerate(slots) if occ is None})
        if open_steps:
            if fs.priest_pool >= 1:
                moves.append(cmd("send", cult=cult))
                moves.extend(cmd("send", cult=cult, n1=step) for step in open_steps)
        else:
            # Every slot on this track is full: the priest is still spent
            # (track +1 step), no priest_pool commitment (handle_send).
            moves.append(cmd("send", cult=cult))
    return moves


def advance_moves(state: GameState, faction: str, fs: FactionState) -> list[ParsedCommand]:
    moves: list[ParsedCommand] = []
    data = FACTIONS[faction]
    ship = data.shipping
    if (
        ship.advance_cost is not None
        and fs.shipping < ship.max_level
        and affordable(fs, ship.advance_cost)
    ):
        moves.append(cmd("advance", reason="ship"))
    dig = data.dig
    if (
        dig.advance_cost is not None
        and fs.dig_level < dig.max_level
        and affordable(fs, dig.advance_cost)
    ):
        moves.append(cmd("advance", reason="dig"))
    return moves


# --------------------------------------------------------------------------
# bridge / marker declines / connect
# --------------------------------------------------------------------------


def bridge_moves(state: GameState, faction: str, fs: FactionState) -> list[ParsedCommand]:
    if fs.bridges_built >= BRIDGE_COUNT:
        return []
    moves: list[ParsedCommand] = []
    for pair in _bridgable_pairs():
        if pair in state.bridges:
            continue
        a, b = tuple(pair)
        if state.hexes[a].owner == faction or state.hexes[b].owner == faction:
            moves.append(cmd("bridge", loc=a, loc2=b))
    return moves


_MARKER_REASON_FOR_KIND: dict[str, str] = {
    "free_d": "FREE_D",
    "free_tp": "FREE_TP",
    "free_tf": "FREE_TF",
    "bridge": "BRIDGE",
}


def marker_decline_moves(state: GameState, faction: str) -> list[ParsedCommand]:
    return [
        cmd("lose_marker", reason=reason)
        for kind, reason in _MARKER_REASON_FOR_KIND.items()
        if own_pending_index(state, faction, kind) is not None
    ]


def connect_moves(state: GameState, faction: str, fs: FactionState) -> list[ParsedCommand]:
    if faction != "mermaids":
        return []
    board = base_board()
    return [
        cmd("connect", loc=hex_key)
        for hex_key, hx in board.hexes.items()
        if hx.color == RIVER and river_town_candidate(state, faction, hex_key) is not None
    ]


# --------------------------------------------------------------------------
# convert / burn / pass / no-op
# --------------------------------------------------------------------------

_EXCHANGE_PAIRS: tuple[tuple[str, str], ...] = (
    ("PW", "C"),
    ("PW", "W"),
    ("PW", "P"),
    ("W", "C"),
    ("P", "C"),
    ("P", "W"),
    ("C", "VP"),
)


def _exchange_rate(faction: str, res1: str, res2: str) -> int | None:
    """``apply.py``'s private ``_exchange_rate``, duplicated (not
    exported API): faction overrides win over ``BASE_EXCHANGE_RATES``.
    """
    overrides = FACTIONS[faction].exchange_rate_overrides
    if res1 in overrides and res2 in overrides[res1]:
        return overrides[res1][res2]
    return BASE_EXCHANGE_RATES.get(res1, {}).get(res2)


def convert_moves(state: GameState, faction: str, fs: FactionState) -> list[ParsedCommand]:
    pairs = set(_EXCHANGE_PAIRS)
    overrides = FACTIONS[faction].exchange_rate_overrides
    for res1, targets in overrides.items():
        pairs.update((res1, res2) for res2 in targets)
    moves: list[ParsedCommand] = []
    for res1, res2 in pairs:
        rate = _exchange_rate(faction, res1, res2)
        if not rate or resource_amount(fs, res1) < rate:
            continue
        moves.append(cmd("convert", res1=res1, n1=rate, res2=res2, n2=1))
    return moves


def burn_moves(fs: FactionState) -> list[ParsedCommand]:
    max_n = fs.power.bowl2 // 2
    return [cmd("burn", n1=n) for n in range(1, max_n + 1)]


def pass_moves(state: GameState, faction: str, fs: FactionState) -> list[ParsedCommand]:
    held = {
        o.bonus for name, o in state.factions.items() if name != faction and o.bonus is not None
    }
    moves = [
        cmd("pass", tile=tile)
        for tile in state.setup.bonus_tiles
        if tile != fs.bonus and tile not in held
    ]
    if state.round >= 6:
        moves.append(cmd("pass"))
    return moves


def exempt_noop_moves() -> list[ParsedCommand]:
    """``wait``: a zero-effect verb (``apply.py``'s ``handle_noop``) that
    is also unconditionally order-exempt (``_ORDER_EXEMPT_VERBS``) -- legal
    for any live faction regardless of phase or active-faction status.
    ``done`` is deliberately **not** included here: unlike ``wait``, it is
    *not* in ``_ORDER_EXEMPT_VERBS``, so it only applies when ``faction``
    is genuinely allowed to act (``active_noop_moves`` below). ``resign``
    is excluded from both -- this engine models it as a no-op with no
    lasting effect (``handle_noop`` never sets ``FactionState.dropped`` or
    anything else), which would misrepresent it as a repeatable,
    consequence-free action to an MCTS/agent consumer.
    """
    return [cmd("wait")]


def active_noop_moves() -> list[ParsedCommand]:
    """``done``: legal only when ``faction`` is otherwise allowed to act
    (``active_faction(state)``, or any faction during ``Phase.INCOME``/
    ``Phase.CLEANUP`` where every verb is phase-exempt) -- ``legal.py``
    only calls this from those contexts.
    """
    return [cmd("done")]


def actions_phase_moves(state: GameState, faction: str) -> tuple[ParsedCommand, ...]:
    """Every legal main-track ACTIONS-phase command for ``faction`` (no
    blocking pending outstanding -- ``legal.py`` only calls this once
    that's been ruled out).

    ``convert``/``wait``/``done`` are deliberately **not** included here:
    both are order-exempt in ``apply.py`` regardless of phase or
    active-faction status (``_ORDER_EXEMPT_VERBS``), so ``legal.py``
    computes them once, unconditionally, as a baseline every
    ``legal_moves_for`` call unions in -- see that module's docstring.
    Folding them in here too would just duplicate entries whenever
    ``faction`` also happens to be ``active_faction(state)``.
    """
    fs = state.factions[faction]
    moves: list[ParsedCommand] = []
    moves += build_moves(state, faction, fs)
    moves += upgrade_moves(state, faction, fs)
    moves += dig_moves(state, faction, fs)
    moves += transform_moves(state, faction, fs)
    moves += power_wheel_moves(state, faction, fs)
    moves += faction_special_moves(state, faction, fs)
    moves += bonus_favor_action_moves(state, faction, fs)
    moves += send_moves(state, faction, fs)
    moves += advance_moves(state, faction, fs)
    if own_pending_index(state, faction, "bridge") is not None:
        moves += bridge_moves(state, faction, fs)
    moves += marker_decline_moves(state, faction)
    moves += connect_moves(state, faction, fs)
    moves += burn_moves(fs)
    moves += pass_moves(state, faction, fs)
    return tuple(moves)
