"""``build``/``upgrade``/``bridge``/``gain_favor``/``gain_town`` handlers.

Ported from the reference implementation (jsnell/terra-mystica, MIT),
``src/commands.pm``:

- ``command_build`` (148-256): unknown/occupied hex (156-159); a live
  ``FREE_D`` marker (161-164) makes the build free and skips the
  reachability check (Task 10, ``actions_power.py``'s Witches'-Ride
  finding: ``build_color_ok``, ``map.pm`` 657-664, is **never** bypassed
  under ``FREE_D`` -- the target must still already be the faction's home
  color there; what ``FREE_D``/``TELEPORT_NO_TF`` remove is cost and the
  ``check_reachable`` call, 207-227, only ever reached from the
  non-``TELEPORT_NO_TF`` branch). Ported here as a queued
  ``PendingDecision(kind="free_d")`` (pushed by ``actions_power.py``'s
  ACTW handler), consumed by ``handle_build`` below.
  For an *ordinary* (non-``FREE_D``) build, though, ``tf_needed`` (171)
  does **not** hard-error on a wrong-colored hex -- it internally dispatches
  ``command $faction_name, "transform $where to $color"`` (213-219) first,
  paying ``spades_available`` to recolor the target to home color in the
  same turn, *before* placing the dwelling, with no separate ``transform``
  ledger row of its own (task-13 report, reference-game row 58: "burn 6.
  action ACT6. transform G2. build F5" -- the explicit ``transform`` names
  a *different* hex prepped for a later turn; F5's own color fix is
  implicit, paid from the 1 spade ACT6 left banked after G2's). This
  engine ports that by folding a same-shaped auto-transform directly into
  ``handle_build`` (reusing ``hooks_for(faction).spade_transform_cost`` for
  Giants' flat-2 override, same as ``handle_transform``) rather than
  literally recursing into ``handle_transform``, since ``handle_build``
  always names an *explicit* target color (home), never the "closest to
  home" default-target selection ``handle_transform`` needs for a bare
  ``transform HEX`` row.
  ``pay``/``gain`` only outside round 0 (224-227, ``$free = ($game{round}
  == 0)`` at 152); ``note_leech`` (239, this module's ``leech.queue_leech``)
  fires **unconditionally**, ``FREE_D`` build included (Task 10 cross-check
  against the corpus's ``action ACTW. build X`` rows: always followed by
  ordinary opponent ``Leech`` rows when adjacent); ``advance_track`` for D
  count/cost (241); ``detect_towns_from`` (252, ``_maybe_queue_town``).
  Score-tile BUILD VP (244-245, ``tiles.scored_vp``) is applied for D,
  guarded by ``state.round >= 1`` (Task 13 fix, matching the same
  ``if ($game{round})`` guard) -- alongside it, favor-tile BUILD VP
  (``maybe_score_favor_tile``, called just before it at 244 -- see
  ``_favor_tile_vp``/``handle_gain_favor``'s own docstring for the
  passive-not-snapshot correction this needed).
- ``command_upgrade`` (258-311): wrong-color is always a hard error
  (266-267, no ``FREE_D``-style bypass for upgrades). ``note_leech`` fires
  *before* cost (280) for the D->TP neighbour-cost rule (282-293): a live
  ``FREE_TP`` marker (Swarmlings ACTS, Task 10 -- queued as
  ``PendingDecision(kind="free_tp")`` by ``actions_power.py``, consumed by
  ``handle_upgrade`` below) makes it free, matching ``command_upgrade``'s
  own ``$faction->{FREE_TP}`` branch (284-287) -- but the leech offers
  were already computed *before* this branch runs, so a free ACTS upgrade
  still queues ordinary leech (cross-checked against the corpus's
  ``action ACTS. Upgrade X to TP`` rows); otherwise, **only when
  ``%this_leech`` came back empty** (no adjacent
  opponent building of a different color) does this block manually
  pre-pay the ``C`` portion of the advance cost a *second* time --
  ``advance_track`` (301) pays the *full* ``advance_cost`` (W and C)
  unconditionally whenever not free. Net: isolated D->TP costs ``{W, 2C}``
  (C doubled), adjacent-to-an-opponent D->TP costs the plain ``{W, C}``
  from ``factions_data.py`` (already the discounted price). Real-rules
  framing: Trading Houses cost double coins unless built next to another
  faction's structure. D->TP-only -- TE/SH/SA always pay their listed
  ``advance_cost`` in full, no adjacency check. ``advance_gain``
  (resources.pm 240-243) is ``_apply_build_gain``; ``detect_towns_from``
  (308) is ``_maybe_queue_town``.
- ``command_bridge`` (706-748): requires a live ``BRIDGE`` marker (709-710
  -- Task 9/10 pushes a ``PendingDecision(kind="bridge")`` for ACT1/ACTE;
  this handler only *consumes* one), requires at least one endpoint to
  already carry a building of the acting faction's color (713-716,
  unconditional here since ``loose-bridge-adjacency`` isn't modeled in
  ``GameOptions``), the ``BRIDGE_COUNT`` cap of 3 per faction (724-725,
  tracked on the new ``FactionState.bridges_built`` field -- see that
  field's docstring), and the ``bridgable`` geometry from ``map.pm``
  ``setup_valid_bridges`` (164-203) -- ``_bridgable_pairs`` ports that
  algorithm (see its own docstring) and was cross-checked against 29 real
  ``bridge X:Y`` ledger rows from the crawled corpus (zero mismatches; 29
  is also the map's well-known total legal-bridge count).
  ``detect_towns_from`` fires per endpoint (746-747); ``_maybe_queue_town``
  scans every faction cluster, so one call after placing is equivalent.

``gain_favor``/``gain_town`` port ``resources.pm`` ``adjust_resource``'s
``FAV``/``TW`` branches (76-92: "taking a tile not allowed" unless a
``GAIN_FAVOR``/``GAIN_TW`` counter is live -- the ``PendingDecision``
amount here, metered down exactly like Task 7's ``convert_w_to_p``) plus
the cult-track advance and building-count-scaled VP that ``towns.py``'s
docstring leaves to a later task (``FavorTile.vp`` for FAV10/FAV11,
``.cult``/``.steps`` for the track advance). Recurring favor **income**
(FAV7/8/9) is *not* applied here -- ``income.py`` reads it live off
``FactionState.favors`` every income phase; applying it here would
double-count.

Town-candidate filtering (``_plain_town_candidates``): ``towns.new_towns``
already returns *only* plain clusters for every non-Mermaids faction (its
``_mermaid_candidates`` short-circuits on ``faction != "mermaids"``). For
Mermaids, ``new_towns`` unconditionally also includes river-join
candidates (per that module's docstring: a river-joined town is only
founded via `connect`, Task 11). ``towns.py`` exports no "plain only"
split, so this module re-derives that subset from the exported
``connectivity.clusters`` (no ``river_skip``) plus ``towns.py``'s own
qualification rule (power via ``building_power_value`` >= threshold minus
stacked FAV5, count with SA doubled >= 4, no overlap with a founded
town). One ``gain_town`` pending is pushed per candidate found *at push
time*; ``handle_gain_town`` re-derives candidates fresh at resolution
time and founds the lexicographically-first one, so a just-founded town's
hexes are excluded before the next pending resolves even though the
original detection pass is stale by then.
"""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache

from bgai.data.ledger_parser import ParsedCommand
from bgai.engine.tm import leech
from bgai.engine.tm.apply import EngineError, pop_pending, push_pending, register_handler
from bgai.engine.tm.board import RIVER, base_board
from bgai.engine.tm.connectivity import clusters, reachable, teleport_crossing
from bgai.engine.tm.cults import advance
from bgai.engine.tm.factions.hooks import hooks_for
from bgai.engine.tm.factions_data import BRIDGE_COUNT, FACTIONS, TOWN_SIZE
from bgai.engine.tm.state import FactionState, GameState, PendingDecision, Phase, with_faction
from bgai.engine.tm.tiles import FAVOR_TILES, TOWN_TILES, scored_vp
from bgai.engine.tm.towns import (
    apply_town_tile,
    building_power_value,
    new_towns,
    record_founded_town,
)

_UPGRADE_FROM: dict[str, str] = {"TP": "D", "TE": "TP", "SH": "TP", "SA": "TE"}
_COST_RES: dict[str, str] = {"W": "workers", "C": "coins"}
_FAV5_TOWN_SIZE_DELTA = FAVOR_TILES["FAV5"].passive.get("TOWN_SIZE", 0)
_TOWN_CULT_GAIN_KEYS = ("FIRE", "WATER", "EARTH", "AIR")


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------


def _pay(
    state: GameState, faction: str, fs: FactionState, cost: dict[str, int], cmd: ParsedCommand
) -> FactionState:
    for res, amount in cost.items():
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


def _find_pending(state: GameState, faction: str, kind: str, cmd: ParsedCommand) -> int:
    for i, p in enumerate(state.pending):
        if p.faction == faction and p.kind == kind:
            return i
    raise EngineError(
        f"no {kind!r} pending queued for {faction}", state=state, faction=faction, cmd=cmd
    )


def _find_pending_optional(state: GameState, faction: str, kind: str) -> int | None:
    """Like ``_find_pending``, but ``None`` (not ``EngineError``) when no
    matching pending is queued -- used for markers (``free_d``/``free_tp``)
    whose absence is the *normal* case, not a caller error.
    """
    for i, p in enumerate(state.pending):
        if p.faction == faction and p.kind == kind:
            return i
    return None


def _consume_amount(
    pending: tuple[PendingDecision, ...], index: int, used: int
) -> tuple[PendingDecision, ...]:
    entry = pending[index]
    remaining = entry.amount - used
    if remaining > 0:
        return pending[:index] + (replace(entry, amount=remaining),) + pending[index + 1 :]
    return pending[:index] + pending[index + 1 :]


def _apply_spade_gain_bonus(
    state: GameState, faction: str, fs: FactionState, spades: int
) -> FactionState:
    """Fold ``hooks_for(faction).extra_dig_gain`` into ``fs``, scaled by
    ``spades`` just gained. A near-duplicate of ``actions_terraform.py``'s
    identically-named-in-spirit helper (``_apply_extra_dig_gain``) -- kept
    as its own small copy here rather than an import, to avoid a
    build<->terraform module coupling neither otherwise needs (see that
    module's docstring: Perl's ``adjust_resource`` gain-mode loop,
    resources.pm 313-392, fires this for *any* positive ``SPADE`` delta,
    not just ``dig``'s -- a Halflings SH's 3-spade ``build_gain`` grant
    goes through the exact same mechanism in real Perl).
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


def _directly_adjacent_to_own_building(state: GameState, faction: str, hex_key: str) -> bool:
    """``actions_terraform.py``'s identically-named helper, duplicated
    locally (same no-cross-coupling rationale that module documents for
    its own near-duplicate helpers): ``hex_key`` must be directly adjacent
    to one of ``faction``'s own building hexes via **plain board
    adjacency only, bridges excluded** (``map.pm`` 641-651 -- see that
    module's docstring for the full citation). Needed here because a live
    ``free_tf`` marker (Nomads' ACTN Sandstorm) can make an ordinary
    ``build``'s own implicit transform free too, not just a bare
    ``transform`` row (task-13 report, reference-game row 174: "action
    ACTN. build F2").
    """
    fs = state.factions[faction]
    own = frozenset().union(*fs.buildings.values()) if fs.buildings else frozenset()
    board = base_board()
    return any(hex_key in board.adjacent.get(b, frozenset()) for b in own)


def _advance_shipping(faction: str, fs: FactionState, units: int) -> FactionState:
    """Port of ``resources.pm``'s ``adjust_resource`` ``GAIN_(TELEPORT|SHIP)``
    branch (250-262 area) for the ``ship`` track specifically: for each
    unit, so long as the track isn't already maxed, grant
    ``ShippingTrack.advance_vp[current level]`` VP (indexed by the level
    *before* the bump -- the same convention ``actions_pass.py``'s
    ``handle_advance`` already uses for a player-invoked ``advance ship``
    row) and bump the level by one. Shared by ``_apply_build_gain``'s
    ``GAIN_SHIP`` case below (a build_gain SH grant, e.g. Mermaids') and
    ``handle_gain_town``'s TW7 wiring -- both are just a different Perl
    call site feeding the exact same ``adjust_resource(faction,
    'GAIN_SHIP', N)`` branch. Missing this was a real bug, not a harmless
    deferral: Nomads' own ``advance_vp`` table (2/3/4 VP per level) is
    nonzero, so silently dropping it produces a wrong VP total (task-13
    report, reference-game row 210: nomads' TW7 grant bumps their
    shipping level 0->1 and should score the 2VP -- no, 3VP, since their
    level was already 1 by then from an earlier "advance ship" row --
    ``advance_vp[1]`` for the 1->2 step).
    """
    track = FACTIONS[faction].shipping
    for _ in range(units):
        level = fs.shipping
        if track.advance_cost is None or level >= track.max_level:
            break
        fs = replace(fs, shipping=level + 1, vp=fs.vp + track.advance_vp[level])
    return fs


def _favor_tile_vp(fs: FactionState, type_: str) -> int:
    """Port of ``scoring.pm``'s ``maybe_score_favor_tile`` (lines 32-47):
    sum, over every favor tile ``fs`` currently holds, that tile's
    ``FavorTile.vp[type_]`` -- 0 for every favor but FAV10 (keys ``TP``)
    and FAV11 (keys ``D``), whose ``.vp`` dict is otherwise empty (see
    ``handle_gain_favor``'s docstring for why this is a passive per-build
    bonus, not a one-time grant-time snapshot).
    """
    return sum(FAVOR_TILES[tile].vp.get(type_, 0) for tile in fs.favors)


def _current_score_tile_vp(state: GameState, type_: str, mode: str) -> int:
    """``tiles.scored_vp`` against this round's tile, or 0 during setup
    (``state.round == 0`` -- ``command_build``'s own ``if ($game{round})``
    guard, ``commands.pm`` 244-245; ``command_upgrade``/the generic
    resource-gain loop never fire during setup at all, so this guard is
    only load-bearing for ``handle_build``'s D case).
    """
    if state.round < 1:
        return 0
    tile = state.setup.score_tiles[state.round - 1]
    return scored_vp(tile, type_, mode)


def _apply_build_gain(state: GameState, faction: str, gain: dict[str, int]) -> GameState:
    """Fold one ``BuildingTrack.build_gain[level]`` dict into state: PW/VP
    apply immediately; SPADE (Halflings' SH) is *also* immediate -- straight
    onto ``fs.spades_available``, plus whatever
    ``hooks_for(faction).extra_dig_gain`` fires for that many spades (the
    same "gain the instant it's granted" timing ``resources.pm``'s
    ``adjust_resource`` uses for every positive ``SPADE`` delta, dig or
    build-gain alike -- see ``actions_terraform.py``'s module docstring
    and this fix's report for why an earlier revision's
    ``halflings_spades``-pending design could silently drop the grant when
    no ``transform``/``lose_spade`` ever followed the SH build).
    GAIN_FAVOR/CONVERT_W_TO_P (Darklings) become metered pendings;
    GAIN_SHIP/GAIN_TELEPORT bump the track directly (no companion ledger
    row observed in the corpus for either -- see task-8 report); ACTx is a
    documented no-op (Task 10 derives special-action availability from
    ``fs.buildings["SH"]``).
    """
    fs = state.factions[faction]
    pendings: list[PendingDecision] = []
    for key, amount in gain.items():
        if not amount:
            continue
        if key == "PW":
            fs = replace(fs, power=fs.power.gain(amount))
        elif key == "VP":
            fs = replace(fs, vp=fs.vp + amount)
        elif key == "GAIN_FAVOR":
            pendings.append(PendingDecision(faction=faction, kind="gain_favor", amount=amount))
        elif key == "SPADE":
            fs = replace(fs, spades_available=fs.spades_available + amount)
            fs = _apply_spade_gain_bonus(state, faction, fs, amount)
            fs = replace(fs, vp=fs.vp + amount * _current_score_tile_vp(state, "SPADE", "gain"))
        elif key == "CONVERT_W_TO_P":
            pendings.append(PendingDecision(faction=faction, kind="convert_w_to_p", amount=amount))
        elif key == "GAIN_SHIP":
            fs = _advance_shipping(faction, fs, amount)
        elif key == "GAIN_TELEPORT":
            fs = replace(fs, teleport_level=fs.teleport_level + amount)
        elif key.startswith("ACT"):
            continue
        else:
            raise ValueError(f"unhandled build_gain key {key!r} for {faction}")
    new_state = with_faction(state, faction, fs)
    return push_pending(new_state, *pendings) if pendings else new_state


def _plain_town_candidates(state: GameState, faction: str) -> tuple[frozenset[str], ...]:
    if faction != "mermaids":
        return new_towns(state, faction)

    if not any(count > 0 for count in state.towns_pool.values()):
        return ()
    fs = state.factions[faction]
    threshold = TOWN_SIZE + _FAV5_TOWN_SIZE_DELTA * fs.favors.count("FAV5")
    founded = state.founded_towns.get(faction, ())

    result = []
    for cluster in clusters(state, faction):
        power = 0
        count = 0
        for hex_key in cluster:
            building = state.hexes[hex_key].building
            assert building is not None
            power += building_power_value(building)
            count += 1
            if building == "SA":
                count += 1
        if power >= threshold and count >= 4 and not any(cluster & prior for prior in founded):
            result.append(cluster)
    return tuple(result)


def _cluster_key(cluster: frozenset[str]) -> str:
    """Canonical, order-independent string encoding of a hex cluster, used
    as a ``gain_town`` pending's ``source`` so ``handle_gain_town`` can
    recover exactly which cluster a pending refers to without having to
    re-derive "the" qualifying cluster at resolution time (see
    ``_maybe_queue_town``'s docstring for why that recomputation was
    unsound).
    """
    return ",".join(sorted(cluster))


def _maybe_queue_town(state: GameState, faction: str) -> GameState:
    """Detect newly-qualifying town clusters and queue one ``gain_town``
    pending each -- **recording** each cluster into ``founded_towns`` at
    detection time, not at tile-choice time.

    Ports ``towns.pm`` ``detect_towns_from`` marking ``$map{$where}{town}``
    the moment a cluster qualifies (lines 77-81), *before* a tile is even
    picked. Recording later (originally: only in ``handle_gain_town``,
    once ``apply_town_tile`` ran) let a second board change between
    detection and tile-choice re-detect the *same still-unresolved*
    cluster (``founded_towns`` hadn't been updated yet) and enqueue a
    second ``gain_town`` pending for one physical town -- fixed per code
    review. Each candidate returned by ``_plain_town_candidates`` in one
    call is a distinct connected component (component clusters are always
    disjoint), so founding all of them here in one pass cannot itself
    double-count.
    """
    candidates = _plain_town_candidates(state, faction)
    for cluster in candidates:
        state = record_founded_town(state, faction, cluster)
        pending = PendingDecision(
            faction=faction, kind="gain_town", amount=1, source=_cluster_key(cluster)
        )
        state = push_pending(state, pending)
    return state


@lru_cache(maxsize=1)
def _bridgable_pairs() -> frozenset[frozenset[str]]:
    """Legal bridge spans: ``map.pm`` ``setup_valid_bridges`` (lines 164-203).

    Two land hexes are bridgable if they sit across a single-river-hex gap
    in one of three geometric layouts (same column two rows apart with a
    river/off-board pair between; or one of the two diagonal one-row
    layouts with a river directly between), *and* they share at least one
    common river-hex neighbor -- which is exactly what ``map.pm``'s
    ``range{1}{other} == 1`` check means under ``setup_hex_ranges``'s
    river-only BFS (a hex's river-distance-1 set is precisely "any hex
    adjacent to a river hex that is itself adjacent to the source").
    Cross-checked against 29 distinct real ``bridge X:Y`` ledger rows
    pulled from the crawled corpus: zero mismatches (see task-8 report).
    """
    board = base_board()
    by_coord = {(h.row, h.col): h.key for h in board.hexes.values()}

    def get(row: int, col: int) -> str | None:
        return by_coord.get((row, col))

    def is_river(key: str | None) -> bool:
        return key is not None and board.hexes[key].color == RIVER

    pairs: set[frozenset[str]] = set()

    def record(this_key: str, other_key: str | None) -> None:
        if other_key is None or board.hexes[other_key].color == RIVER:
            return
        this_rivers = {n for n in board.adjacent[this_key] if board.hexes[n].color == RIVER}
        other_rivers = {
            n for n in board.adjacent.get(other_key, frozenset()) if board.hexes[n].color == RIVER
        }
        if this_rivers & other_rivers:
            pairs.add(frozenset({this_key, other_key}))

    for h in board.hexes.values():
        if h.color == RIVER:
            continue
        row, col = h.row, h.col
        offset_col = col - 1 if row % 2 == 0 else col

        sw, se = get(row + 1, offset_col), get(row + 1, offset_col + 1)
        if (
            (is_river(sw) and is_river(se))
            or (is_river(sw) and se is None)
            or (is_river(se) and sw is None)
        ):
            record(h.key, get(row + 2, col))

        if is_river(get(row, col - 1)) and is_river(get(row + 1, offset_col)):
            record(h.key, get(row + 1, offset_col - 1))
        if is_river(get(row, col + 1)) and is_river(get(row + 1, offset_col + 1)):
            record(h.key, get(row + 1, offset_col + 2))

    return frozenset(pairs)


# --------------------------------------------------------------------------
# Handlers
# --------------------------------------------------------------------------


def handle_build(state: GameState, faction: str, cmd: ParsedCommand) -> GameState:
    assert cmd.loc is not None
    hex_key = cmd.loc
    if hex_key not in state.hexes:
        raise EngineError(f"unknown hex {hex_key!r}", state=state, faction=faction, cmd=cmd)

    hex_state = state.hexes[hex_key]
    if hex_state.building is not None:
        raise EngineError(
            f"{hex_key} already has a {hex_state.building}", state=state, faction=faction, cmd=cmd
        )

    setup = state.phase == Phase.SETUP_DWELLINGS
    color = FACTIONS[faction].color
    tf_needed = hex_state.color != color

    free_d_index = None if setup else _find_pending_optional(state, faction, "free_d")

    if tf_needed and (setup or free_d_index is not None):
        # ACTW's FREE_D marker (module docstring) does NOT bypass this
        # check -- build_color_ok is never skipped in Perl, only cost and
        # reachability are. Setup dwellings are always placed on an
        # already-matching hex by construction, so this branch is a hard
        # error there too.
        raise EngineError(
            f"{hex_key} is {hex_state.color}, not {faction}'s home color {color}",
            state=state,
            faction=faction,
            cmd=cmd,
        )

    if not setup and free_d_index is None and hex_key not in reachable(state, faction):
        raise EngineError(
            f"{hex_key} is not reachable by {faction}", state=state, faction=faction, cmd=cmd
        )

    fs = state.factions[faction]
    d_track = FACTIONS[faction].buildings["D"]
    if len(fs.buildings["D"]) >= d_track.max_count:
        raise EngineError(
            f"{faction} already has the maximum {d_track.max_count} dwellings",
            state=state,
            faction=faction,
            cmd=cmd,
        )

    free_tf_index: int | None = None
    if tf_needed:
        # command_build's own implicit "transform $where to $color" dispatch
        # (commands.pm 213-219, module docstring) -- an ordinary (non-FREE_D)
        # build on a wrong-colored hex pays spades_available to recolor it
        # to home color before placing the dwelling, in the same ledger row,
        # with no separate `transform` command of its own (empirically:
        # reference-game row 58, "burn 6. action ACT6. transform G2. build
        # F5" -- F5 is brown, 1 spade from red->yellow's distance away, and
        # the only explicit `transform` in the row targets G2, a different
        # hex prepped for a later turn; task-13 report). Since Perl's
        # internal dispatch goes through the *same* `command_transform`,
        # a live `free_tf` marker (Nomads' ACTN Sandstorm) makes that
        # implicit transform free too, gated by direct hex adjacency
        # instead of spades_available/reachability (reference-game row
        # 174: "action ACTN. build F2" -- see actions_terraform.py's
        # `handle_transform` for the sibling bare-`transform` path this
        # mirrors, and `_directly_adjacent_to_own_building`'s docstring
        # for why it's duplicated here rather than imported).
        free_tf_index = _find_pending_optional(state, faction, "free_tf")
        if free_tf_index is not None:
            if not _directly_adjacent_to_own_building(state, faction, hex_key):
                raise EngineError(
                    f"{hex_key} is not directly adjacent to a {faction} building "
                    "(ACTN requires direct hex adjacency)",
                    state=state,
                    faction=faction,
                    cmd=cmd,
                )
            cost = 0
        else:
            cost = hooks_for(faction).spade_transform_cost(state, faction, hex_state.color, color)
            if cost > fs.spades_available:
                raise EngineError(
                    f"{hex_key} needs {cost} spades to transform to {faction}'s home color "
                    f"{color} ({faction} has {fs.spades_available})",
                    state=state,
                    faction=faction,
                    cmd=cmd,
                )
        fs = replace(fs, spades_available=fs.spades_available - cost)
        hex_state = replace(hex_state, color=color)

    if not setup and free_d_index is None:
        # commands.pm 220-227 (tf_needed=false) / map.pm transform_cost's own
        # check_reachable call (tf_needed=true, folded into the implicit
        # transform above) -- either way, a build that could only reach
        # hex_key via the faction's TeleportTrack (Dwarves tunnel / Fakirs
        # carpet) pays that crossing's own W cost and gains its own VP,
        # *in addition to* the D building's ordinary cost below. Empty
        # ({}, 0) for a direct/shipping-reached hex (connectivity.py's
        # ``teleport_crossing`` docstring). Skipped entirely under FREE_D
        # (ACTW grants FREE_D+TELEPORT_NO_TF together, Game/Constants.pm
        # line 76; TELEPORT_NO_TF's branch, commands.pm 207-210, never
        # calls check_reachable at all) -- the ``free_d_index is None``
        # guard on this whole block already matches that. Also skipped if
        # ``hex_key`` already equals ``fs.teleported_hex`` -- an earlier
        # ``transform`` on this same hex *this same turn* already paid
        # this fee (``FactionState.teleported_hex`` docstring, task-14
        # fix); a build in a *later* turn on the same hex pays fresh
        # (``teleported_hex`` is reset every turn, so it won't still
        # match by then).
        if hex_key != fs.teleported_hex:
            teleport_cost, teleport_vp = teleport_crossing(state, faction, hex_key)
            if teleport_cost:
                fs = _pay(state, faction, fs, teleport_cost, cmd)
            if teleport_vp:
                fs = replace(fs, vp=fs.vp + teleport_vp)
            if teleport_cost or teleport_vp:
                fs = replace(fs, teleported_hex=hex_key)
        fs = _pay(state, faction, fs, d_track.cost, cmd)

    fs = replace(fs, buildings={**fs.buildings, "D": fs.buildings["D"] | {hex_key}})
    if not setup:
        fs = replace(fs, vp=fs.vp + _favor_tile_vp(fs, "D") + _current_score_tile_vp(state, "D", "build"))
    new_hexes = dict(state.hexes)
    new_hexes[hex_key] = replace(hex_state, building="D", owner=faction)
    new_state = replace(with_faction(state, faction, fs), hexes=new_hexes)

    if free_d_index is not None:
        new_state = pop_pending(new_state, free_d_index)
    if free_tf_index is not None:
        new_state = pop_pending(new_state, free_tf_index)

    if not setup:
        new_state = leech.queue_leech(new_state, faction, hex_key)

    return _maybe_queue_town(new_state, faction)


def handle_upgrade(state: GameState, faction: str, cmd: ParsedCommand) -> GameState:
    assert cmd.loc is not None and cmd.building is not None
    hex_key, new_type = cmd.loc, cmd.building
    if hex_key not in state.hexes:
        raise EngineError(f"unknown hex {hex_key!r}", state=state, faction=faction, cmd=cmd)

    hex_state = state.hexes[hex_key]
    color = FACTIONS[faction].color
    if hex_state.color != color:
        raise EngineError(
            f"{hex_key} is {hex_state.color}, not {faction}'s home color {color}",
            state=state,
            faction=faction,
            cmd=cmd,
        )

    old_type = _UPGRADE_FROM.get(new_type)
    if old_type is None:
        raise EngineError(
            f"unknown building type {new_type!r}", state=state, faction=faction, cmd=cmd
        )
    if hex_state.building != old_type:
        raise EngineError(
            f"{hex_key} contains {hex_state.building}, wanted {old_type}",
            state=state,
            faction=faction,
            cmd=cmd,
        )

    fs = state.factions[faction]
    track = FACTIONS[faction].buildings[new_type]
    if len(fs.buildings[new_type]) >= track.max_count:
        raise EngineError(
            f"{faction} already has the maximum {track.max_count} {new_type}s",
            state=state,
            faction=faction,
            cmd=cmd,
        )

    offers = leech.offers_for_build(state, faction, hex_key)

    free_tp_index = _find_pending_optional(state, faction, "free_tp") if new_type == "TP" else None

    if free_tp_index is not None:
        cost: dict[str, int] = {}
    elif new_type == "TP":
        cost = dict(track.cost)
        if not offers:  # no adjacent opponent building -> isolated surcharge
            cost["C"] = cost.get("C", 0) * 2
    else:
        cost = track.cost

    fs = _pay(state, faction, fs, cost, cmd)

    level_index = len(fs.buildings[new_type])  # count BEFORE this copy is added
    fs_buildings = dict(fs.buildings)
    fs_buildings[old_type] = fs_buildings[old_type] - {hex_key}
    fs_buildings[new_type] = fs_buildings[new_type] | {hex_key}
    fs = replace(fs, buildings=fs_buildings)
    fs = replace(
        fs,
        vp=fs.vp + _favor_tile_vp(fs, new_type) + _current_score_tile_vp(state, new_type, "build"),
    )

    new_hexes = dict(state.hexes)
    new_hexes[hex_key] = replace(hex_state, building=new_type, owner=faction)
    new_state = replace(with_faction(state, faction, fs), hexes=new_hexes)

    if free_tp_index is not None:
        new_state = pop_pending(new_state, free_tp_index)

    gain = track.build_gain[level_index] if level_index < len(track.build_gain) else {}
    new_state = _apply_build_gain(new_state, faction, gain)

    if new_type == "SH":
        new_state = hooks_for(faction).on_stronghold_built(new_state, faction)

    new_state = leech.queue_leech(new_state, faction, hex_key, offers=offers)
    return _maybe_queue_town(new_state, faction)


def handle_bridge(state: GameState, faction: str, cmd: ParsedCommand) -> GameState:
    assert cmd.loc is not None and cmd.loc2 is not None
    a, b = cmd.loc, cmd.loc2
    idx = _find_pending(state, faction, "bridge", cmd)

    if a not in state.hexes or b not in state.hexes:
        raise EngineError(f"unknown bridge hex ({a}:{b})", state=state, faction=faction, cmd=cmd)

    fs = state.factions[faction]
    if fs.bridges_built >= BRIDGE_COUNT:
        raise EngineError(
            f"{faction} has already placed all {BRIDGE_COUNT} bridges",
            state=state,
            faction=faction,
            cmd=cmd,
        )

    pair = frozenset({a, b})
    if pair not in _bridgable_pairs():
        raise EngineError(
            f"{a}:{b} is not a legal bridge span", state=state, faction=faction, cmd=cmd
        )

    if state.hexes[a].owner != faction and state.hexes[b].owner != faction:
        raise EngineError(
            f"bridge {a}:{b} must touch a {faction} building", state=state, faction=faction, cmd=cmd
        )

    new_state = pop_pending(state, idx)
    new_state = replace(new_state, bridges=new_state.bridges | {pair})
    new_fs = replace(fs, bridges_built=fs.bridges_built + 1)
    new_state = with_faction(new_state, faction, new_fs)
    return _maybe_queue_town(new_state, faction)


def handle_gain_favor(state: GameState, faction: str, cmd: ParsedCommand) -> GameState:
    """``+FAVx``: taking a favor tile never scores VP by itself, even for
    FAV10/FAV11 (``FavorTile.vp``). ``FavorTile.vp`` is a *passive* bonus
    scored by ``_favor_tile_vp`` at every later matching build/upgrade
    while the tile is held (``scoring.pm``'s ``maybe_score_favor_tile``,
    called from ``command_build``/``command_upgrade`` right before
    ``maybe_score_current_score_tile`` -- see ``tiles.scored_vp``'s own
    docstring for that sibling mechanic), *not* a one-time snapshot taken
    at grant time. An earlier revision of this handler scored
    ``len(fs.buildings[btype]) * per_unit`` here instead -- disproved by
    replay (task-13 report, reference-game row 59): mermaids gain FAV11
    (``vp={"D": 2}``) with 1 dwelling already on the board and VP is
    unchanged (delta 0); building a fresh dwelling ten rows later (row 69)
    is what grants the +2.
    """
    assert cmd.tile is not None
    tile = cmd.tile
    idx = _find_pending(state, faction, "gain_favor", cmd)

    if state.favors_pool.get(tile, 0) < 1:
        raise EngineError(
            f"favor tile {tile} not available in pool", state=state, faction=faction, cmd=cmd
        )
    fs = state.factions[faction]
    if tile in fs.favors:
        raise EngineError(f"{faction} already holds {tile}", state=state, faction=faction, cmd=cmd)

    favor = FAVOR_TILES[tile]
    fs = replace(fs, favors=fs.favors + (tile,))

    result = advance(
        state.cults[faction][favor.cult],
        favor.steps,
        keys_available=fs.keys,
        track_open=state.cult_10[favor.cult] is None,
    )
    new_cults = {f: dict(v) for f, v in state.cults.items()}
    new_cults[faction][favor.cult] = result.new_value
    new_cult_10 = dict(state.cult_10)
    new_keys = fs.keys
    new_cult_blocked = fs.cult_blocked
    if result.key_spent:
        new_keys -= 1
        new_cult_10[favor.cult] = faction
    if result.blocked_at_9:
        new_cult_blocked = fs.cult_blocked | {favor.cult}
    fs = replace(
        fs, power=fs.power.gain(result.power_gained), keys=new_keys, cult_blocked=new_cult_blocked
    )

    new_pool = dict(state.favors_pool)
    new_pool[tile] = new_pool[tile] - 1

    new_state = replace(state, cults=new_cults, cult_10=new_cult_10, favors_pool=new_pool)
    new_state = with_faction(new_state, faction, fs)
    new_state = replace(new_state, pending=_consume_amount(new_state.pending, idx, 1))

    if tile == "FAV5":
        # resources.pm's adjust_resource FAV branch, the "Hack" comment
        # (250-395 area): taking FAV5 immediately re-scans every building
        # hex the faction owns for a town that *now* qualifies under the
        # lowered TOWN_SIZE threshold (FAV5's own passive, already read by
        # towns.py's new_towns/_town_threshold once fs.favors includes
        # it) -- not deferred to that hex's next build/upgrade. Reusing
        # _maybe_queue_town (the same detection this module already runs
        # after build/upgrade/bridge) rather than re-deriving the scan
        # (task-13 report, corpus game 4pLeague_S10_D1L1_G3 row 321:
        # swarmlings' "upgrade H2 to TE. gain_favor FAV5. gain_town TW8"
        # -- the TE upgrade alone doesn't reach TOWN_SIZE=7, but FAV5's
        # TOWN_SIZE=6 threshold, applied within the same row, does).
        new_state = _maybe_queue_town(new_state, faction)

    return new_state


def _advance_cult_track(state: GameState, faction: str, cult: str, steps: int) -> GameState:
    """Shared single-track cult-advance fold: ``cults.advance`` plus the
    resulting power/key/``cult_10``/``cult_blocked`` bookkeeping, threaded
    onto ``state``. Used by :func:`_apply_town_cult_gains` and
    :func:`_retry_blocked_cults` below (a near-identical inline copy also
    lives in ``handle_gain_favor``, left alone since it predates this
    helper and is already covered by its own passing tests -- see that
    handler's own docstring).
    """
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
    return with_faction(replace(state, cults=new_cults, cult_10=new_cult_10), faction, new_fs)


def _apply_town_cult_gains(state: GameState, faction: str, tile: str) -> GameState:
    """Drive ``cults.advance`` for ``tile``'s FIRE/WATER/EARTH/AIR gain
    keys (TW5/TW6) -- ``towns.py``'s own docstring leaves these
    deliberately unapplied by ``apply_town_tile`` ("do NOT import cult
    logic here; just expose the gain... the caller must read
    ``TOWN_TILES[tile].gain`` for these keys and drive ``cults.advance``
    itself"). This handler is that caller -- one track at a time via
    :func:`_advance_cult_track`, so a threshold crossed on an earlier
    track in the same grant can't affect a later one's key/blocked
    bookkeeping.

    Missing this was a real engine bug, not a documented deferral: the
    task brief for towns.py explicitly named this the caller's job, but
    no caller existed before this replay harness ran the reference game
    end to end (task-13 report, row 203: darklings' TW5 grant should add
    1 step to all four cult tracks -- and the power crossing FIRE=3,
    WATER=5, EARTH=8 grants -- but nothing drove it).
    """
    gain = TOWN_TILES[tile].gain
    for cult in _TOWN_CULT_GAIN_KEYS:
        steps = gain.get(cult, 0)
        if steps:
            state = _advance_cult_track(state, faction, cult, steps)
    return state


def _retry_blocked_cults(state: GameState, faction: str) -> GameState:
    """``resources.pm`` ``adjust_resource``'s ``KEY`` branch (355-362, cited
    in ``cults.py`` ``advance``'s own docstring): every town tile grants at
    least 1 KEY (``Game/Constants.pm`` ``%tiles``, TW1-TW8 all have
    ``KEY => 1`` or ``2`` in ``.gain`` -- ``apply_town_tile`` already folds
    that resource in). Whenever a faction's KEY balance increases and now
    covers *every* cult track still parked at 9 for lack of a key
    (``fs.cult_blocked``), each of those tracks automatically retries its
    final +1 step -- Perl's own ``$faction->{KEY} >= scalar keys
    %{$faction->{cult_blocked}}`` gates the whole batch at once, not
    cult-by-cult, so nothing retries unless the new KEY balance can cover
    all of them together. Called once per ``gain_town`` resolution, after
    that tile's own KEY/cult grants (``apply_town_tile``/
    :func:`_apply_town_cult_gains`) have already landed.

    Missing this was a real engine bug: a faction whose cult step got
    capped at 9 for lack of a key earlier in the same turn, then gains a
    key from a same-row ``gain_town``, should retroactively reach 10 --
    task-13 report follow-up, ``4pLeague_S10_D1L1_G6`` row 315: nomads'
    FAV5 grant (2 FIRE steps, 8->10) blocked at 9 with 0 keys; the very
    next command in the row, ``gain_town TW7``, grants exactly the 1 key
    needed and should retroactively bump FIRE to 10.
    """
    fs = state.factions[faction]
    blocked = fs.cult_blocked
    if not blocked or fs.keys < len(blocked):
        return state
    for cult in sorted(blocked):
        state = _advance_cult_track(state, faction, cult, 1)
    fs = state.factions[faction]
    return with_faction(state, faction, replace(fs, cult_blocked=frozenset()))


def _apply_town_ship_gain(state: GameState, faction: str, tile: str) -> GameState:
    """Drive ``_advance_shipping`` for ``tile``'s ``GAIN_SHIP`` key (TW7)
    -- the other half of the deferral ``towns.py``'s docstring flags
    alongside the cult-step keys ``_apply_town_cult_gains`` handles
    (``carpet_range``, TeleportTrack's sibling key, stays deferred: no
    corpus evidence yet that any Dwarves/Fakirs game needs it, and this
    engine's roster of games sampled so far never grants TW7 to either).
    """
    units = TOWN_TILES[tile].gain.get("GAIN_SHIP", 0)
    if not units:
        return state
    fs = _advance_shipping(faction, state.factions[faction], units)
    return with_faction(state, faction, fs)


def handle_gain_town(state: GameState, faction: str, cmd: ParsedCommand) -> GameState:
    """Pop the matching ``gain_town`` pending(s) and apply ``cmd.tile`` to
    the cluster(s) they were queued for (``pending.source``, see
    ``_maybe_queue_town``/``_cluster_key``). Each cluster was already
    recorded into ``founded_towns`` at *detection* time -- this handler
    only grants the tile(s), it does not re-derive or re-record the
    cluster(s).

    ``cmd.n1`` (ledger raw ``+NTWx``, default 1) is not a scaled reward --
    it is a *count* of separate ``gain_town`` pendings being resolved with
    the *same* tile type in one ledger row (``commands.pm``'s generic
    ``+N<TYPE>`` handler, lines 85-92: ``$faction->{GAIN_TW} -= $delta``
    against a single counter that ``_maybe_queue_town`` can push above 1
    when two clusters qualify at once, e.g. a FAV5 rescan -- fix #13's
    docstring -- surfacing a second cluster the very same row an upgrade's
    own ``_maybe_queue_town`` call already queued one for). ``resources.pm``
    ``adjust_resource``'s own ``TW`` branch (355-362) confirms this: ``for
    (1..$delta) { gain $faction, $tiles{$type}{gain}, 'TW' }`` -- the
    *entire* per-tile gain (VP/resources/cult steps/shipping) is applied
    ``$delta`` times, once per pending resolved, not scaled once. Task-13
    report, ``4pLeague_S10_D1L1_G4`` row 257 (raw ``+2TW1``): engine
    previously read only ``cmd.tile`` and ignored ``cmd.n1`` entirely,
    granting TW1 once instead of twice.
    """
    assert cmd.tile is not None
    count = cmd.n1 if cmd.n1 is not None else 1
    new_state = state
    for _ in range(count):
        idx = _find_pending(new_state, faction, "gain_town", cmd)
        pending = new_state.pending[idx]
        if pending.source is None:
            raise EngineError(
                "gain_town pending missing its cluster source (internal bookkeeping bug -- "
                "_maybe_queue_town should always set it)",
                state=state,
                faction=faction,
                cmd=cmd,
            )

        new_state = pop_pending(new_state, idx)
        try:
            new_state = apply_town_tile(new_state, faction, cmd.tile)
        except ValueError as exc:
            raise EngineError(str(exc), state=state, faction=faction, cmd=cmd) from exc
        new_state = _apply_town_cult_gains(new_state, faction, cmd.tile)
        new_state = _apply_town_ship_gain(new_state, faction, cmd.tile)
        new_state = _retry_blocked_cults(new_state, faction)
    return new_state


register_handler("build", handle_build)
register_handler("upgrade", handle_upgrade)
register_handler("bridge", handle_bridge)
register_handler("gain_favor", handle_gain_favor)
register_handler("gain_town", handle_gain_town)
