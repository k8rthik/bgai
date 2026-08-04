"""``pass``/``advance``/``connect`` handlers: end-of-turn bonus-tile swap,
shipping/dig track advancement, and Mermaids' river-hex town connection.

Ported from the reference implementation (jsnell/terra-mystica, MIT):

- ``commands.pm`` ``command_pass`` (roughly lines 750-800): scores the held
  bonus tile's ``pass_vp`` table (``tiles.py`` ``BonusTile.pass_vp``, one
  entry per unit of the named building/track currently on the board --
  ``"ship"`` reads ``FactionState.shipping``'s current level directly,
  every other key reads a building-type count), every held favor tile's
  own ``pass_vp`` table (only FAV12 has one -- a count-indexed lookup, not
  a per-unit rate, per ``tiles.py``'s own docstring), and
  ``hooks_for(faction).pass_vp_extra`` (Engineers' stronghold bridge bonus,
  registered in this module -- see ``_EngineersHooks`` below). The faction
  then returns its old bonus tile (if any) and, unless this is a bare
  round-6 pass (``cmd.tile is None``), takes ``cmd.tile`` -- paying out
  whatever coins have accumulated on it via ``GameState.bonus_coins``
  (``round_flow.end_of_round``'s "+1 coin on every untaken tile" rule,
  cross-checked against the crawled corpus: game 4pLeague_S10_D1L1_G1 row
  83, nomads takes BON7 after it sat unclaimed since round 1 and gains
  exactly 1 C; row 298, nomads takes BON4 after 4 unclaimed rounds and
  gains exactly 4 C; the round-6 bare ``pass`` rows -- e.g. that same
  game's rows 361/362/366/367 -- carry no ``tile`` and show a 0 C delta).
  Marks ``fs.passed`` and appends to ``passed_order`` -- consumed by
  ``round_flow.advance_turn``'s ACTIONS-phase faction rotation and, under
  ``variable_turn_order``, by ``round_flow.end_of_round``'s next-round
  ``turn_order``.

  During ``Phase.SETUP_BONUS`` the same ``pass`` verb instead means "pick
  your starting bonus tile" (crawled corpus: game 4pLeague_S10_D1L1_G1 rows
  37-40, ``mermaids pass BON1`` / ``nomads pass BON5`` / ... in *reverse*
  seat order, immediately after the SETUP_DWELLINGS snake-order dwelling
  rows) -- no old tile to return, no accumulated coins (every
  ``bonus_coins`` entry is still 0 pre-round-1), no VP, and ``fs.passed``/
  ``passed_order`` are left untouched (those belong to the real
  round-by-round ACTIONS-phase pass, not the one-time initial pick).

- ``commands.pm`` ``command_advance`` (``advance ship``/``advance dig``):
  pays the faction's current-level ``ShippingTrack.advance_cost``/
  ``DigTrack.advance_cost`` once, scores ``advance_vp[level]`` (indexed by
  the level *before* advancing -- cross-checked against the corpus, see
  ``handle_advance``'s docstring), and bumps the track by one level.
  Dwarves (no shipping track) and Darklings (dig track capped at level 0,
  never advances) both have ``advance_cost=None``, which this handler
  reads as "no such track to advance" rather than defaulting to some
  fallback cost. Fakirs' carpet-flight range and Dwarves' tunneling level
  are *not* reached through this verb -- the ledger grammar
  (``ledger_parser.py``) only recognizes ``advance ship``/``advance dig``,
  never a "carpet"/"teleport" variant, and both factions' ranges instead
  advance immediately as part of their stronghold's ``build_gain``
  (``GAIN_TELEPORT``, applied by ``actions_build.py`` at SH build time --
  see that module's ``_apply_build_gain``). Mermaids' free first shipping
  level from their SH (``GAIN_SHIP`` in ``factions_data.py``'s
  ``build_gain``) is likewise **already applied immediately** by
  ``actions_build.py``'s ``_apply_build_gain`` (the ``GAIN_SHIP`` branch
  bumps ``fs.shipping`` directly, capped at the track's ``max_level``) --
  the task brief flagged this as an open seam from Task 8, but it turns
  out no further plumbing is needed here: this module's ``handle_advance``
  only ever processes the *paid* ``advance ship``/``advance dig`` ledger
  rows, which are wholly independent of that one-time free grant.

- ``towns.pm`` ``check_mermaid_river_connection_town`` (``connect``, e.g.
  ``connect R1``): re-detects Mermaids' town candidates via
  ``towns.new_towns`` (which, for Mermaids, already includes river-hex
  join candidates alongside plain ones -- see that module's docstring) and
  picks whichever qualifying cluster touches the named river hex, pushing
  a ``gain_town`` pending exactly like ``actions_build.py``'s
  ``_maybe_queue_town`` does for a plain build/upgrade/bridge. Plain
  clusters are always already recorded into ``founded_towns`` (and their
  pendings already queued) by the preceding ``build``/``upgrade``/
  ``bridge`` row in the same turn by the time ``connect`` runs, so
  ``new_towns`` at that point can only surface a genuinely new
  river-joined candidate -- cross-checked against the crawled corpus (game
  4pLeague_S10_D1L1_G1 row 208: ``dig 1. build A3. connect R1.
  gain_town TW1``, all one ledger row/turn). **Known limitation**: this
  reasoning assumes ``connect`` always follows a ``build``/``upgrade``/
  ``bridge`` row earlier in the *same* turn, which is the only pattern
  observed across the sampled corpus rows -- a hypothetical ``connect``
  fired with no board change earlier that turn is unverified (there is
  nothing in principle wrong with ``handle_connect`` in that case, since
  it re-derives candidates fresh rather than trusting a stale detection
  pass, but no corpus row was found to confirm it).

Engineers' stronghold pass-VP hook (``scoring.pm`` lines 166-178, quoted in
``factions_data.py``'s ``notes`` field): "3 VP for each bridge whose two
ends are gray hexes with buildings." Registered here (not in
``actions_build.py``, which only grants the stronghold itself) since it
only ever fires at pass time. ``state.bridges`` is a flat,
faction-agnostic hex-pair set (``state.py``'s own docstring), but gray is
Engineers' unique home color and ``handle_bridge`` requires an endpoint
already owned by the bridging faction, so "a bridge with two gray,
built-on endpoints" is unambiguous without needing per-faction bridge
ownership bookkeeping.
"""

from __future__ import annotations

from dataclasses import replace

from bgai.data.ledger_parser import ParsedCommand
from bgai.engine.tm.apply import EngineError, push_pending, register_handler
from bgai.engine.tm.board import RIVER, base_board
from bgai.engine.tm.factions.hooks import HOOKS, FactionHooks, hooks_for
from bgai.engine.tm.factions_data import FACTIONS
from bgai.engine.tm.state import FactionState, GameState, PendingDecision, Phase, with_faction
from bgai.engine.tm.tiles import BONUS_TILES, FAVOR_TILES
from bgai.engine.tm.towns import new_towns, record_founded_town

_ADVANCE_COST_RES: dict[str, str] = {"W": "workers", "C": "coins", "P": "priests"}
_ENGINEERS_GRAY = "gray"


class _EngineersHooks(FactionHooks):
    """``scoring.pm`` 166-178 (module docstring): 3 VP at pass time per
    bridge connecting two built-on gray hexes, once the stronghold is up.
    """

    def pass_vp_extra(self, state: GameState, faction: str) -> int:
        fs = state.factions[faction]
        if not fs.buildings.get("SH"):
            return 0
        count = 0
        for pair in state.bridges:
            a, b = tuple(pair)
            ha, hb = state.hexes[a], state.hexes[b]
            if (
                ha.color == _ENGINEERS_GRAY
                and hb.color == _ENGINEERS_GRAY
                and ha.building is not None
                and hb.building is not None
            ):
                count += 1
        return count * 3


HOOKS["engineers"] = _EngineersHooks()


# --------------------------------------------------------------------------
# pass
# --------------------------------------------------------------------------


def _count_for_pass_vp(fs: FactionState, key: str) -> int:
    if key == "ship":
        return fs.shipping
    return len(fs.buildings.get(key, frozenset()))


def _bonus_pass_vp(fs: FactionState) -> int:
    if fs.bonus is None:
        return 0
    bon = BONUS_TILES[fs.bonus]
    return sum(per_unit * _count_for_pass_vp(fs, key) for key, per_unit in bon.pass_vp)


def _favor_pass_vp(fs: FactionState) -> int:
    total = 0
    for favor_id in fs.favors:
        fav = FAVOR_TILES[favor_id]
        for building, table in fav.pass_vp.items():
            count = len(fs.buildings.get(building, frozenset()))
            index = min(count, len(table) - 1)
            total += table[index]
    return total


def _handle_setup_bonus_pick(state: GameState, faction: str, cmd: ParsedCommand) -> GameState:
    """``Phase.SETUP_BONUS``: choose a starting bonus tile (module docstring)."""
    assert cmd.tile is not None
    tile = cmd.tile
    if tile not in state.setup.bonus_tiles:
        raise EngineError(
            f"{tile} is not in this game's bonus pool", state=state, faction=faction, cmd=cmd
        )
    if any(other.bonus == tile for other in state.factions.values()):
        raise EngineError(f"{tile} is already held", state=state, faction=faction, cmd=cmd)
    fs = replace(state.factions[faction], bonus=tile)
    return with_faction(state, faction, fs)


def handle_pass(state: GameState, faction: str, cmd: ParsedCommand) -> GameState:
    """``pass[ BONx]``: score pass-VP, return the old bonus tile, and (unless
    this is a bare round-6 pass) take ``cmd.tile`` with its accumulated
    ``bonus_coins`` (module docstring). During ``Phase.SETUP_BONUS`` this is
    instead the one-time starting-tile pick, with none of that machinery.
    A ``pass`` row is only ever legal during ``Phase.ACTIONS`` (the real
    round-by-round pass) or ``Phase.SETUP_BONUS`` (the one-time pick) --
    any other phase raises.
    """
    if state.phase == Phase.SETUP_BONUS:
        return _handle_setup_bonus_pick(state, faction, cmd)
    if state.phase != Phase.ACTIONS:
        raise EngineError(
            f"pass is not legal during {state.phase.name}", state=state, faction=faction, cmd=cmd
        )

    fs = state.factions[faction]
    vp = _bonus_pass_vp(fs) + _favor_pass_vp(fs) + hooks_for(faction).pass_vp_extra(state, faction)
    fs = replace(fs, vp=fs.vp + vp, bonus=None, passed=True)
    new_state = with_faction(state, faction, fs)
    new_state = replace(new_state, passed_order=new_state.passed_order + (faction,))

    if cmd.tile is None:
        return new_state  # round 6 bare pass: no tile taken

    tile = cmd.tile
    if tile not in new_state.setup.bonus_tiles:
        raise EngineError(
            f"{tile} is not in this game's bonus pool", state=state, faction=faction, cmd=cmd
        )
    if any(other.bonus == tile for other in new_state.factions.values()):
        raise EngineError(f"{tile} is already held", state=state, faction=faction, cmd=cmd)

    coins = new_state.bonus_coins.get(tile, 0)
    old_fs = new_state.factions[faction]
    fs2 = replace(old_fs, bonus=tile, coins=old_fs.coins + coins)
    new_bonus_coins = dict(new_state.bonus_coins)
    new_bonus_coins[tile] = 0
    new_state = with_faction(new_state, faction, fs2)
    return replace(new_state, bonus_coins=new_bonus_coins)


# --------------------------------------------------------------------------
# advance (ship/dig)
# --------------------------------------------------------------------------


def _pay(
    state: GameState, faction: str, fs: FactionState, cost: dict[str, int], cmd: ParsedCommand
) -> FactionState:
    for res, amount in cost.items():
        attr = _ADVANCE_COST_RES[res]
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


def handle_advance(state: GameState, faction: str, cmd: ParsedCommand) -> GameState:
    """``advance ship``/``advance dig``: pay the current level's
    ``advance_cost``, score ``advance_vp[level]`` (index = level *before*
    advancing -- module docstring), bump the track by one. ``cmd.n1`` (a
    rare early-era explicit target level, e.g. "advance ship to 1") is
    purely informational here, same treatment ``handle_dig`` gives
    ``cmd.loc`` in ``actions_terraform.py``.
    """
    assert cmd.reason in ("ship", "dig")
    fs = state.factions[faction]
    data = FACTIONS[faction]

    if cmd.reason == "ship":
        ship_track = data.shipping
        level = fs.shipping
        if ship_track.advance_cost is None or level >= ship_track.max_level:
            raise EngineError(
                f"{faction} has no shipping advance available at level {level}",
                state=state,
                faction=faction,
                cmd=cmd,
            )
        fs = _pay(state, faction, fs, ship_track.advance_cost, cmd)
        fs = replace(fs, shipping=level + 1, vp=fs.vp + ship_track.advance_vp[level])
    else:
        dig_track = data.dig
        level = fs.dig_level
        if dig_track.advance_cost is None or level >= dig_track.max_level:
            raise EngineError(
                f"{faction} has no dig advance available at level {level}",
                state=state,
                faction=faction,
                cmd=cmd,
            )
        fs = _pay(state, faction, fs, dig_track.advance_cost, cmd)
        fs = replace(fs, dig_level=level + 1, vp=fs.vp + dig_track.advance_vp[level])

    return with_faction(state, faction, fs)


# --------------------------------------------------------------------------
# connect (Mermaids river town)
# --------------------------------------------------------------------------


def _cluster_key(cluster: frozenset[str]) -> str:
    return ",".join(sorted(cluster))


def handle_connect(state: GameState, faction: str, cmd: ParsedCommand) -> GameState:
    """``connect R<n>``: Mermaids-only river-hex town join (module docstring)."""
    assert cmd.loc is not None
    river = cmd.loc
    if faction != "mermaids":
        raise EngineError(
            f"{faction} cannot connect (Mermaids-only)", state=state, faction=faction, cmd=cmd
        )
    if river not in state.hexes:
        raise EngineError(f"unknown hex {river!r}", state=state, faction=faction, cmd=cmd)
    if state.hexes[river].color != RIVER:
        raise EngineError(f"{river} is not a river hex", state=state, faction=faction, cmd=cmd)

    board_adjacent = _river_neighbors(river)
    candidates = new_towns(state, faction)
    matches = sorted(
        (c for c in candidates if board_adjacent & c),
        key=_cluster_key,
    )
    if not matches:
        raise EngineError(
            f"no qualifying town cluster touches river hex {river}",
            state=state,
            faction=faction,
            cmd=cmd,
        )
    cluster = matches[0]
    new_state = record_founded_town(state, faction, cluster)
    pending = PendingDecision(
        faction=faction, kind="gain_town", amount=1, source=_cluster_key(cluster)
    )
    return push_pending(new_state, pending)


def _river_neighbors(river: str) -> frozenset[str]:
    board = base_board()
    return board.adjacent.get(river, frozenset())


register_handler("pass", handle_pass)
register_handler("advance", handle_advance)
register_handler("connect", handle_connect)
