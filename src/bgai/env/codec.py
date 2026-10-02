"""Structural codec: ``SimState`` <-> JSON-safe dict <-> bytes.

Why not pickle: a pickled ``SimState`` is bound to this exact class layout
and to the Python that wrote it, so it cannot be a stored artefact, cannot
cross a language boundary, and cannot be diffed when an engine field is
added. An explicit codec can -- and it fails loudly on an unknown field
instead of silently resurrecting a stale shape.

Every encoder here writes JSON-safe primitives only. Sets become *sorted*
lists so two encodings of the same state are byte-identical (a frozenset's
iteration order is not stable across processes), which is what makes
``state_bytes`` usable as a position key.

Round-trip contract, pinned in tests/test_env_codec.py:
``sim_from_dict(sim_to_dict(sim)) == sim`` and
``sim_from_bytes(sim_to_bytes(sim)) == sim`` for states drawn from live
play at 2, 3, 4 and 5 players.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import orjson

from bgai.arena.driver import SimState
from bgai.engine.tm.power import Power
from bgai.engine.tm.setup import GameOptions, GameSetup
from bgai.engine.tm.state import (
    FactionState,
    GameState,
    HexState,
    PendingDecision,
    Phase,
)
from bgai.engine.tm.tiles import ScoringTile
from bgai.env.config import EnvError

CODEC_VERSION = 1
"""Bump on any change to the dict shape below. ``sim_from_dict`` refuses a
payload whose version it does not know rather than guessing."""


class CodecError(EnvError, ValueError):
    """A payload that is not a valid encoded state."""


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------


def _pairs_out(pairs: Sequence[Sequence[Any]]) -> list[list[Any]]:
    return [list(pair) for pair in pairs]


def _pairs_in(raw: Any) -> tuple[tuple[Any, ...], ...]:
    return tuple(tuple(pair) for pair in raw)


def _require(payload: Mapping[str, Any], key: str) -> Any:
    if key not in payload:
        raise CodecError(f"encoded state is missing required key {key!r}")
    return payload[key]


# --------------------------------------------------------------------------
# setup
# --------------------------------------------------------------------------


def _options_out(options: GameOptions) -> dict[str, bool]:
    return {f.name: bool(getattr(options, f.name)) for f in _fields(GameOptions)}


def _options_in(raw: Mapping[str, Any]) -> GameOptions:
    known = {f.name for f in _fields(GameOptions)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise CodecError(f"unknown GameOptions fields {unknown}")
    return GameOptions(**{k: bool(v) for k, v in raw.items()})


def _fields(cls: type) -> tuple[Any, ...]:
    from dataclasses import fields

    return fields(cls)


def _score_tile_out(tile: ScoringTile) -> dict[str, Any]:
    return {
        "cult": tile.cult,
        "req": tile.req,
        "vp_mode": tile.vp_mode,
        "vp": _pairs_out(tile.vp),
        "cult_income": _pairs_out(tile.cult_income),
        "vp_display": tile.vp_display,
        "income_display": tile.income_display,
    }


def _score_tile_in(raw: Mapping[str, Any]) -> ScoringTile:
    return ScoringTile(
        cult=raw["cult"],
        req=int(raw["req"]),
        vp_mode=raw["vp_mode"],
        vp=_pairs_in(raw["vp"]),
        cult_income=_pairs_in(raw["cult_income"]),
        vp_display=raw.get("vp_display", ""),
        income_display=raw.get("income_display", ""),
    )


def setup_to_dict(setup: GameSetup) -> dict[str, Any]:
    return {
        "game_id": setup.game_id,
        "options": _options_out(setup.options),
        "factions": list(setup.factions),
        "score_tiles": [_score_tile_out(t) for t in setup.score_tiles],
        "bonus_tiles": list(setup.bonus_tiles),
        "player_count": setup.player_count,
        "dropped_at_row": {k: int(v) for k, v in sorted(setup.dropped_at_row.items())},
    }


def setup_from_dict(raw: Mapping[str, Any]) -> GameSetup:
    return GameSetup(
        game_id=_require(raw, "game_id"),
        options=_options_in(_require(raw, "options")),
        factions=tuple(_require(raw, "factions")),
        score_tiles=tuple(_score_tile_in(t) for t in _require(raw, "score_tiles")),
        bonus_tiles=tuple(_require(raw, "bonus_tiles")),
        player_count=int(_require(raw, "player_count")),
        dropped_at_row={k: int(v) for k, v in raw.get("dropped_at_row", {}).items()},
    )


# --------------------------------------------------------------------------
# faction / hex / pending
# --------------------------------------------------------------------------


def _faction_out(fs: FactionState) -> dict[str, Any]:
    return {
        "name": fs.name,
        "coins": fs.coins,
        "workers": fs.workers,
        "priests": fs.priests,
        "priest_pool": fs.priest_pool,
        "power": [fs.power.bowl1, fs.power.bowl2, fs.power.bowl3],
        "vp": fs.vp,
        "shipping": fs.shipping,
        "dig_level": fs.dig_level,
        "teleport_level": fs.teleport_level,
        "buildings": {k: sorted(v) for k, v in sorted(fs.buildings.items())},
        "favors": list(fs.favors),
        "bonus": fs.bonus,
        "towns": list(fs.towns),
        "keys": fs.keys,
        "passed": fs.passed,
        "actions_used": sorted(fs.actions_used),
        "cult_blocked": sorted(fs.cult_blocked),
        "bridges_built": fs.bridges_built,
        "spades_available": fs.spades_available,
        "extra_actions": fs.extra_actions,
        "dropped": fs.dropped,
        "teleported_hex": fs.teleported_hex,
    }


def _faction_in(raw: Mapping[str, Any]) -> FactionState:
    bowls = _require(raw, "power")
    return FactionState(
        name=_require(raw, "name"),
        coins=int(raw["coins"]),
        workers=int(raw["workers"]),
        priests=int(raw["priests"]),
        priest_pool=int(raw["priest_pool"]),
        power=Power(int(bowls[0]), int(bowls[1]), int(bowls[2])),
        vp=int(raw["vp"]),
        shipping=int(raw["shipping"]),
        dig_level=int(raw["dig_level"]),
        teleport_level=int(raw["teleport_level"]),
        buildings={k: frozenset(v) for k, v in raw["buildings"].items()},
        favors=tuple(raw["favors"]),
        bonus=raw["bonus"],
        towns=tuple(raw["towns"]),
        keys=int(raw["keys"]),
        passed=bool(raw["passed"]),
        actions_used=frozenset(raw["actions_used"]),
        cult_blocked=frozenset(raw["cult_blocked"]),
        bridges_built=int(raw["bridges_built"]),
        spades_available=int(raw["spades_available"]),
        extra_actions=int(raw["extra_actions"]),
        dropped=bool(raw["dropped"]),
        teleported_hex=raw["teleported_hex"],
    )


def _hex_out(hx: HexState) -> list[Any]:
    return [hx.color, hx.building, hx.owner]


def _hex_in(raw: Sequence[Any]) -> HexState:
    return HexState(color=raw[0], building=raw[1], owner=raw[2])


def _pending_out(p: PendingDecision) -> dict[str, Any]:
    return {
        "faction": p.faction,
        "kind": p.kind,
        "amount": p.amount,
        "source": p.source,
        "options": list(p.options),
    }


def _pending_in(raw: Mapping[str, Any]) -> PendingDecision:
    return PendingDecision(
        faction=raw["faction"],
        kind=raw["kind"],
        amount=int(raw.get("amount", 0)),
        source=raw.get("source"),
        options=tuple(raw.get("options", ())),
    )


# --------------------------------------------------------------------------
# game state
# --------------------------------------------------------------------------


def game_to_dict(game: GameState) -> dict[str, Any]:
    return {
        "setup": setup_to_dict(game.setup),
        "round": game.round,
        "phase": game.phase.name,
        "turn_order": list(game.turn_order),
        "passed_order": list(game.passed_order),
        "active_index": game.active_index,
        "pending": [_pending_out(p) for p in game.pending],
        "hexes": {k: _hex_out(v) for k, v in sorted(game.hexes.items())},
        "bridges": sorted(sorted(pair) for pair in game.bridges),
        "factions": {k: _faction_out(v) for k, v in sorted(game.factions.items())},
        "cults": {
            k: dict(sorted(v.items())) for k, v in sorted(game.cults.items())
        },
        "cult_10": dict(sorted(game.cult_10.items())),
        "priest_slots": {
            k: list(v) for k, v in sorted(game.priest_slots.items())
        },
        "favors_pool": dict(sorted(game.favors_pool.items())),
        "towns_pool": dict(sorted(game.towns_pool.items())),
        "bonus_coins": dict(sorted(game.bonus_coins.items())),
        "power_actions_taken": sorted(game.power_actions_taken),
        "founded_towns": {
            k: [sorted(town) for town in v]
            for k, v in sorted(game.founded_towns.items())
        },
    }


def game_from_dict(raw: Mapping[str, Any]) -> GameState:
    try:
        phase = Phase[_require(raw, "phase")]
    except KeyError as exc:
        raise CodecError(f"unknown phase {raw.get('phase')!r}") from exc
    return GameState(
        setup=setup_from_dict(_require(raw, "setup")),
        round=int(_require(raw, "round")),
        phase=phase,
        turn_order=tuple(_require(raw, "turn_order")),
        passed_order=tuple(_require(raw, "passed_order")),
        active_index=int(_require(raw, "active_index")),
        pending=tuple(_pending_in(p) for p in _require(raw, "pending")),
        hexes={k: _hex_in(v) for k, v in _require(raw, "hexes").items()},
        bridges=frozenset(frozenset(pair) for pair in _require(raw, "bridges")),
        factions={
            k: _faction_in(v) for k, v in _require(raw, "factions").items()
        },
        cults={
            k: {ck: int(cv) for ck, cv in v.items()}
            for k, v in _require(raw, "cults").items()
        },
        cult_10=dict(_require(raw, "cult_10")),
        priest_slots={
            k: tuple(v) for k, v in _require(raw, "priest_slots").items()
        },
        favors_pool={k: int(v) for k, v in _require(raw, "favors_pool").items()},
        towns_pool={k: int(v) for k, v in _require(raw, "towns_pool").items()},
        bonus_coins={k: int(v) for k, v in _require(raw, "bonus_coins").items()},
        power_actions_taken=frozenset(_require(raw, "power_actions_taken")),
        founded_towns={
            k: tuple(frozenset(town) for town in v)
            for k, v in _require(raw, "founded_towns").items()
        },
    )


# --------------------------------------------------------------------------
# sim state
# --------------------------------------------------------------------------


def sim_to_dict(sim: SimState) -> dict[str, Any]:
    """The whole paused game: engine state plus the driver's turn bookkeeping.

    The driver fields matter as much as the board does -- ``fresh_taken``
    and ``free_used`` decide which moves the next offer contains, so a
    snapshot without them restores a *different* game.
    """
    return {
        "codec_version": CODEC_VERSION,
        "game": game_to_dict(sim.game),
        "fresh_taken": sim.fresh_taken,
        "free_used": sim.free_used,
        "prev_verb": sim.prev_verb,
        "decisions": sim.decisions,
        "income_marker": (
            None if sim.income_marker is None else list(sim.income_marker)
        ),
    }


def sim_from_dict(raw: Mapping[str, Any]) -> SimState:
    version = raw.get("codec_version")
    if version != CODEC_VERSION:
        raise CodecError(
            f"encoded state is codec v{version}, this code is v{CODEC_VERSION}"
        )
    marker = raw.get("income_marker")
    return SimState(
        game=game_from_dict(_require(raw, "game")),
        fresh_taken=bool(_require(raw, "fresh_taken")),
        free_used=int(_require(raw, "free_used")),
        prev_verb=raw.get("prev_verb"),
        decisions=int(_require(raw, "decisions")),
        income_marker=None if marker is None else (int(marker[0]), str(marker[1])),
    )


def sim_to_bytes(sim: SimState) -> bytes:
    """Canonical bytes for a paused game. Deterministic: equal states
    encode to equal bytes, so this doubles as a transposition key.
    """
    return orjson.dumps(sim_to_dict(sim))


def sim_from_bytes(payload: bytes) -> SimState:
    try:
        raw = orjson.loads(payload)
    except orjson.JSONDecodeError as exc:
        raise CodecError(f"not valid encoded-state JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise CodecError(f"encoded state must be a JSON object, got {type(raw).__name__}")
    return sim_from_dict(raw)
