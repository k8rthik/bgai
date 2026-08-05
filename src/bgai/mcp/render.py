"""Text rendering: commands (parser round-trip), state, moves, factual diffs.

`render_command` emits ledger-grammar text that `ledger_parser.parse_command`
parses back to the same fields -- moves cross the MCP boundary as strings in
both directions. `render_state`/`diff_factions` are strictly factual (the
spec's facts/judgment boundary): numbers and board contents, no evaluation.
"""

from __future__ import annotations

from bgai.data.ledger_parser import ParsedCommand
from bgai.engine.tm.board import base_board
from bgai.engine.tm.factions_data import CULTS
from bgai.engine.tm.state import GameState, active_faction
from bgai.engine.tm.tiles import BONUS_TILES, FAVOR_TILES, SCORE_TILES, TOWN_TILES

_BARE_VERBS = frozenset(
    {"wait", "done", "resign", "score_resources", "setup",
     "other_income_for_faction", "cult_income_for_faction", "all_income_for_faction"}
)


def render_command(cmd: ParsedCommand) -> str:
    """Ledger-grammar text for `cmd`; raises ValueError on a verb this
    renderer cannot express (never emit text that will not parse back).
    """
    v = cmd.verb
    if v in _BARE_VERBS:
        return v
    if v == "build":
        return f"build {cmd.loc}"
    if v == "upgrade":
        return f"upgrade {cmd.loc} to {cmd.building}"
    if v == "transform":
        return f"transform {cmd.loc} to {cmd.color}" if cmd.color else f"transform {cmd.loc}"
    if v == "dig":
        return f"dig {cmd.n1} {cmd.loc}" if cmd.loc else f"dig {cmd.n1}"
    if v == "send":
        return f"send p to {cmd.cult} for {cmd.n1}" if cmd.n1 else f"send p to {cmd.cult}"
    if v == "action":
        return f"action {cmd.tile}"
    if v == "pass":
        return f"pass {cmd.tile}" if cmd.tile else "pass"
    if v in ("leech", "decline"):
        amount = f" {cmd.n1}" if cmd.n1 is not None else ""
        source = f" from {cmd.target}" if cmd.target else ""
        return f"{v}{amount}{source}"
    if v == "convert":
        return f"convert {cmd.n1}{cmd.res1} to {cmd.n2}{cmd.res2}"
    if v == "burn":
        return f"burn {cmd.n1}"
    if v == "advance":
        return f"advance {cmd.reason}"
    if v == "bridge":
        return f"bridge {cmd.loc}:{cmd.loc2}"
    if v == "connect":
        return f"connect {cmd.loc}:{cmd.loc2}" if cmd.loc2 else f"connect {cmd.loc}"
    if v == "gain_favor":
        return f"+{cmd.tile}"
    if v == "gain_town":
        n = cmd.n1 or 1
        return f"+{n}{cmd.tile}" if n != 1 else f"+{cmd.tile}"
    if v == "gain_cult":
        return f"+{cmd.n1 or 1}{cmd.cult}"
    if v == "lose_cult":
        return f"-{cmd.n1 or 1}{cmd.cult}"
    if v == "lose_marker":
        return f"-{cmd.reason}"
    if v == "lose_spade":
        return f"-{cmd.n1 or 1}spade"
    if v == "score_vp":
        return f"+{cmd.n1}vp for {cmd.reason}"
    raise ValueError(f"unrenderable verb {v!r} ({cmd!r})")


def render_moves(moves: tuple[ParsedCommand, ...], faction: str) -> str:
    lines = [f"legal moves for {faction}:"]
    lines += [f"  {i}. {render_command(m)}" for i, m in enumerate(moves, start=1)]
    return "\n".join(lines)


def _mapping_text(mapping) -> str:
    return "/".join(f"{k}:{v}" for k, v in mapping.items()) if mapping else "-"


def _score_tile_text(tile) -> str:
    vp = ",".join(f"{k}+{n}" for k, n in tile.vp)
    income = ",".join(f"{n}{k}" for k, n in tile.cult_income)
    return f"{tile.vp_mode} {vp}; cult income {income} per {tile.req} {tile.cult}"


def _bonus_text(name: str) -> str:
    tile = BONUS_TILES[name]
    parts = [f"income {_mapping_text(tile.income)}"]
    if tile.special_action:
        parts.append(f"action {_mapping_text(tile.special_action)}")
    if tile.pass_vp:
        parts.append("pass_vp " + ",".join(f"{k}+{n}" for k, n in tile.pass_vp))
    if tile.passive:
        parts.append(f"passive {_mapping_text(tile.passive)}")
    return "; ".join(parts)


def _faction_block(state: GameState, name: str) -> list[str]:
    fs = state.factions[name]
    p = fs.power
    flags = []
    if fs.passed:
        flags.append("PASSED")
    if fs.dropped:
        flags.append("DROPPED")
    header = f"{name}{(' [' + ','.join(flags) + ']') if flags else ''}:"
    lines = [header]
    lines.append(
        f"  VP {fs.vp} | C {fs.coins} W {fs.workers} P {fs.priests}"
        f" (pool {fs.priest_pool}) | power {p.bowl1}/{p.bowl2}/{p.bowl3}"
        f" | ship {fs.shipping} dig {fs.dig_level}"
    )
    buildings = " ".join(
        f"{b}[{','.join(sorted(hexes))}]" for b, hexes in fs.buildings.items() if hexes
    )
    lines.append(f"  buildings: {buildings or 'none'}")
    extras = []
    if fs.favors:
        extras.append("favors " + ",".join(fs.favors))
    if fs.bonus:
        extras.append(f"bonus {fs.bonus} ({_bonus_text(fs.bonus)})")
    if fs.towns:
        extras.append("towns " + ",".join(fs.towns))
    if fs.keys:
        extras.append(f"keys {fs.keys}")
    if fs.spades_available:
        extras.append(f"spades {fs.spades_available}")
    if fs.extra_actions:
        extras.append(f"extra_actions {fs.extra_actions}")
    if extras:
        lines.append("  " + " | ".join(extras))
    return lines


def _pending_text(state: GameState) -> list[str]:
    lines = []
    for p in state.pending:
        if p.kind.startswith("driver_") or p.kind == "cultist_leech_watch":
            continue  # internal markers, not player-facing decisions
        detail = f"{p.faction}: {p.kind}"
        if p.amount:
            detail += f" amount={p.amount}"
        if p.source:
            detail += f" from={p.source}"
        lines.append(f"  {detail}")
    return ["pending decisions:"] + lines if lines else []


def render_state(state: GameState, viewer: str | None = None) -> str:
    """Compact factual state text. `viewer` only reorders the faction blocks
    (viewer first) -- TM has no hidden information.
    """
    lines: list[str] = []
    lines.append(
        f"round {state.round}/6 | phase {state.phase.name}"
        f" | turn order {' > '.join(state.turn_order)}"
        f" | active: {active_faction(state)}"
    )
    tile_names = _score_tile_names(state)
    for i, (name, tile) in enumerate(zip(tile_names, state.setup.score_tiles, strict=True)):
        marker = " <- current" if i + 1 == state.round else ""
        lines.append(f"  round {i + 1} tile {name}: {_score_tile_text(tile)}{marker}")

    lines += _pending_text(state)

    factions = list(state.setup.factions)
    if viewer in factions:
        factions.remove(viewer)
        factions.insert(0, viewer)
    for name in factions:
        lines += _faction_block(state, name)

    lines.append("cults (position; 10-slot holder):")
    for cult in CULTS:
        positions = " ".join(f"{f}:{state.cults[f][cult]}" for f in state.setup.factions)
        ten = state.cult_10.get(cult)
        slots = state.priest_slots[cult]
        taken = sum(1 for s in slots if s is not None)
        lines.append(
            f"  {cult}: {positions} | 10-slot: {ten or 'open'}"
            f" | priest slots used {taken}/{len(slots)}"
        )

    board = base_board()
    occupied = [
        f"  {key} {hx.color} {hx.building} {hx.owner}"
        for key, hx in sorted(state.hexes.items())
        if hx.building is not None
    ]
    lines.append("board (occupied hexes):")
    lines += occupied or ["  none"]
    terraformed = [
        f"  {key} now {hx.color} (was {board.hexes[key].color})"
        for key, hx in sorted(state.hexes.items())
        if hx.building is None and hx.color != board.hexes[key].color
    ]
    if terraformed:
        lines.append("terraformed empty hexes:")
        lines += terraformed

    pools = []
    favors_left = {t: n for t, n in state.favors_pool.items() if n > 0}
    towns_left = {t: n for t, n in state.towns_pool.items() if n > 0}
    pools.append("favor tiles left: " + (_mapping_text(favors_left)))
    pools.append("town tiles left: " + (_mapping_text(towns_left)))
    coins = {t: n for t, n in state.bonus_coins.items() if n > 0}
    if coins:
        pools.append("bonus-tile coins: " + _mapping_text(coins))
    if state.power_actions_taken:
        pools.append("power actions taken: " + ",".join(sorted(state.power_actions_taken)))
    lines += pools
    return "\n".join(lines)


def _score_tile_names(state: GameState) -> list[str]:
    names = []
    for tile in state.setup.score_tiles:
        name = next((n for n, t in SCORE_TILES.items() if t == tile), "SCORE?")
        names.append(name)
    return names


def diff_factions(before: GameState, after: GameState) -> str:
    """Per-faction factual deltas between two states: resources, VP, power,
    cults, tracks, buildings gained/lost, plus new pending decisions and
    any phase/round change. Arithmetic only -- no evaluation.
    """
    lines: list[str] = []
    if (before.round, before.phase) != (after.round, after.phase):
        lines.append(
            f"phase: round {before.round} {before.phase.name}"
            f" -> round {after.round} {after.phase.name}"
        )
    for name in before.setup.factions:
        b, a = before.factions[name], after.factions[name]
        deltas = []
        for label, get in (
            ("VP", lambda f: f.vp), ("C", lambda f: f.coins), ("W", lambda f: f.workers),
            ("P", lambda f: f.priests), ("ship", lambda f: f.shipping),
            ("dig", lambda f: f.dig_level), ("keys", lambda f: f.keys),
        ):
            d = get(a) - get(b)
            if d:
                deltas.append(f"{label} {d:+d}")
        if b.power != a.power:
            deltas.append(
                f"power {b.power.bowl1}/{b.power.bowl2}/{b.power.bowl3}"
                f" -> {a.power.bowl1}/{a.power.bowl2}/{a.power.bowl3}"
            )
        for cult in CULTS:
            d = after.cults[name][cult] - before.cults[name][cult]
            if d:
                deltas.append(f"{cult} {d:+d}")
        for building, hexes in a.buildings.items():
            gained = hexes - b.buildings.get(building, frozenset())
            lost = b.buildings.get(building, frozenset()) - hexes
            if gained:
                deltas.append(f"+{building}@{','.join(sorted(gained))}")
            if lost:
                deltas.append(f"-{building}@{','.join(sorted(lost))}")
        new_favors = set(a.favors) - set(b.favors)
        if new_favors:
            deltas.append("+favors " + ",".join(sorted(new_favors)))
        new_towns = set(a.towns) - set(b.towns)
        if new_towns:
            deltas.append("+towns " + ",".join(sorted(new_towns)))
        if a.passed and not b.passed:
            deltas.append("passed")
        if deltas:
            lines.append(f"{name}: " + ", ".join(deltas))
    new_pending = [p for p in after.pending if p not in before.pending]
    for p in new_pending:
        if p.kind.startswith("driver_") or p.kind == "cultist_leech_watch":
            continue
        detail = f"new decision for {p.faction}: {p.kind}"
        if p.amount:
            detail += f" amount={p.amount}"
        if p.source:
            detail += f" from={p.source}"
        lines.append(detail)
    return "\n".join(lines) if lines else "no factual changes"
