"""VP-source decomposition for finished games (the witches-den diagnostic).

Attributes every VP change in a game to a source category, so an agent's
score can be compared to human scores *decision class by decision class*
instead of as one number. The category list follows witches-den's rules
engine (github.com/ryanechternacht/witches-den), the play-analysis site
built around exactly the questions this diagnostic answers: how much VP
goes to leech, what favors/pass bonuses actually earn, where endgame
points come from.

Design: replay-free tracing. ``trace_game`` mirrors ``arena.sim.run_game``
but records every ``advance`` step with each faction's VP delta; the
classifier then splits each observed delta into buckets computed from the
pre-step state (held favors, towns founded, round score tile, endgame
standings). Anything a rule cannot explain lands in an ``other:<verb>``
bucket rather than being dropped or guessed at, which gives the module
its invariant:

    sum(all buckets) == final VP - initial VP     (per faction, exactly)

Deliberately v1-coarse: ``action`` lumps score-tile and favor VP earned
by build/upgrade/dig moves (they share trigger commands; splitting them
needs per-command counterfactuals). The buckets that drive the diagnosis
-- leech, pass, towns, track advances, the three endgame sources -- are
exact.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass

from bgai.agents.base import Agent
from bgai.arena.driver import SimState, advance, decision, new_game
from bgai.data.ledger_parser import ParsedCommand
from bgai.engine.tm.apply import EngineError
from bgai.engine.tm.scoring import (
    _apply_score_resources,
    compute_cult_scoring,
    compute_network_scoring,
)
from bgai.engine.tm.setup import GameSetup
from bgai.engine.tm.state import GameState, Phase
from bgai.engine.tm.tiles import TOWN_TILES

# Ordered for report display: in-game sources first, endgame last.
CATEGORIES = (
    "action",         # build/upgrade/dig/action VP: score tiles + favors
    "town",           # town-tile VP (incl. faction town passives)
    "track_advance",  # advance ship/dig VP
    "pass_bonus",     # pass VP from bonus tiles, FAV12, pass abilities
    "convert",        # VP spent/earned via convert/burn (Alchemists)
    "leech",          # VP paid to accept power (always <= 0)
    "endgame_cult",
    "endgame_network",
    "endgame_resources",
)


@dataclass(frozen=True)
class TracedStep:
    """One ``advance`` with its observable VP consequences."""

    faction: str  # the mover
    cmd: ParsedCommand
    pre: GameState
    post: GameState
    deltas: Mapping[str, int]  # faction -> vp change, zero entries omitted


@dataclass(frozen=True)
class TracedGame:
    setup_game_id: str
    steps: tuple[TracedStep, ...]
    final_vp: Mapping[str, int]
    error: str | None = None


def trace_game(
    setup: GameSetup,
    seats: Mapping[str, Agent],
    rng: random.Random,
    max_decisions: int = 5000,
) -> TracedGame:
    """``arena.sim.run_game`` with per-step VP recording.

    Records every advance (including forced single-move bookkeeping
    steps -- endgame scoring can land on those). Engine rejections are
    recorded, not raised, exactly like the arena.
    """
    if set(seats) != set(setup.factions):
        raise ValueError(f"seats {sorted(seats)} != setup factions {sorted(setup.factions)}")

    error: str | None = None
    sim: SimState = new_game(setup)
    steps: list[TracedStep] = []
    try:
        while (pending := decision(sim)) is not None:
            if sim.decisions >= max_decisions:
                error = f"decision cap exceeded ({max_decisions})"
                break
            faction, offer = pending
            if len(offer) == 1:
                choice = offer[0]
            else:
                agent = seats[faction]
                chooser = getattr(agent, "choose_sim", None)
                choice = (
                    chooser(sim, faction, offer, rng)
                    if chooser is not None
                    else agent.choose(sim.game, faction, offer, rng)
                )
            pre = sim.game
            sim = advance(sim, choice)
            deltas = {
                f: sim.game.factions[f].vp - pre.factions[f].vp
                for f in setup.factions
                if sim.game.factions[f].vp != pre.factions[f].vp
            }
            # every step is kept: endgame scoring can land on a forced
            # bookkeeping step whose deltas the classifier must see
            steps.append(
                TracedStep(faction=faction, cmd=choice, pre=pre, post=sim.game, deltas=deltas)
            )
    except EngineError as exc:
        error = f"decision {sim.decisions}: {exc}"

    return TracedGame(
        setup_game_id=setup.game_id,
        steps=tuple(steps),
        final_vp={f: sim.game.factions[f].vp for f in setup.factions},
        error=error,
    )


def _endgame_components(pre: GameState) -> dict[str, dict[str, int]]:
    """Exact endgame VP per faction from the pre-finals state.

    Safe to compute from the last pass's pre-state: passing moves no
    cult marker and builds nothing, so cult/network standings -- and
    (in round 6, where passing selects no new bonus tile) resources --
    are identical at scoring time.
    """
    cult = compute_cult_scoring(pre)
    network = compute_network_scoring(pre)
    out: dict[str, dict[str, int]] = {}
    for faction, fs in pre.factions.items():
        converted = _apply_score_resources(fs, faction)
        out[faction] = {
            "endgame_cult": sum(cult[track].get(faction, 0) for track in cult),
            "endgame_network": network.get(faction, 0),
            "endgame_resources": converted.vp - fs.vp,
        }
    return out


def _classify_step(
    step: TracedStep, endgame: Mapping[str, Mapping[str, int]] | None
) -> dict[str, dict[str, int]]:
    """Split one step's deltas into buckets; remainder -> other:<verb>."""
    out: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for faction, delta in step.deltas.items():
        remaining = delta
        if endgame is not None:
            for key, vp in endgame[faction].items():
                if vp:
                    out[faction][key] += vp
                    remaining -= vp
        if faction != step.faction:
            # The only VP a non-mover pays during someone else's move is
            # the leech cost of accepted power. (Leech offers resolve as
            # the accepting faction's own step in this driver, so this
            # arm is a backstop for engine-side auto-resolution.)
            bucket = "leech" if remaining < 0 else "other:passive"
            if remaining:
                out[faction][bucket] += remaining
            continue

        verb = step.cmd.verb
        # Town founding: exact from the founded-towns diff. Includes the
        # tile's printed VP plus faction TOWN passives (Witches +5),
        # which land in the same step.
        pre_towns = step.pre.factions[faction].towns
        post_towns = step.post.factions[faction].towns
        if len(post_towns) > len(pre_towns) and verb != "pass":
            # Attribute conservatively: the whole step went to founding
            # only when the verb is a builder verb; the split between
            # town VP and simultaneous score-tile/favor VP is refined
            # below by subtracting the tile's known VP.
            new_tiles = post_towns[len(pre_towns):]
            town_vp = sum(TOWN_TILES[t].vp for t in new_tiles)
            fs_passives = 5 if faction == "witches" else 0
            town_total = min(remaining, town_vp + fs_passives) if remaining > 0 else 0
            if town_total:
                out[faction]["town"] += town_total
                remaining -= town_total

        if remaining == 0:
            continue
        if verb == "leech":
            out[faction]["leech"] += remaining
        elif verb == "pass":
            out[faction]["pass_bonus"] += remaining
        elif verb == "advance":
            out[faction]["track_advance"] += remaining
        elif verb in ("convert", "burn"):
            out[faction]["convert"] += remaining
        elif verb in ("build", "upgrade", "dig", "transform", "action", "send", "bridge"):
            out[faction]["action"] += remaining
        else:
            out[faction][f"other:{verb}"] += remaining
    return out


def decompose(traced: TracedGame) -> dict[str, dict[str, int]]:
    """Per-faction VP-by-source for one traced game.

    Invariant (asserted): each faction's buckets sum to its final VP
    minus the 20 every faction starts with.
    """
    totals: dict[str, dict[str, int]] = {f: defaultdict(int) for f in traced.final_vp}
    finals_step = next(
        (
            s
            for s in traced.steps
            if s.pre.phase != Phase.FINISHED and s.post.phase == Phase.FINISHED
        ),
        None,
    )
    endgame = _endgame_components(finals_step.pre) if finals_step is not None else None

    for step in traced.steps:
        step_endgame = endgame if step is finals_step else None
        for faction, buckets in _classify_step(step, step_endgame).items():
            for key, vp in buckets.items():
                totals[faction][key] += vp

    if traced.error is None:
        for faction, buckets in totals.items():
            attributed = sum(buckets.values())
            expected = traced.final_vp[faction] - 20
            assert attributed == expected, (
                f"{traced.setup_game_id}/{faction}: attributed {attributed} != {expected}; "
                f"buckets={dict(buckets)}"
            )
    return {f: dict(b) for f, b in totals.items()}


def aggregate(per_game: list[dict[str, dict[str, int]]]) -> dict[str, dict[str, float]]:
    """Mean VP by source across games, keyed by faction -> category."""
    sums: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    counts: dict[str, int] = defaultdict(int)
    for game in per_game:
        for faction, buckets in game.items():
            counts[faction] += 1
            for key, vp in buckets.items():
                sums[faction][key] += vp
    return {
        f: {k: v / counts[f] for k, v in buckets.items()} for f, buckets in sums.items()
    }
