"""What, concretely, does the imitation agent fail to do?

Calibration (docs/decisions.md C1/C2) established that our agents score
~65 VP against ~136 for the humans they imitate. That is a verdict, not
a diagnosis. This compares END-OF-GAME board state between agent games
and real corpus games on the same map, so the deficit can be attributed
to specific mechanics: buildings not built, towns not founded, cults not
climbed, tracks not advanced.

Usage:
    uv run python tools/diagnose_gap.py [n_agent_games] [n_corpus_games]
"""

from __future__ import annotations

import random
import statistics
import sys
from pathlib import Path

import polars as pl

from bgai.agents.imitation import ImitationAgent
from bgai.arena.driver import advance, decision, new_game
from bgai.arena.setups import clean_game_ids, sample_setup
from bgai.engine.tm.factions_data import CULTS
from bgai.engine.tm.replay import replay_game
from bgai.engine.tm.state import GameState

CKPT = Path("data/checkpoints/imitation/checkpoint.pt")
BUILDINGS = ("D", "TP", "TE", "SH", "SA")


def snapshot(state: GameState) -> list[dict[str, float]]:
    """Per-faction end-of-game summary."""
    rows = []
    for name, fs in state.factions.items():
        rows.append(
            {
                "vp": fs.vp,
                **{b: len(fs.buildings.get(b, ())) for b in BUILDINGS},
                "towns": len(fs.towns),
                "favors": len(fs.favors),
                "cult_sum": sum(state.cults[name][c] for c in CULTS),
                "cult_max": max(state.cults[name][c] for c in CULTS),
                "shipping": fs.shipping,
                "dig": fs.dig_level,
                "power_total": fs.power.bowl1 + fs.power.bowl2 + fs.power.bowl3,
            }
        )
    return rows


def agent_games(n: int) -> list[dict[str, float]]:
    agent = ImitationAgent(CKPT, name="imitation")
    rng = random.Random(31)
    rows: list[dict[str, float]] = []
    for i in range(n):
        setup = sample_setup(rng)
        sim = new_game(setup)
        game_rng = random.Random(1000 + i)
        while (pending := decision(sim)) is not None:
            faction, offer = pending
            sim = advance(sim, agent.choose(sim.game, faction, offer, game_rng))
        rows.extend(snapshot(sim.game))
    return rows


def corpus_games(n: int) -> list[dict[str, float]]:
    moves_df = pl.read_parquet("data/datasets/moves.parquet")
    deltas_df = pl.read_parquet("data/datasets/deltas.parquet")
    ids = list(clean_game_ids())
    rng = random.Random(17)
    rows: list[dict[str, float]] = []
    picked = rng.sample(ids, n)
    for game_id in picked:
        captured: list[GameState] = []

        def hook(state, faction, cmd):
            captured.append(state)

        result = replay_game(game_id, moves_df, deltas_df, on_decision=hook)
        if result.error is not None or not captured:
            continue
        rows.extend(snapshot(captured[-1]))
    return rows


def main() -> None:
    n_agent = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    n_corpus = int(sys.argv[2]) if len(sys.argv) > 2 else 60

    ours = agent_games(n_agent)
    theirs = corpus_games(n_corpus)
    print(f"agent factions: {len(ours)}   corpus factions: {len(theirs)}\n")
    keys = ["vp", *BUILDINGS, "towns", "favors", "cult_sum", "cult_max",
            "shipping", "dig", "power_total"]
    print(f"{'metric':12} {'agent':>8} {'human':>8} {'ratio':>7}")
    for key in keys:
        a = statistics.fmean(r[key] for r in ours)
        h = statistics.fmean(r[key] for r in theirs)
        ratio = a / h if h else float("nan")
        print(f"{key:12} {a:8.2f} {h:8.2f} {ratio:7.2f}")


if __name__ == "__main__":
    main()
