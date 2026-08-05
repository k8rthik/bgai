"""Does sampling cause the development deficit?

The agent samples its policy (T=1.0) because that beat argmax on
PLACEMENT (decision D5.6). But placement is relative -- it says nothing
about whether the agent is developing its economy. The end-state
diagnostic showed our agents build ~29% of the structures humans build
and essentially never found towns, so the question is whether sampling
buys relative wins at the cost of absolute play.

Also counts main actions taken per round, to test the simplest
explanation for under-building: passing too early.

Usage: uv run python tools/diagnose_temp.py [n_games]
"""

from __future__ import annotations

import random
import statistics
import sys
from collections import Counter
from pathlib import Path

from bgai.agents.imitation import ImitationAgent
from bgai.arena.driver import advance, decision, new_game
from bgai.arena.setups import sample_setup
from bgai.engine.tm.factions_data import CULTS

CKPT = Path("data/checkpoints/imitation/checkpoint.pt")
BUILDINGS = ("D", "TP", "TE", "SH", "SA")


def play(temperature: float, n_games: int) -> dict[str, float]:
    agent = ImitationAgent(CKPT, name="a", temperature=temperature)
    setup_rng = random.Random(31)
    vps, structures, towns, cults, verbs = [], [], [], [], Counter()
    actions_per_round: list[float] = []
    for i in range(n_games):
        setup = sample_setup(setup_rng)
        sim = new_game(setup)
        rng = random.Random(1000 + i)
        per_round: Counter = Counter()
        while (pending := decision(sim)) is not None:
            faction, offer = pending
            choice = agent.choose(sim.game, faction, offer, rng)
            verbs[choice.verb] += 1
            if choice.verb in ("build", "upgrade", "action", "dig", "send", "advance"):
                per_round[sim.game.round] += 1
            sim = advance(sim, choice)
        state = sim.game
        for name, fs in state.factions.items():
            vps.append(fs.vp)
            structures.append(sum(len(fs.buildings.get(b, ())) for b in BUILDINGS))
            towns.append(len(fs.towns))
            cults.append(sum(state.cults[name][c] for c in CULTS))
        rounds = [r for r in per_round if 1 <= r <= 6]
        if rounds:
            actions_per_round.append(statistics.fmean(per_round[r] for r in rounds) / 4)
    total_verbs = sum(verbs.values())
    return {
        "vp": statistics.fmean(vps),
        "structures": statistics.fmean(structures),
        "towns": statistics.fmean(towns),
        "cult_sum": statistics.fmean(cults),
        "actions_per_player_per_round": statistics.fmean(actions_per_round),
        "pass_share": verbs["pass"] / total_verbs,
        "convert_share": verbs["convert"] / total_verbs,
        "burn_share": verbs["burn"] / total_verbs,
        "build_share": verbs["build"] / total_verbs,
    }


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    print(f"{'metric':30} {'T=0 (argmax)':>14} {'T=1 (sample)':>14}   human")
    cold = play(0.0, n)
    warm = play(1.0, n)
    human = {
        "vp": 111.7, "structures": 12.9, "towns": 2.33, "cult_sum": 21.4,
        "actions_per_player_per_round": float("nan"),
    }
    for key in cold:
        h = human.get(key)
        h_str = f"{h:.2f}" if isinstance(h, float) and h == h else "-"
        print(f"{key:30} {cold[key]:14.3f} {warm[key]:14.3f}   {h_str}")


if __name__ == "__main__":
    main()
