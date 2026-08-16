"""Live play REPL: you enter the other seats' moves, the agent answers.

    uv run python -u tools/live_agent.py \
        --checkpoint data/checkpoints/selfplay_leg4/current.pt \
        --agent-seat 0 --sims 512 [--setup <corpus_game_id>] [--seed 7]

Runs the SIM driver (the one MCTS natively searches), so the agent plays
at full evaluated strength with tree reuse between its turns. Human
seats submit snellman ledger syntax ("build E5", "upgrade F6 to TP",
"pass BON3", "leech 2 from witches", "action ACT5. transform G3" one
command per prompt); moves are matched against the engine's legal offer,
so anything accepted is guaranteed legal. Type "moves" to list the
current offer, "state" for a VP/resource summary, "quit" to stop.

Timing: ~0.8 s per agent move at 512 simulations (tree reuse included).
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

from bgai.agents.mcts import MCTSAgent
from bgai.arena.driver import advance, decision, new_game
from bgai.arena.setup_factory import fresh_setup
from bgai.data.ledger_parser import parse_command
from bgai.engine.tm.legal_match import match_index
from bgai.engine.tm.setup import load_setup
from bgai.mcp.render import render_command


def _status(sim) -> str:
    game = sim.game
    parts = []
    for f in game.setup.factions:
        fs = game.factions[f]
        parts.append(
            f"{f}: {fs.vp}VP {fs.coins}C {fs.workers}W {fs.priests}P"
        )
    return f"round {game.round} [{game.phase.name}]  " + " | ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--agent-seat", type=int, default=0, help="seat index 0-3 the agent plays")
    parser.add_argument("--sims", type=int, default=512)
    parser.add_argument("--setup", default=None, help="corpus game id for the table setup")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    setup = load_setup(args.setup) if args.setup else fresh_setup(seed=args.seed)
    agent = MCTSAgent(
        Path(args.checkpoint), simulations=args.sims, top_k=8, max_depth=48, leaf_batch=16
    )
    rng = random.Random(args.seed)
    sim = new_game(setup)
    agent_faction = setup.factions[args.agent_seat]
    print(f"table: {setup.factions} | agent plays seat {args.agent_seat} ({agent_faction})")

    while (pending := decision(sim)) is not None:
        faction, offer = pending
        if not offer:
            print(f"!! zero-move decision for {faction}; aborting")
            break
        if len(offer) == 1:
            agent.note_advance(offer[0])
            sim = advance(sim, offer[0])
            continue
        if faction == agent_faction:
            t0 = time.perf_counter()
            choice = agent.choose_sim(sim, faction, offer, rng)
            agent.note_advance(choice)
            print(f">> AGENT ({faction}, {time.perf_counter() - t0:.1f}s): {render_command(choice)}")
            sim = advance(sim, choice)
            continue
        while True:
            text = input(f"[{faction}] move> ").strip()
            if text == "quit":
                return
            if text == "state":
                print(_status(sim))
                continue
            if text == "moves":
                for i, m in enumerate(offer):
                    print(f"  {i:3d}: {render_command(m)}")
                continue
            if text.isdigit() and int(text) < len(offer):
                choice = offer[int(text)]
                break
            cmd = parse_command(text)
            if cmd is None:
                print("  ?? could not parse; try 'moves' to list options or an index")
                continue
            idx = match_index(offer, cmd)
            if idx is None:
                print("  !! not a legal move here; 'moves' lists the offer")
                continue
            choice = offer[idx]
            break
        agent.note_advance(choice)
        sim = advance(sim, choice)

    print("game over.")
    print(_status(sim))


if __name__ == "__main__":
    main()
