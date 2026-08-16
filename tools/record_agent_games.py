"""Record agent games into the corpus-schema parquet store.

    uv run python -u tools/record_agent_games.py \
        --checkpoint data/checkpoints/selfplay_leg4/current.pt \
        --games 200 --sims 256 --workers 10 --batch leg4_256

Traces games in a worker pool and appends them to data/agent_games/
(moves/deltas/meta part files, corpus-shaped). Stats then come from
polars queries, never from re-tracing.
"""

from __future__ import annotations

import argparse
import random
from multiprocessing import get_context
from pathlib import Path

_W: dict[str, object] = {}


def _init(checkpoint: str, sims: int) -> None:
    import torch

    from bgai.agents.mcts import MCTSAgent

    torch.set_num_threads(1)
    _W["agent"] = MCTSAgent(
        Path(checkpoint), simulations=sims, top_k=8, max_depth=48, leaf_batch=16
    )


def _one(seed: int):
    from bgai.arena.setups import sample_setup
    from bgai.knowledge.vp_decompose import trace_game

    rng = random.Random(seed)
    setup = sample_setup(rng)
    agent = _W["agent"]
    agent.reset_tree()  # type: ignore[union-attr]
    seats = {f: agent for f in setup.factions}
    traced = trace_game(setup, seats, rng)  # type: ignore[arg-type]
    return traced if traced.error is None else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--sims", type=int, default=256)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch", required=True, help="batch name; also the game-id prefix")
    parser.add_argument("--store", type=Path, default=Path("data/agent_games"))
    args = parser.parse_args()

    from bgai.knowledge.game_store import append_batch

    ctx = get_context("spawn")
    games = []
    with ctx.Pool(
        processes=args.workers, initializer=_init, initargs=(args.checkpoint, args.sims)
    ) as pool:
        seeds = [args.seed * 1_000_000 + i for i in range(args.games)]
        for i, traced in enumerate(pool.imap_unordered(_one, seeds)):
            if traced is not None:
                games.append((traced, f"{args.batch}_{len(games):04d}"))
            if (i + 1) % 20 == 0:
                print(f"{i + 1}/{args.games} traced ({len(games)} ok)", flush=True)
    append_batch(games, store=args.store, batch_name=args.batch)
    print(f"wrote {len(games)} games to {args.store} as batch {args.batch!r}", flush=True)


if __name__ == "__main__":
    main()
