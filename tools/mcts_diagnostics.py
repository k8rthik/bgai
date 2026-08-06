"""D6.9: separate H1 (value-error amplification) from H2 (too-shallow
search) for Phase 6's tight zero -- 64-sim max^n MCTS does not beat the
imitation policy it searches over (docs/decisions.md D6.5-D6.8).

Diagnostic A -- blend the leaf value. ``agents/leaf_eval.py`` mixes the
learned value head with an engine-computed VP-share projection that is
exact by construction at any state (``scoring.projected_vp``). Sweep
``value_blend_w`` in {0, 0.25, 0.5, 1.0} at a fixed 64 simulations; w=0 is
the control (today's behaviour). If search suddenly beats the policy at
w>0, H1 is confirmed.

Diagnostic B -- depth ladder. Same value head (w=0), simulations in
{64, 256, 1024}, same seed (hence the same setup sequence) across rungs.
Monotone degradation with depth -> H1. Flat-then-improving -> H2.

Both reuse the exact paired protocol D6.5-D6.7 used: both agents seated
TWICE per game via the mirrored 4-seat rotation (``arena/series.py``,
``arena/rotation.py``), so setup and seat effects cancel within a game.
"rank-sum diff" below is, per game, mean(rank of agent A's two seats) -
mean(rank of agent B's two seats); 0-based competition rank (0 = best),
so negative favours A. Reported as mean +/- standard error, with t.

Usage:
    uv run python tools/mcts_diagnostics.py time --tables 2 --sims 64,256,1024
    uv run python tools/mcts_diagnostics.py a --tables 50
    uv run python tools/mcts_diagnostics.py b --tables 50,38,10
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

from bgai.agents.imitation import ImitationAgent
from bgai.agents.mcts import MCTSAgent
from bgai.arena.series import run_series
from bgai.arena.sim import GameResult

CKPT = Path("data/checkpoints/imitation/checkpoint.pt")
SEED = 20260806


def _paired_stat(results: tuple[GameResult, ...], a_name: str, b_name: str) -> dict[str, float]:
    diffs: list[float] = []
    a_vp: list[int] = []
    b_vp: list[int] = []
    a_rank: list[int] = []
    b_rank: list[int] = []
    errors = 0
    for r in results:
        if r.error is not None:
            errors += 1
            continue
        ranks_a = [r.ranks[f] for f, name in r.seats.items() if name == a_name]
        ranks_b = [r.ranks[f] for f, name in r.seats.items() if name == b_name]
        diffs.append(sum(ranks_a) / len(ranks_a) - sum(ranks_b) / len(ranks_b))
        a_rank.extend(ranks_a)
        b_rank.extend(ranks_b)
        a_vp.extend(r.vps[f] for f, name in r.seats.items() if name == a_name)
        b_vp.extend(r.vps[f] for f, name in r.seats.items() if name == b_name)

    n = len(diffs)
    mean = sum(diffs) / n if n else float("nan")
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1) if n > 1 else 0.0
    se = math.sqrt(var / n) if n else float("nan")
    t = mean / se if se > 0 else float("nan")
    return {
        "n": n,
        "errors": errors,
        "mean_diff": mean,
        "se": se,
        "t": t,
        "a_mean_rank": sum(a_rank) / len(a_rank) if a_rank else float("nan"),
        "b_mean_rank": sum(b_rank) / len(b_rank) if b_rank else float("nan"),
        "a_mean_vp": sum(a_vp) / len(a_vp) if a_vp else float("nan"),
        "b_mean_vp": sum(b_vp) / len(b_vp) if b_vp else float("nan"),
    }


def _report(label: str, stat: dict[str, float], elapsed: float) -> None:
    print(
        f"{label:28} n={stat['n']:4} err={stat['errors']:<3} "
        f"diff={stat['mean_diff']:+.3f} +/- {stat['se']:.3f}  t={stat['t']:+.2f}  "
        f"a_rank={stat['a_mean_rank']:.3f} b_rank={stat['b_mean_rank']:.3f} "
        f"a_vp={stat['a_mean_vp']:.1f} b_vp={stat['b_mean_vp']:.1f}  "
        f"({elapsed:.0f}s, {elapsed / max(stat['n'] + stat['errors'], 1):.2f}s/game)"
    )


def _paired_run(a, b, n_tables: int, seed: int = SEED):
    agents = {a.name: a, b.name: b}
    base_seats = (a.name, b.name, a.name, b.name)
    t0 = time.time()
    series = run_series(agents=agents, base_seats=base_seats, n_tables=n_tables, seed=seed)
    elapsed = time.time() - t0
    stat = _paired_stat(series.results, a.name, b.name)
    return stat, elapsed


def diagnostic_a(tables: int) -> None:
    print(f"# Diagnostic A -- leaf value blend sweep, 64 sims, {tables} tables ({tables*4} games/cell)")
    for w in (0.0, 0.25, 0.5, 1.0):
        mcts = MCTSAgent(CKPT, name=f"mcts_w{w:g}", simulations=64, value_blend_w=w)
        imitation = ImitationAgent(CKPT, name="imitation")
        stat, elapsed = _paired_run(mcts, imitation, tables)
        _report(f"w={w:g}", stat, elapsed)


def diagnostic_b(table_counts: list[int]) -> None:
    sims_ladder = (64, 256, 1024)
    print(f"# Diagnostic B -- depth ladder, w=0, tables={table_counts}, SAME seed each rung")
    for sims, tables in zip(sims_ladder, table_counts, strict=True):
        mcts = MCTSAgent(CKPT, name=f"mcts_{sims}", simulations=sims, value_blend_w=0.0)
        imitation = ImitationAgent(CKPT, name="imitation")
        stat, elapsed = _paired_run(mcts, imitation, tables)
        _report(f"sims={sims}", stat, elapsed)


def timing(tables: int, sims_list: list[int]) -> None:
    print(f"# timing calibration, {tables} tables ({tables*4} games) per rung")
    for sims in sims_list:
        mcts = MCTSAgent(CKPT, name=f"mcts_{sims}", simulations=sims)
        imitation = ImitationAgent(CKPT, name="imitation")
        _, elapsed = _paired_run(mcts, imitation, tables, seed=1)
        print(f"sims={sims:5} {elapsed:7.1f}s total  {elapsed/(tables*4):6.3f}s/game")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="which", required=True)

    p_a = sub.add_parser("a")
    p_a.add_argument("--tables", type=int, default=50)

    p_b = sub.add_parser("b")
    p_b.add_argument("--tables", type=str, default="50,38,10")

    p_t = sub.add_parser("time")
    p_t.add_argument("--tables", type=int, default=2)
    p_t.add_argument("--sims", type=str, default="64,256,1024")

    args = parser.parse_args()
    if args.which == "a":
        diagnostic_a(args.tables)
    elif args.which == "b":
        diagnostic_b([int(x) for x in args.tables.split(",")])
    elif args.which == "time":
        timing(args.tables, [int(x) for x in args.sims.split(",")])


if __name__ == "__main__":
    main()
