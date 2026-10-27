"""Why do bots found fewer towns? Cluster development, bots vs humans.

    uv run python tools/town_diagnosis.py \
        --checkpoint data/checkpoints/champion_ftcos_it14/current.pt --games 300 --workers 18

Plays bot games in-engine (the arena's own decision/advance loop, at the
champion eval config) and replays the human games on the same setups
(``replay_game``); either way every faction is sampled at each of its
passes, i.e. at the end of each round. (Recorded bot stores can't be
replayed: they carry no income rows, so ``replay_game`` never pays income
and the states drift from round 1.) Each sample holds:

- towns founded so far;
- the best unfinished cluster -- connected buildings not part of a founded
  town -- as (power value, structure count) against the faction's town
  threshold (7, or 6 with FAV5; 4 structures, SA counting twice);
- the number of separate clusters.

"Near town" = an unfinished cluster with >= 3 structures and power within 2
of the threshold. Conversion = of faction-rounds ending with a near town,
the share that founded a town by the end of the next round. Reaching near
towns as often as humans but converting less points at valuing the finish;
reaching them less points at how clusters get built.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from multiprocessing import get_context
from pathlib import Path

import polars as pl

_W: dict[str, object] = {}
NEAR_POWER_SLACK = 2
NEAR_MIN_STRUCTURES = 3


def _init(moves_path: str, deltas_path: str, ids: list[str]) -> None:
    moves = pl.scan_parquet(moves_path).filter(pl.col("game_id").is_in(ids)).collect()
    deltas = pl.scan_parquet(deltas_path).filter(pl.col("game_id").is_in(ids)).collect()
    _W.update(moves=moves, deltas=deltas)


def _sample(state, faction: str) -> dict:
    from bgai.engine.tm.connectivity import clusters
    from bgai.engine.tm.towns import _cluster_power_and_count, _town_threshold

    fs = state.factions[faction]
    founded = state.founded_towns.get(faction, ())
    open_clusters = [c for c in clusters(state, faction) if not any(c & f for f in founded)]
    stats = [_cluster_power_and_count(state, c) for c in open_clusters]
    power, count = max(stats, default=(0, 0))
    threshold = _town_threshold(fs)
    return {
        "towns": len(founded),
        "best_power": power,
        "best_count": count,
        "threshold": threshold,
        "n_clusters": len(open_clusters),
        "near": count >= NEAR_MIN_STRUCTURES and power >= threshold - NEAR_POWER_SLACK,
    }


def _one(item: tuple[str, str]) -> tuple[str, dict[str, list[dict]], str | None]:
    """Replay stored game ``item[0]`` under id ``item[1]``: a bot game runs
    under its setup's id so ``load_setup`` finds the configuration. Done per
    game -- two bot games can share a setup."""
    from bgai.engine.tm.replay import replay_game

    stored, game_id = item
    relabel = pl.lit(game_id).alias("game_id")
    moves = _W["moves"].filter(pl.col("game_id") == stored).with_columns(relabel)  # type: ignore[union-attr]
    deltas = _W["deltas"].filter(pl.col("game_id") == stored).with_columns(relabel)  # type: ignore[union-attr]

    samples: dict[str, list[dict]] = defaultdict(list)

    def hook(state, faction, cmd) -> None:
        if getattr(cmd, "verb", None) == "pass":
            samples[faction].append(_sample(state, faction))

    result = replay_game(game_id, moves, deltas, on_decision=hook)
    return stored, dict(samples), result.error


def _init_play(checkpoint: str, sims: int, c_puct: float) -> None:
    import torch

    from bgai.agents.mcts import MCTSAgent

    torch.set_num_threads(1)
    _W["agent"] = MCTSAgent(Path(checkpoint), simulations=sims, top_k=8, max_depth=48,
                            leaf_batch=16, c_puct=c_puct)


def _play_one(seed: int) -> tuple[str, dict[str, list[dict]], str | None]:
    """One self-play game (every seat the same agent), sampled at passes."""
    import random

    from bgai.arena.driver import advance, decision, new_game
    from bgai.arena.setups import sample_setup
    from bgai.engine.tm.apply import EngineError

    rng = random.Random(seed)
    setup = sample_setup(rng)
    agent = _W["agent"]
    agent.reset_tree()  # type: ignore[union-attr]
    sim = new_game(setup)
    samples: dict[str, list[dict]] = defaultdict(list)
    try:
        while (pending := decision(sim)) is not None:
            faction, offer = pending
            choice = offer[0] if len(offer) == 1 else agent.choose_sim(sim, faction, offer, rng)  # type: ignore[union-attr]
            if choice.verb == "pass":
                samples[faction].append(_sample(sim.game, faction))
            sim = advance(sim, choice)
    except EngineError as exc:
        return setup.game_id, dict(samples), str(exc)
    return setup.game_id, dict(samples), None


def play_all(checkpoint: Path, games: int, workers: int, sims: int, c_puct: float,
             seed: int) -> dict[str, dict[str, list[dict]]]:
    out: dict[str, dict[str, list[dict]]] = {}
    errors = 0
    seeds = [seed * 1_000_000 + i for i in range(games)]
    with get_context("spawn").Pool(workers, _init_play, (str(checkpoint), sims, c_puct)) as pool:
        for i, (setup_id, samples, err) in enumerate(pool.imap_unordered(_play_one, seeds)):
            if err:
                errors += 1
            else:
                out[f"{setup_id}#{i}"] = samples
            if (i + 1) % 25 == 0:
                print(f"  played {i + 1}/{games} ({errors} errors)", flush=True)
    print(f"  played {len(out)}/{games} games ({errors} errors)", flush=True)
    return out


def replay_all(moves: Path, deltas: Path, ids: list[str], rename: dict[str, str],
               workers: int) -> dict[str, dict[str, list[dict]]]:
    items = [(g, rename.get(g, g)) for g in ids]
    out: dict[str, dict[str, list[dict]]] = {}
    errors = 0
    with get_context("spawn").Pool(workers, _init, (str(moves), str(deltas), ids)) as pool:
        for gid, samples, err in pool.imap_unordered(_one, items, chunksize=4):
            if err:
                errors += 1
            else:
                out[gid] = samples
    print(f"  replayed {len(out)}/{len(ids)} games ({errors} errors)", flush=True)
    return out


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def summarize(games: dict[str, dict[str, list[dict]]]) -> dict[str, dict]:
    """Per faction: towns by round end, near-town rate, conversion, best cluster."""
    acc: dict[str, dict] = defaultdict(lambda: defaultdict(list))
    for samples in games.values():
        for fac, seq in samples.items():
            rounds = seq[1:7]  # seq[0] is the setup pass (round 0)
            for r, s in enumerate(rounds, start=1):
                acc[fac][f"towns_r{r}"].append(s["towns"])
                acc[fac][f"near_r{r}"].append(s["near"])
                acc[fac][f"power_r{r}"].append(s["best_power"])
                acc[fac][f"clusters_r{r}"].append(s["n_clusters"])
            for a, b in zip(rounds, rounds[1:], strict=False):
                if a["near"]:
                    acc[fac]["converted"].append(b["towns"] > a["towns"])
    return {fac: {k: _mean([float(x) for x in v]) for k, v in d.items()} | {"n": len(d["towns_r1"])}
            for fac, d in acc.items()}


def report(bot: dict, hum: dict) -> None:
    keys = [("towns by end R3/R6", "towns_r3", "towns_r6", "{:.2f}"),
            ("near-town at end R3/R5", "near_r3", "near_r5", "{:.0%}"),
            ("best open cluster power R3/R5", "power_r3", "power_r5", "{:.1f}"),
            ("open clusters R3/R6", "clusters_r3", "clusters_r6", "{:.1f}")]
    for fac in sorted(bot):
        b, h = bot[fac], hum.get(fac)
        if not h:
            continue
        print(f"\n## {fac} (bot seats {b['n']}, human {h['n']})")
        for label, k1, k2, fmt in keys:
            bp = f"{fmt.format(b[k1])} / {fmt.format(b[k2])}"
            hp = f"{fmt.format(h[k1])} / {fmt.format(h[k2])}"
            print(f"  {label:32s} bot {bp:14s} hum {hp}")
        print(f"  {'near-town -> town next round':32s} bot {b.get('converted', float('nan')):.0%}"
              f"{'':10s} hum {h.get('converted', float('nan')):.0%}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--games", type=int, default=300)
    ap.add_argument("--sims", type=int, default=512)
    ap.add_argument("--c-puct", type=float, default=2.5)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--human-dir", type=Path, default=Path("data/datasets"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    print("bot games (in-engine):", flush=True)
    bot = play_all(args.checkpoint, args.games, args.workers, args.sims, args.c_puct, args.seed)
    setups = sorted({key.split("#")[0] for key in bot})
    print("human games on the same setups:", flush=True)
    hum = replay_all(args.human_dir / "moves.parquet", args.human_dir / "deltas.parquet",
                     setups, {}, args.workers)
    bs, hs = summarize(bot), summarize(hum)
    report(bs, hs)
    if args.json:
        args.json.write_text(json.dumps({"bot": bs, "human": hs}, indent=1))


if __name__ == "__main__":
    main()
