"""Where does an agent stand among human players? The frozen yardstick for
the top-decile goal.

Plays ONE game per corpus setup with all four seats held by the same agent
(a mirrored arena series would replay that deterministic game four times),
then reports:

  * bot mean VP per seat, with its standard error over setups;
  * the humans who played those same setups, and the paired difference;
  * where the bot's mean falls among human PLAYER AVERAGES (4-player base
    map, base-game factions only, players with >= MIN_GAMES such games):
    an average is compared with averages, not with single games.

Target: the 90th percentile of those player averages (top decile).

    uv run python tools/human_standing.py run --agent 'mcts:ckpt=...,sims=512,...' \\
        --setups 200 --seed 1 --workers 18 --out data/standing/champ.json
    uv run python tools/human_standing.py report data/standing/*.json
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import random
import sys
import time
from functools import lru_cache
from pathlib import Path

import polars as pl

from bgai.arena.series import table_setup_rng
from bgai.arena.setups import sample_setup
from bgai.arena.sim import run_game
from bgai.training.vocab import FACTION_NAMES

POPULATION = Path("data/catalogue/population.parquet")
GAMES_META = Path("data/datasets/games_meta.parquet")
MIN_GAMES = 20
TARGET_PCT = 0.90

_WORKER: dict[str, object] = {}


def _init(spec: str) -> None:
    from bgai.arena.run import _build_agents

    agents = _build_agents(spec)
    if len(agents) != 1:
        raise ValueError(f"need exactly one agent, got {list(agents)}")
    _WORKER["agent"] = next(iter(agents.values()))


def _play(args: tuple[int, int]) -> dict:
    seed, table = args
    setup = sample_setup(table_setup_rng(seed, table))
    agent = _WORKER["agent"]
    started = time.time()
    result = run_game(setup, {f: agent for f in setup.factions}, random.Random(f"{seed}:{table}"))
    return {
        "table": table,
        "game_id": result.setup_game_id,
        "vps": dict(result.vps),
        "error": result.error,
        "decisions": result.decisions,
        "seconds": round(time.time() - started, 1),
    }


def run(args: argparse.Namespace) -> None:
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done: dict[int, dict] = {}
    if args.out.exists():
        done = {g["table"]: g for g in json.loads(args.out.read_text())["games"]}
    todo = [(args.seed, t) for t in range(args.setups) if t not in done]
    print(f"{len(done)} setups done, {len(todo)} to play", flush=True)

    def save() -> None:
        payload = {
            "agent": args.agent, "seed": args.seed, "label": args.label,
            "games": [done[t] for t in sorted(done)],
        }
        tmp = args.out.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload))
        tmp.replace(args.out)

    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers, initializer=_init, initargs=(args.agent,)) as pool:
        for game in pool.imap_unordered(_play, todo):
            done[game["table"]] = game
            save()
            vps = list(game["vps"].values())
            print(
                f"table {game['table']:>4} {game['game_id']:<24} mean {sum(vps) / 4:6.1f} "
                f"{game['seconds']:>7.0f}s err={game['error']}",
                flush=True,
            )
    print(json.dumps(summarize(json.loads(args.out.read_text())), indent=2))


@lru_cache(maxsize=1)
def human_player_averages() -> pl.Series:
    """Average VP per human player over 4p base-map games whose four
    factions are all base-game factions (the engine's scope)."""
    pop = pl.read_parquet(
        POPULATION, columns=["game_id", "base_map", "player_count", "player", "faction", "vp"]
    )
    base_map = pop.group_by("base_map").len().sort("len", descending=True)["base_map"][0]
    four = pop.filter((pl.col("player_count") == 4) & (pl.col("base_map") == base_map))
    base14 = set(FACTION_NAMES)
    ok_games = (
        four.group_by("game_id")
        .agg(pl.col("faction").is_in(base14).all().alias("ok"), pl.len().alias("n"))
        .filter(pl.col("ok") & (pl.col("n") == 4))["game_id"]
    )
    per_player = (
        four.filter(pl.col("game_id").is_in(ok_games.to_list()))
        .group_by("player")
        .agg(pl.col("vp").mean().alias("avg"), pl.len().alias("n"))
        .filter(pl.col("n") >= MIN_GAMES)
    )
    return per_player["avg"].sort()


@lru_cache(maxsize=1)
def human_vps_by_game() -> dict[str, dict[str, int]]:
    meta = pl.read_parquet(GAMES_META, columns=["game_id", "final_vp"])
    return {g: json.loads(v) for g, v in zip(meta["game_id"], meta["final_vp"], strict=True)}


def percentile_of(value: float, sorted_avgs: pl.Series) -> float:
    return float((sorted_avgs <= value).sum()) / sorted_avgs.len()


def summarize(run_payload: dict) -> dict:
    games = [g for g in run_payload["games"] if g["error"] is None]
    errors = len(run_payload["games"]) - len(games)
    humans = human_vps_by_game()
    table_means = [sum(g["vps"].values()) / 4 for g in games]
    paired = []
    for g, bot_mean in zip(games, table_means, strict=True):
        human = humans.get(g["game_id"])
        if human and set(human) == set(g["vps"]):
            paired.append(bot_mean - sum(human.values()) / 4)
    avgs = human_player_averages()
    n = len(table_means)
    mean = sum(table_means) / n if n else math.nan
    se = (
        math.sqrt(sum((m - mean) ** 2 for m in table_means) / (n - 1) / n) if n > 1 else math.nan
    )
    d_mean = sum(paired) / len(paired) if paired else math.nan
    d_se = (
        math.sqrt(sum((d - d_mean) ** 2 for d in paired) / (len(paired) - 1) / len(paired))
        if len(paired) > 1 else math.nan
    )
    target = float(avgs.quantile(TARGET_PCT))
    return {
        "label": run_payload.get("label"),
        "agent": run_payload["agent"],
        "setups": n,
        "errors": errors,
        "bot_vp_mean": round(mean, 2),
        "bot_vp_se": round(se, 2),
        "human_pct_of_mean": round(percentile_of(mean, avgs), 3),
        "target_vp_p90": round(target, 2),
        "gap_to_target": round(target - mean, 2),
        "paired_setups": len(paired),
        "bot_minus_same_setup_humans": round(d_mean, 2),
        "paired_se": round(d_se, 2),
        "human_players": avgs.len(),
        "human_p50": round(float(avgs.quantile(0.5)), 2),
    }


def report(args: argparse.Namespace) -> None:
    rows = [summarize(json.loads(p.read_text())) for p in args.files]
    cols = ["label", "setups", "errors", "bot_vp_mean", "bot_vp_se", "human_pct_of_mean",
            "bot_minus_same_setup_humans", "paired_se", "gap_to_target"]
    print("\t".join(cols))
    for r in rows:
        print("\t".join(str(r[c]) for c in cols))
    if rows:
        r = rows[0]
        print(f"target (p{int(TARGET_PCT * 100)} of {r['human_players']} human player averages): "
              f"{r['target_vp_p90']} VP; human median {r['human_p50']}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--agent", required=True, help="one arena agent spec")
    r.add_argument("--label", default=None)
    r.add_argument("--setups", type=int, default=200)
    r.add_argument("--seed", type=int, default=1)
    r.add_argument("--workers", type=int, default=8)
    r.add_argument("--out", type=Path, required=True)
    rep = sub.add_parser("report")
    rep.add_argument("files", type=Path, nargs="+")
    args = parser.parse_args(argv)
    if args.cmd == "run":
        run(args)
    else:
        report(args)


if __name__ == "__main__":
    sys.exit(main())
