"""Per-faction play profile from a corpus-schema game store, bots vs humans.

    uv run python tools/faction_profile.py --bot-batch champ_it14_512

Profiles the recorded bot games (data/agent_games, one batch) and the human
games played on the SAME setups (data/datasets, matched by setup_game_id),
through identical code, so every difference is a behaviour difference:
starting hexes, round-1 temple / stronghold / sanctuary, stronghold timing,
round-1 favor tiles, towns, dig / shipping advances and final VP.

Rounds come from each faction's passes (see ``with_rounds``): round 0 is
setup (initial dwellings).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import polars as pl

COLS = ["game_id", "row", "seq", "faction", "kind", "verb", "loc", "building", "tile", "reason"]


def with_rounds(moves: pl.DataFrame) -> pl.DataFrame:
    """A move's round = how many times its faction has passed before it.

    Every faction passes once per round, and the setup bonus-tile pick is a
    pass too, so round 0 is setup (initial dwellings) for every faction. The
    human and bot logs mark income differently (two income blocks per round
    vs none at all); passes are the one marker both share.
    """
    moves = moves.sort(["game_id", "row", "seq"])
    passes = (pl.col("verb") == "pass").cast(pl.Int32)
    before = passes.cum_sum().over(["game_id", "faction"]) - passes
    return moves.with_columns(before.alias("round"))


def load(moves_path: Path | list[Path], game_ids: set[str] | None) -> pl.DataFrame:
    paths = moves_path if isinstance(moves_path, list) else [moves_path]
    lf = pl.concat([pl.scan_parquet(p).select(COLS) for p in paths])
    if game_ids is not None:
        lf = lf.filter(pl.col("game_id").is_in(list(game_ids)))
    return with_rounds(lf.collect())


def share_in_round(upgrades: pl.DataFrame, building: str, rnd: int, n: int) -> float:
    hit = upgrades.filter((pl.col("building") == building) & (pl.col("round") == rnd))
    return hit["game_id"].n_unique() / n


def profile(moves: pl.DataFrame, final_vp: dict[str, dict[str, int]]) -> dict[str, dict]:
    dec = moves.filter(pl.col("kind") == "decision")
    out: dict[str, dict] = {}
    for (fac,), g in dec.group_by(["faction"]):
        games = g["game_id"].unique().to_list()
        n = len(games)
        starts = (
            g.filter((pl.col("round") == 0) & (pl.col("verb") == "build"))
            .group_by("game_id")
            .agg(pl.col("loc").sort().str.join("+").alias("hexes"))
        )
        up = g.filter(pl.col("verb") == "upgrade")
        first_sh = (
            up.filter(pl.col("building") == "SH").group_by("game_id").agg(pl.col("round").min())
        )
        favors_r1 = Counter(
            g.filter((pl.col("verb") == "gain_favor") & (pl.col("round") == 1))["tile"].to_list()
        )
        adv = g.filter(pl.col("verb") == "advance")
        vps = [final_vp[gid][fac] for gid in games if gid in final_vp and fac in final_vp[gid]]
        out[fac] = {
            "n": n,
            "starts": Counter(starts["hexes"].to_list()).most_common(3),
            "r1_temple": share_in_round(up, "TE", 1, n),
            "r1_sh": share_in_round(up, "SH", 1, n),
            "r1_sa": share_in_round(up, "SA", 1, n),
            "sh_never": 1 - first_sh.height / n,
            "sh_median_round": first_sh["round"].median() if first_sh.height else None,
            "r1_favors": favors_r1.most_common(3),
            "r1_favor_total": sum(favors_r1.values()),
            "towns": g.filter(pl.col("verb") == "gain_town").height / n,
            "dig_adv": adv.filter(pl.col("reason") == "dig").height / n,
            "ship_adv": adv.filter(pl.col("reason") == "ship").height / n,
            "ship_by_r2": (
                adv.filter((pl.col("reason") == "ship") & (pl.col("round") <= 2)).height / n
            ),
            "vp": sum(vps) / len(vps) if vps else float("nan"),
        }
    return out


def fmt_starts(starts: list[tuple[str, int]], n: int) -> str:
    return ", ".join(f"{h} {100 * c / n:.0f}%" for h, c in starts)


def fmt_fav(favs: list[tuple[str, int]], total: int) -> str:
    return ", ".join(f"{t} {100 * c / total:.0f}%" for t, c in favs) if total else "-"


def report(bot: dict, hum: dict) -> None:
    for fac in sorted(bot):
        b, h = bot[fac], hum.get(fac)
        if h is None:
            continue
        print(f"\n## {fac}  (bot seats {b['n']}, human seats {h['n']})")
        print(f"  start hexes  bot: {fmt_starts(b['starts'], b['n'])}")
        print(f"               hum: {fmt_starts(h['starts'], h['n'])}")
        print(f"  R1 temple / SH / SA   bot {b['r1_temple']:.0%} / {b['r1_sh']:.0%} / "
              f"{b['r1_sa']:.0%}   hum {h['r1_temple']:.0%} / {h['r1_sh']:.0%} / {h['r1_sa']:.0%}")
        print(f"  SH never / median rnd bot {b['sh_never']:.0%} / {b['sh_median_round']}"
              f"   hum {h['sh_never']:.0%} / {h['sh_median_round']}")
        print(f"  R1 favors    bot: {fmt_fav(b['r1_favors'], b['r1_favor_total'])}")
        print(f"               hum: {fmt_fav(h['r1_favors'], h['r1_favor_total'])}")
        print(f"  towns {b['towns']:.2f} vs {h['towns']:.2f} | dig adv {b['dig_adv']:.2f} vs "
              f"{h['dig_adv']:.2f} | ship adv {b['ship_adv']:.2f} vs {h['ship_adv']:.2f} "
              f"(by R2 {b['ship_by_r2']:.2f} vs {h['ship_by_r2']:.2f}) | "
              f"VP {b['vp']:.1f} vs {h['vp']:.1f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bot-batch", required=True)
    ap.add_argument("--bot-store", type=Path, default=Path("data/agent_games"))
    ap.add_argument("--human-moves", type=Path, default=Path("data/datasets/moves.parquet"))
    ap.add_argument("--human-meta", type=Path, default=Path("data/datasets/games_meta.parquet"))
    ap.add_argument("--json", type=Path, default=None, help="also write both profiles here")
    args = ap.parse_args()

    bmeta = pl.read_parquet(args.bot_store / "meta" / f"{args.bot_batch}.parquet")
    bot_vp = {r["game_id"]: json.loads(r["final_vp"]) for r in bmeta.iter_rows(named=True)}
    setups = set(bmeta["setup_game_id"].to_list())
    hmeta = pl.read_parquet(args.human_meta).filter(pl.col("game_id").is_in(list(setups)))
    hum_vp = {r["game_id"]: json.loads(r["final_vp"]) for r in hmeta.iter_rows(named=True)}

    bot = profile(load(args.bot_store / "moves" / f"{args.bot_batch}.parquet", None), bot_vp)
    hum = profile(load(args.human_moves, setups), hum_vp)
    print(f"bot: {bmeta.height} games ({args.bot_batch}); "
          f"human: {hmeta.height} games on the same setups")
    report(bot, hum)
    if args.json:
        args.json.write_text(json.dumps({"bot": bot, "human": hum}, indent=1, default=str))


if __name__ == "__main__":
    main()
