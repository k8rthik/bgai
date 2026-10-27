"""Do bots see the town-finishing move? Search coverage at near-town states.

    uv run python tools/town_finish_diag.py \\
        --checkpoint data/checkpoints/champion_ftcos_it14/current.pt --games 300 --workers 18

At every decision where the mover has a legal build or upgrade that would
found a town (tested by ``apply()`` on the pre-state: ``founded_towns``
grows), records for the bot -- playing in-engine at the champion eval config
-- the best finishing move's prior rank, whether it survives the ``top_k``
cut, its visit share, its search value against the chosen move's, and
whether a finishing move was played. Human games on the same setups are
replayed and the same availability test is run on every decision.

Per faction-round: was a finisher available at some point, and was a town
founded in that round? The bot's finisher statistics then say whether a
missed town was never searched (outside top_k / low prior) or searched and
rejected (search valued something else more).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import polars as pl

_W: dict[str, object] = {}


def _finishers(state, faction: str) -> list:
    """Legal builds/upgrades for ``faction`` that would found a town now."""
    from bgai.engine.tm.apply import EngineError, apply
    from bgai.engine.tm.legal_actions import build_moves, upgrade_moves

    fs = state.factions[faction]
    before = len(state.founded_towns.get(faction, ()))
    out = []
    for cmd in build_moves(state, faction, fs) + upgrade_moves(state, faction, fs):
        try:
            after = apply(state, faction, cmd)
        except (EngineError, ValueError, KeyError):
            continue
        if len(after.founded_towns.get(faction, ())) > before:
            out.append(cmd)
    return out


def _same(a, b) -> bool:
    return (a.verb, a.loc, a.building) == (b.verb, b.loc, b.building)


# ---------------------------------------------------------------- bot side


def _init_play(checkpoint: str, sims: int, c_puct: float, top_k: int) -> None:
    import torch

    from bgai.agents.mcts import MCTSAgent

    torch.set_num_threads(1)
    _W["agent"] = MCTSAgent(Path(checkpoint), simulations=sims, top_k=top_k, max_depth=48,
                            leaf_batch=16, c_puct=c_puct)
    _W["top_k"] = top_k


def _decision_record(root, offer, finish_idx: list[int], chosen: int) -> dict:
    priors = np.asarray(root.priors, dtype=np.float64)
    order = np.argsort(-priors)
    rank = {int(i): r for r, i in enumerate(order)}
    best = min(finish_idx, key=lambda i: rank[i])
    visits = root.visits.astype(np.float64)
    total = max(1.0, visits.sum())
    seat = root.sim.game.setup.factions.index(root.faction)

    def q(i: int) -> float | None:
        return float(root.action_value[i, seat] / visits[i]) if visits[i] > 0 else None

    return {
        "n_offer": len(offer),
        "finish_rank": rank[best] + 1,
        "finish_prior": float(priors[best]),
        "in_top_k": rank[best] < int(_W["top_k"]) or len(offer) <= int(_W["top_k"]),
        "finish_visit_share": float(sum(visits[i] for i in finish_idx) / total),
        "chosen_finisher": chosen in finish_idx,
        "q_finish": q(best),
        "q_chosen": q(chosen),
    }


def _play_one(seed: int) -> dict:
    import random

    from bgai.arena.driver import advance, decision, new_game
    from bgai.arena.setups import sample_setup
    from bgai.engine.tm.apply import EngineError

    rng = random.Random(seed)
    setup = sample_setup(rng)
    agent = _W["agent"]
    agent.reset_tree()  # type: ignore[union-attr]
    sim = new_game(setup)
    rounds: dict[str, dict[int, dict]] = defaultdict(dict)
    decisions: list[dict] = []
    passes: dict[str, int] = defaultdict(int)
    try:
        while (pending := decision(sim)) is not None:
            faction, offer = pending
            rnd = passes[faction]
            towns_before = len(sim.game.founded_towns.get(faction, ()))
            fins = _finishers(sim.game, faction) if len(offer) > 1 else []
            finish_idx = [i for i, c in enumerate(offer) if any(_same(c, f) for f in fins)]
            if len(offer) == 1:
                choice = offer[0]
            else:
                root = agent.search(sim)  # type: ignore[union-attr]
                choice = agent.choose_sim(sim, faction, offer, rng)  # type: ignore[union-attr]
                if finish_idx and root is not None:
                    chosen = next(i for i, c in enumerate(offer) if c is choice)
                    decisions.append({"faction": faction, "round": rnd,
                                      **_decision_record(root, offer, finish_idx, chosen)})
            slot = rounds[faction].setdefault(rnd, {"available": False, "founded": False})
            slot["available"] |= bool(finish_idx)
            sim = advance(sim, choice)
            slot["founded"] |= len(sim.game.founded_towns.get(faction, ())) > towns_before
            if choice.verb == "pass":
                passes[faction] += 1
    except EngineError as exc:
        return {"setup": setup.game_id, "error": str(exc)}
    return {"setup": setup.game_id, "rounds": _plain(rounds), "decisions": decisions}


def _plain(rounds: dict) -> dict:
    return {f: {str(r): v for r, v in rs.items()} for f, rs in rounds.items()}


# -------------------------------------------------------------- human side


def _init_replay(moves_path: str, deltas_path: str, ids: list[str]) -> None:
    _W["moves"] = pl.scan_parquet(moves_path).filter(pl.col("game_id").is_in(ids)).collect()
    _W["deltas"] = pl.scan_parquet(deltas_path).filter(pl.col("game_id").is_in(ids)).collect()


def _replay_one(game_id: str) -> dict:
    from bgai.engine.tm.apply import EngineError, apply
    from bgai.engine.tm.replay import replay_game

    rounds: dict[str, dict[int, dict]] = defaultdict(dict)
    passes: dict[str, int] = defaultdict(int)

    def hook(state, faction, cmd) -> None:
        rnd = passes[faction]
        slot = rounds[faction].setdefault(rnd, {"available": False, "founded": False})
        if cmd.verb in ("build", "upgrade", "pass") or not slot["available"]:
            slot["available"] |= bool(_finishers(state, faction))
        if cmd.verb in ("build", "upgrade"):
            try:
                after = apply(state, faction, cmd)
            except (EngineError, ValueError, KeyError):
                after = state
            before = len(state.founded_towns.get(faction, ()))
            slot["founded"] |= len(after.founded_towns.get(faction, ())) > before
        if cmd.verb == "pass":
            passes[faction] += 1

    res = replay_game(game_id, _W["moves"], _W["deltas"], on_decision=hook)  # type: ignore[arg-type]
    if res.error:
        return {"setup": game_id, "error": res.error}
    return {"setup": game_id, "rounds": _plain(rounds)}


# ----------------------------------------------------------------- summary


def summarize(games: list[dict]) -> dict[str, dict]:
    acc: dict[str, dict] = defaultdict(lambda: {"fr": 0, "avail": 0, "founded_given_avail": 0,
                                                "founded_total": 0})
    for g in games:
        for fac, rs in g.get("rounds", {}).items():
            for r, slot in rs.items():
                if not 1 <= int(r) <= 6:
                    continue
                a = acc[fac]
                a["fr"] += 1
                a["founded_total"] += slot["founded"]
                if slot["available"]:
                    a["avail"] += 1
                    a["founded_given_avail"] += slot["founded"]
    return dict(acc)


def summarize_decisions(games: list[dict]) -> dict[str, dict]:
    by: dict[str, list[dict]] = defaultdict(list)
    for g in games:
        for d in g.get("decisions", []):
            by[d["faction"]].append(d)
    out = {}
    for fac, ds in by.items():
        not_chosen = [d for d in ds if not d["chosen_finisher"]]
        q_gap = [d["q_chosen"] - d["q_finish"] for d in not_chosen
                 if d["in_top_k"] and d["q_finish"] is not None and d["q_chosen"] is not None]
        out[fac] = {
            "n": len(ds),
            "chosen": sum(d["chosen_finisher"] for d in ds) / len(ds),
            "in_top_k": sum(d["in_top_k"] for d in ds) / len(ds),
            "missed_outside_top_k": sum(not d["in_top_k"] for d in not_chosen) / max(1, len(ds)),
            "missed_searched": sum(d["in_top_k"] for d in not_chosen) / max(1, len(ds)),
            "median_rank": float(np.median([d["finish_rank"] for d in ds])),
            "mean_visit_share": float(np.mean([d["finish_visit_share"] for d in ds])),
            "mean_q_gap_when_rejected": float(np.mean(q_gap)) if q_gap else float("nan"),
        }
    return out


def report(bot_rounds: dict, hum_rounds: dict, bot_dec: dict) -> None:
    print(f"\n{'faction':15s} | {'finisher avail (rounds)':>23s} | {'town | avail':>13s} | "
          f"{'bot decisions w/ finisher':>25s}")
    print(f"{'':15s} | {'bot':>11s} {'human':>11s} | {'bot':>6s} {'human':>6s} | "
          f"{'n':>5s} {'played':>6s} {'top8':>5s} {'rank':>5s} {'cut':>5s} "
          f"{'rejected':>8s} {'dQ':>6s}")
    for fac in sorted(bot_rounds):
        b, h, d = bot_rounds[fac], hum_rounds.get(fac), bot_dec.get(fac)
        if not h:
            continue
        rate = lambda a: a["avail"] / a["fr"] if a["fr"] else float("nan")  # noqa: E731
        conv = lambda a: a["founded_given_avail"] / a["avail"] if a["avail"] else float("nan")  # noqa: E731
        dpart = (f"{d['n']:5d} {d['chosen']:6.0%} {d['in_top_k']:5.0%} {d['median_rank']:5.1f} "
                 f"{d['missed_outside_top_k']:5.0%} {d['missed_searched']:8.0%} "
                 f"{d['mean_q_gap_when_rejected']:+6.3f}") if d else "-"
        print(f"{fac:15s} | {rate(b):11.0%} {rate(h):11.0%} | "
              f"{conv(b):6.0%} {conv(h):6.0%} | {dpart}")
    print("\ncolumns: finisher avail = share of faction-rounds with a town-founding build/upgrade "
          "legal at some point; town|avail = share of those rounds in which a town was founded; "
          "played = decisions where a finisher was chosen; top8 = best finisher inside the top_k "
          "prior cut; cut / rejected = decisions where no finisher was played and it was outside "
          "top_k / inside top_k; dQ = chosen minus finisher search value when rejected.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--games", type=int, default=300)
    ap.add_argument("--sims", type=int, default=512)
    ap.add_argument("--c-puct", type=float, default=2.5)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--human-dir", type=Path, default=Path("data/datasets"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    ctx = get_context("spawn")
    seeds = [args.seed * 1_000_000 + i for i in range(args.games)]
    bot: list[dict] = []
    with ctx.Pool(args.workers, _init_play,
                  (str(args.checkpoint), args.sims, args.c_puct, args.top_k)) as pool:
        for i, g in enumerate(pool.imap_unordered(_play_one, seeds)):
            bot.append(g)
            if (i + 1) % 25 == 0:
                print(f"played {i + 1}/{args.games}", flush=True)
    ok = [g for g in bot if "error" not in g]
    setups = sorted({g["setup"] for g in ok})
    print(f"bot: {len(ok)}/{len(bot)} games ok; replaying {len(setups)} human games", flush=True)
    with ctx.Pool(args.workers, _init_replay,
                  (str(args.human_dir / "moves.parquet"), str(args.human_dir / "deltas.parquet"),
                   setups)) as pool:
        hum = [g for g in pool.imap_unordered(_replay_one, setups) if "error" not in g]
    print(f"human: {len(hum)}/{len(setups)} games ok", flush=True)
    br, hr, bd = summarize(ok), summarize(hum), summarize_decisions(ok)
    report(br, hr, bd)
    if args.json:
        args.json.write_text(json.dumps({"bot_rounds": br, "human_rounds": hr, "bot_decisions": bd},
                                        indent=1))


if __name__ == "__main__":
    main()
