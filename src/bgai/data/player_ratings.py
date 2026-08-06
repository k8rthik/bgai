"""Self-computed player strength ratings over the whole snellman population.

snellman publishes no player rating at any crawler-permitted endpoint, but
``population.parquet`` (built by ``bgai.data.population`` from the permitted
monthly events feeds) carries every game's seats, scores, and finishing order.
That is all TrueSkill needs, so strength can be estimated for all ~8k players
without fetching a single ledger.

Why bother: C5 established that Phase 6's bottleneck is the value head's
*coverage* of the state space, not its sample count -- and coverage means games
by players outside the Div 1-3 tournament field. Ratings let us sample that
population deliberately (a spread of strengths) instead of crawling everything,
which is both a better dataset and a far smaller request to make of someone
else's free service.

Games are rated in chronological order so a rating reflects only what came
before it -- the same no-lookahead discipline the imitation split uses.

Run: ``python -m bgai.data.player_ratings [population.parquet] [out.parquet]``
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl
import trueskill

DEFAULT_POPULATION = Path("data/catalogue/population.parquet")
DEFAULT_OUT = Path("data/catalogue/player_ratings.parquet")

BASE_MAP = "126fe960806d587c78546b30f1a90853b1ada468"


def playable(population: pl.DataFrame, player_count: int = 4) -> pl.DataFrame:
    """Seats of games this engine can actually replay: base map, given seat
    count, no Fire & Ice (the engine is base-game, 14 factions)."""
    scope = population.filter(
        (pl.col("base_map") == BASE_MAP) & (pl.col("player_count") == player_count)
    )
    # join-then-match rather than list.eval: an options-less game types as
    # List(Null), which .str.contains rejects outright.
    flagged = (
        scope.group_by("game_id")
        .agg(pl.col("options").first().alias("o"))
        .with_columns(
            pl.col("o")
            .cast(pl.List(pl.String))
            .list.join(",")
            .fill_null("")
            .str.contains("fire-and-ice")
            .alias("fire_and_ice")
        )
    )
    keep = flagged.filter(~pl.col("fire_and_ice")).select("game_id")
    return scope.join(keep, on="game_id", how="inner")


def rate(seats: pl.DataFrame) -> pl.DataFrame:
    """TrueSkill over games in chronological order. One row per player."""
    env = trueskill.TrueSkill(draw_probability=0.05)
    ratings: dict[str, trueskill.Rating] = {}
    games_played: dict[str, int] = {}

    ordered = seats.sort(["last_update", "game_id"])
    for _game_id, group in ordered.group_by(["game_id"], maintain_order=True):
        players = group["player"].to_list()
        places = group["place"].to_list()
        if len(players) < 2:
            continue
        teams = [[ratings.setdefault(p, env.create_rating())] for p in players]
        updated = env.rate(teams, ranks=places)
        for player, team in zip(players, updated, strict=True):
            ratings[player] = team[0]
            games_played[player] = games_played.get(player, 0) + 1

    return pl.DataFrame(
        {
            "player": list(ratings),
            "mu": [ratings[p].mu for p in ratings],
            "sigma": [ratings[p].sigma for p in ratings],
            # conservative estimate: the same mu-3*sigma the arena reports
            "conservative": [ratings[p].mu - 3 * ratings[p].sigma for p in ratings],
            "games": [games_played.get(p, 0) for p in ratings],
        }
    ).sort("conservative", descending=True)


def main(argv: list[str] | None = None) -> None:
    args = argv if argv is not None else sys.argv[1:]
    pop_path = Path(args[0]) if args else DEFAULT_POPULATION
    out_path = Path(args[1]) if len(args) > 1 else DEFAULT_OUT

    population = pl.read_parquet(pop_path)
    seats = playable(population)
    print(
        f"rating {seats['game_id'].n_unique():,} playable games, "
        f"{seats['player'].n_unique():,} players",
        flush=True,
    )
    table = rate(seats)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.write_parquet(out_path)

    print(f"wrote {out_path}: {table.height:,} players")
    experienced = table.filter(pl.col("games") >= 20)
    print(f"\ntop 10 (>= 20 games):\n{experienced.head(10)}")
    print(f"\nconservative-rating quantiles (>= 20 games, n={experienced.height:,}):")
    for q in (0.1, 0.25, 0.5, 0.75, 0.9, 0.99):
        print(f"  p{int(q * 100):<3} {experienced['conservative'].quantile(q):.2f}")


if __name__ == "__main__":
    main()
