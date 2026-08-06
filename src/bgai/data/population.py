"""Whole-population index built from the crawler-permitted events feeds.

``snellman_feeds`` caches ``/data/events/<YYYY-MM>.json``, which lists every
game finished that month. ``build_events_index`` there answers "which of *our
catalogued* games qualify"; this module answers the wider question the value
head needs: **who played every game on the site, and what did they score.**

Each monthly feed carries, per game, the player/faction pairing and each
faction's cumulative VP (``events.faction.<faction>.vp.round.all``, which sums
to ``events.faction.all.vp.round.all`` -- verified on 2024-06). That is enough
to rank every game without fetching a single ledger, so player strength can be
estimated from permitted data alone.

Why this exists: snellman publishes no player rating anywhere crawler-permitted,
and Phase-6's C5 finding says the value head's deficit is *state-space coverage*,
not sample count. Coverage means games by players weaker than the Div 1-3
tournament field -- so we need a strength estimate over the whole population to
sample against, and we need it before deciding which ledgers are worth
requesting.

Run: ``python -m bgai.data.population [events_dir] [out_dir]``
"""

from __future__ import annotations

import sys
from pathlib import Path

import orjson
import polars as pl

DEFAULT_EVENTS_DIR = Path("data/raw/events")
DEFAULT_OUT_DIR = Path("data/catalogue")


def _game_rows(game: dict, month: str) -> list[dict]:
    """One row per seat. Empty if the feed entry lacks scores or pairings."""
    game_id = game.get("game")
    factions = game.get("factions") or []
    per_faction = (game.get("events") or {}).get("faction") or {}
    if not game_id or not factions:
        return []

    options = sorted(
        key.removeprefix("option-")
        for key in ((game.get("events") or {}).get("global") or {})
        if key.startswith("option-")
    )

    rows: list[dict] = []
    for seat in factions:
        player, faction = seat.get("player"), seat.get("faction")
        if not player or not faction:
            continue
        vp = ((per_faction.get(faction) or {}).get("vp") or {}).get("round", {}).get("all")
        if vp is None:  # abandoned/incomplete game: no score recorded
            continue
        rows.append(
            {
                "game_id": game_id,
                "month": month,
                "base_map": game.get("base_map"),
                "player_count": game.get("player_count"),
                "last_update": game.get("last_update"),
                "options": options,
                "player": player,
                "faction": faction,
                "vp": int(vp),
            }
        )
    # A game is only usable if every seat scored; partial rows would bias ranks.
    return rows if len(rows) == len(factions) else []


def build_population(events_dir: Path) -> pl.DataFrame:
    """One row per (game, seat) across every cached month, with placement."""
    rows: list[dict] = []
    months = sorted(events_dir.glob("*.json"))
    for index, path in enumerate(months, start=1):
        for game in orjson.loads(path.read_bytes()):
            rows.extend(_game_rows(game, path.stem))
        if index % 20 == 0 or index == len(months):
            print(f"  {index}/{len(months)} months, {len(rows)} seats", flush=True)

    frame = pl.DataFrame(rows)
    # 0-based competition rank within each game; ties share the better rank.
    return frame.with_columns(
        (
            pl.col("vp").rank(method="min", descending=True).over("game_id").cast(pl.Int32) - 1
        ).alias("place")
    )


def main(argv: list[str] | None = None) -> None:
    args = argv if argv is not None else sys.argv[1:]
    events_dir = Path(args[0]) if args else DEFAULT_EVENTS_DIR
    out_dir = Path(args[1]) if len(args) > 1 else DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"reading {events_dir}", flush=True)
    population = build_population(events_dir)
    out = out_dir / "population.parquet"
    population.write_parquet(out)

    games = population["game_id"].n_unique()
    players = population["player"].n_unique()
    print(f"\nwrote {out}: {population.height} seats, {games} games, {players} players")
    by_count = population.group_by("player_count").len().sort("player_count")
    by_map = population.group_by("base_map").len().sort("len", descending=True).head(3)
    print("\nplayer_count:", by_count.to_dicts())
    print("\ntop base_maps:", by_map.to_dicts())


if __name__ == "__main__":
    main()
