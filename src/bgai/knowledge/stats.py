"""Win-conditioned corpus statistics for the rung-4 compendium.

Pure polars functions over ``moves.parquet``/``deltas.parquet``. The
output is aggregate statistics only -- no positions, no retrieved moves
(spec: the compendium teaches principles with effect sizes, never plays).
Round attribution derives from each faction's k-th income row (round k
starts at the earliest k-th income row across factions; rows before the
first batch are setup, round 0).
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

_INCOME_VERBS = ("other_income_for_faction", "all_income_for_faction")
_TIMING_VERBS = ("build", "upgrade", "send", "action", "gain_town", "dig")


def final_standings(deltas: pl.DataFrame) -> pl.DataFrame:
    """Per (game_id, faction): final VP (last delta row), rank, won flag
    (ties share rank 1 -- snellman shares wins)."""
    last = (
        deltas.sort("row")
        .group_by(["game_id", "faction"], maintain_order=True)
        .agg(pl.col("vp_value").last().alias("final_vp"))
    )
    return last.with_columns(
        pl.col("final_vp").rank("min", descending=True).over("game_id").alias("rank")
    ).with_columns((pl.col("rank") == 1).alias("won"))


def _round_starts(moves: pl.DataFrame) -> pl.DataFrame:
    income = moves.filter(pl.col("verb").is_in(_INCOME_VERBS)).sort("row")
    per_faction = income.with_columns(
        pl.col("row").cum_count().over(["game_id", "faction"]).alias("round")
    )
    return (
        per_faction.group_by(["game_id", "round"])
        .agg(pl.col("row").min().alias("start_row"))
        .sort(["game_id", "start_row"])
        .with_columns(pl.col("round").cast(pl.Int32))
    )


def round_index(moves: pl.DataFrame) -> pl.DataFrame:
    """Per (game_id, row): the round it belongs to (0 = setup)."""
    starts = _round_starts(moves)
    rows = moves.select(["game_id", "row"]).unique().sort(["game_id", "row"])
    joined = rows.join_asof(
        starts,
        left_on="row",
        right_on="start_row",
        by="game_id",
        strategy="backward",
        check_sortedness=False,  # both frames are pre-sorted by (game_id, row) above
    )
    return joined.with_columns(pl.col("round").fill_null(0)).select(
        ["game_id", "row", "round"]
    )


def _decisions_with_context(moves: pl.DataFrame, deltas: pl.DataFrame) -> pl.DataFrame:
    ri = round_index(moves)
    st = final_standings(deltas).select(["game_id", "faction", "won"])
    return moves.join(ri, on=["game_id", "row"]).join(st, on=["game_id", "faction"])


def faction_win_rates(moves: pl.DataFrame, deltas: pl.DataFrame) -> pl.DataFrame:
    st = final_standings(deltas)
    factions = moves.select(["game_id", "faction"]).unique()
    joined = factions.join(st, on=["game_id", "faction"], how="inner")
    return (
        joined.group_by("faction")
        .agg(
            pl.len().alias("games"),
            pl.col("won").sum().alias("wins"),
            pl.col("won").mean().alias("win_rate"),
            pl.col("final_vp").mean().alias("mean_final_vp"),
        )
        .sort("win_rate", descending=True)
    )


def _games_per_group(df: pl.DataFrame) -> pl.DataFrame:
    return df.group_by(["faction", "won"]).agg(
        pl.col("game_id").n_unique().alias("games")
    )


def timing_curves(moves: pl.DataFrame, deltas: pl.DataFrame) -> pl.DataFrame:
    """Per faction x won x round x verb: mean occurrences per game."""
    df = _decisions_with_context(moves, deltas).filter(pl.col("verb").is_in(_TIMING_VERBS))
    games = _games_per_group(_decisions_with_context(moves, deltas))
    counts = df.group_by(["faction", "won", "round", "verb"]).agg(pl.len().alias("n"))
    return (
        counts.join(games, on=["faction", "won"])
        .with_columns((pl.col("n") / pl.col("games")).alias("mean_per_game"))
        .sort(["faction", "won", "round", "verb"])
    )


def opening_patterns(moves: pl.DataFrame, deltas: pl.DataFrame) -> pl.DataFrame:
    """Per faction x won: mean dwellings by end of round 1, and mean round
    of the first upgrade to each building type."""
    df = _decisions_with_context(moves, deltas)
    early_builds = (
        df.filter((pl.col("verb") == "build") & (pl.col("round") <= 1))
        .group_by(["game_id", "faction", "won"])
        .agg(pl.len().alias("early_dwellings"))
        .group_by(["faction", "won"])
        .agg(pl.col("early_dwellings").mean().alias("mean_dwellings_by_r1"))
    )
    first_upgrades = (
        df.filter(pl.col("verb") == "upgrade")
        .group_by(["game_id", "faction", "won", "building"])
        .agg(pl.col("round").min().alias("first_round"))
        .group_by(["faction", "won", "building"])
        .agg(pl.col("first_round").mean().alias("mean_first_upgrade_round"))
        .pivot(on="building", index=["faction", "won"], values="mean_first_upgrade_round")
    )
    return early_builds.join(first_upgrades, on=["faction", "won"], how="left").sort(
        ["faction", "won"]
    )


def tile_pick_rates(moves: pl.DataFrame, deltas: pl.DataFrame) -> pl.DataFrame:
    """Bonus-tile picks (pass rows with a tile) by faction x round x won,
    and favor-tile picks by faction x won -- as counts per game."""
    df = _decisions_with_context(moves, deltas)
    picks = df.filter(
        (pl.col("verb").is_in(("pass", "gain_favor"))) & pl.col("tile").is_not_null()
    )
    games = _games_per_group(df)
    return (
        picks.group_by(["faction", "won", "round", "verb", "tile"])
        .agg(pl.len().alias("n"))
        .join(games, on=["faction", "won"])
        .with_columns((pl.col("n") / pl.col("games")).alias("picks_per_game"))
        .sort(["faction", "won", "round", "tile"])
    )


_MAIN_VERBS = ("build", "upgrade", "action", "advance", "pass", "send", "dig", "transform",
               "bridge", "connect")


def leech_behavior(moves: pl.DataFrame, deltas: pl.DataFrame) -> pl.DataFrame:
    """Win-conditioned leech accept rate by offer amount: every `leech`
    (accept) or `decline` row carries the offered amount in n1."""
    df = _decisions_with_context(moves, deltas).filter(
        pl.col("verb").is_in(("leech", "decline")) & pl.col("n1").is_not_null()
    )
    return (
        df.group_by(["won", pl.col("n1").alias("amount")])
        .agg(pl.len().alias("offers"), (pl.col("verb") == "leech").mean().alias("accept_rate"))
        .sort(["won", "amount"])
    )


def vp_progression(moves: pl.DataFrame, deltas: pl.DataFrame) -> pl.DataFrame:
    """Mean VP at the end of each round, winners vs losers -- the shape of
    the engine-first / score-late arc."""
    ri = round_index(moves)
    st = final_standings(deltas).select(["game_id", "faction", "won"])
    d = (
        deltas.join(ri, on=["game_id", "row"], how="inner")
        .join(st, on=["game_id", "faction"])
    )
    end = d.group_by(["game_id", "faction", "won", "round"]).agg(
        pl.col("vp_value").max().alias("vp_end")
    )
    return (
        end.group_by(["won", "round"])
        .agg(pl.col("vp_end").mean().alias("mean_vp"), pl.len().alias("n"))
        .sort(["won", "round"])
    )


def action_tempo(moves: pl.DataFrame, deltas: pl.DataFrame) -> pl.DataFrame:
    """Main-track actions per game per round, winners vs losers -- measures
    the action-economy gap."""
    df = _decisions_with_context(moves, deltas).filter(
        pl.col("verb").is_in(_MAIN_VERBS) & (pl.col("round") >= 1)
    )
    counts = df.group_by(["faction", "won", "round", "game_id"]).agg(pl.len().alias("n"))
    per_round = counts.group_by(["won", "round"]).agg(
        (pl.col("n").sum() / pl.len()).alias("actions_per_game")  # per faction-game
    )
    return per_round.sort(["won", "round"])


def _md_table(df: pl.DataFrame, float_digits: int = 2) -> str:
    cols = df.columns
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in df.to_dicts():
        cells = []
        for c in cols:
            v = row[c]
            cells.append(f"{v:.{float_digits}f}" if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_compendium_tables(
    moves: pl.DataFrame, deltas: pl.DataFrame, out_dir: Path
) -> None:
    """One markdown file of win-conditioned tables per faction, plus an
    overview. Numbers only -- the prose playbooks are a separate, human-
    reviewed LLM step (GENERATE.md)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rates = faction_win_rates(moves, deltas)
    curves = timing_curves(moves, deltas)
    openings = opening_patterns(moves, deltas)
    tiles = tile_pick_rates(moves, deltas)

    overview = [
        "# TM corpus overview (win-conditioned statistics)",
        "",
        f"Games: {moves['game_id'].n_unique()}",
        "",
        "## Faction win rates",
        "",
        _md_table(rates),
        "",
        "## Leech accept rate by offer amount (winners vs losers)",
        "",
        _md_table(leech_behavior(moves, deltas)),
        "",
        "## Mean VP by end of round (winners vs losers)",
        "",
        _md_table(vp_progression(moves, deltas)),
        "",
        "## Main-track actions per game per round (winners vs losers)",
        "",
        _md_table(action_tempo(moves, deltas)),
        "",
    ]
    (out_dir / "_overview.md").write_text("\n".join(overview))

    for faction in sorted(rates["faction"].to_list()):
        sections = [f"# {faction}: win-conditioned corpus statistics", ""]
        rate_row = rates.filter(pl.col("faction") == faction)
        sections += ["## Win rate", "", _md_table(rate_row), ""]
        sections += [
            "## Opening patterns (dwellings by end of round 1; mean first-upgrade round)",
            "",
            _md_table(openings.filter(pl.col("faction") == faction)),
            "",
        ]
        sections += [
            "## Timing curves (mean occurrences per game, by round)",
            "",
            _md_table(curves.filter(pl.col("faction") == faction)),
            "",
        ]
        faction_tiles = tiles.filter(pl.col("faction") == faction).filter(
            pl.col("picks_per_game") >= 0.05
        )
        sections += ["## Tile picks (per game)", "", _md_table(faction_tiles), ""]
        (out_dir / f"{faction}.md").write_text("\n".join(sections))
