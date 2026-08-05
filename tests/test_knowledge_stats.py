"""Corpus stats mining for the rung-4 compendium (LLM-harness plan, task 12)."""

import polars as pl
import pytest

from bgai.knowledge.stats import (
    faction_win_rates,
    final_standings,
    opening_patterns,
    round_index,
    tile_pick_rates,
    timing_curves,
    write_compendium_tables,
)


def _moves_row(game_id, row, faction, verb, **kw):
    base = dict(
        game_id=game_id, row=row, seq=0, faction=faction, kind="DECISION", verb=verb,
        loc=None, loc2=None, building=None, tile=None, cult=None, color=None,
        target=None, reason=None, res1=None, res2=None, n1=None, n2=None,
    )
    base.update(kw)
    return base


@pytest.fixture()
def fixture_frames():
    rows = []
    for game in ("G_A", "G_B"):
        rows += [
            _moves_row(game, 1, "x", "build", loc="A1"),
            _moves_row(game, 2, "y", "build", loc="B1"),
            _moves_row(game, 3, "x", "pass", tile="BON1"),
            _moves_row(game, 4, "y", "pass", tile="BON2"),
            # round 1
            _moves_row(game, 10, "x", "other_income_for_faction", kind="INCOME"),
            _moves_row(game, 11, "y", "other_income_for_faction", kind="INCOME"),
            _moves_row(game, 12, "x", "build", loc="A2"),
            _moves_row(game, 13, "y", "upgrade", loc="B1", building="TP"),
            _moves_row(game, 14, "y", "pass", tile="BON3"),
            _moves_row(game, 15, "x", "pass", tile="BON1"),
            # round 2
            _moves_row(game, 20, "x", "other_income_for_faction", kind="INCOME"),
            _moves_row(game, 21, "y", "other_income_for_faction", kind="INCOME"),
            _moves_row(game, 22, "x", "send", cult="FIRE"),
            _moves_row(game, 23, "y", "gain_favor", tile="FAV11"),
        ]
    moves = pl.DataFrame(rows)
    deltas = pl.DataFrame(
        [
            dict(game_id="G_A", row=30, faction="x", vp_value=50),
            dict(game_id="G_A", row=30, faction="y", vp_value=40),
            dict(game_id="G_A", row=5, faction="x", vp_value=20),
            dict(game_id="G_B", row=30, faction="x", vp_value=30),
            dict(game_id="G_B", row=30, faction="y", vp_value=60),
        ]
    )
    return moves, deltas


def test_final_standings(fixture_frames):
    _moves, deltas = fixture_frames
    st = final_standings(deltas)
    rec = {(r["game_id"], r["faction"]): r for r in st.to_dicts()}
    assert rec[("G_A", "x")]["final_vp"] == 50
    assert rec[("G_A", "x")]["won"] is True
    assert rec[("G_A", "y")]["won"] is False
    assert rec[("G_B", "y")]["won"] is True


def test_round_index_boundaries(fixture_frames):
    moves, _deltas = fixture_frames
    ri = {(r["game_id"], r["row"]): r["round"] for r in round_index(moves).to_dicts()}
    assert ri[("G_A", 1)] == 0  # setup
    assert ri[("G_A", 12)] == 1
    assert ri[("G_A", 15)] == 1
    assert ri[("G_A", 22)] == 2


def test_faction_win_rates(fixture_frames):
    moves, deltas = fixture_frames
    wr = {r["faction"]: r for r in faction_win_rates(moves, deltas).to_dicts()}
    assert wr["x"]["games"] == 2
    assert wr["x"]["win_rate"] == pytest.approx(0.5)
    assert wr["y"]["mean_final_vp"] == pytest.approx(50.0)


def test_timing_curves_cell(fixture_frames):
    moves, deltas = fixture_frames
    tc = timing_curves(moves, deltas)
    cell = tc.filter(
        (pl.col("faction") == "x") & pl.col("won") & (pl.col("round") == 1)
        & (pl.col("verb") == "build")
    )
    # x won only G_A; in its round 1 it built once.
    assert cell["mean_per_game"].to_list() == [1.0]


def test_opening_and_tiles_run(fixture_frames):
    moves, deltas = fixture_frames
    op = opening_patterns(moves, deltas)
    assert {"faction", "won"} <= set(op.columns)
    tp = tile_pick_rates(moves, deltas)
    assert "tile" in tp.columns


def test_write_tables(tmp_path, fixture_frames):
    moves, deltas = fixture_frames
    write_compendium_tables(moves, deltas, tmp_path)
    assert (tmp_path / "_overview.md").exists()
    x_md = (tmp_path / "x.md").read_text()
    assert "win" in x_md.lower()


@pytest.mark.slow
def test_real_corpus_tables(tmp_path):
    moves = pl.read_parquet("data/datasets/moves.parquet")
    deltas = pl.read_parquet("data/datasets/deltas.parquet")
    write_compendium_tables(moves, deltas, tmp_path)
    darklings = (tmp_path / "darklings.md").read_text()
    assert "win rate" in darklings.lower()
