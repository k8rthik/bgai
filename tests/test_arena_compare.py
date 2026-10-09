"""Head-to-head comparison statistics over saved arena results.

A candidate checkpoint is judged against a baseline by the arena's
mirrored tables: every table is one setup played in 4 seat rotations, so
table-level differences are the paired unit and seeds are pooled by
concatenating tables, never by averaging summary percentages.
"""

from __future__ import annotations

import json
import math

import pytest

from bgai.arena.compare import (
    compare_runs,
    head_to_head,
    load_run,
    render_table,
    series_to_run,
)


def _game(table: int, cand_ranks: tuple[int, int], base_ranks: tuple[int, int],
          error: str | None = None) -> dict:
    """One game: the candidate holds seats a,b and the baseline c,d."""
    ranks = dict(zip("abcd", (*cand_ranks, *base_ranks), strict=True))
    return {
        "table": table,
        "seats": {"a": "cand", "b": "cand", "c": "base", "d": "base"},
        "ranks": ranks,
        "vps": {f: 120 - 10 * r for f, r in ranks.items()},
        "error": error,
    }


def _run(games: list[dict], seed: int = 1) -> dict:
    return {"seed": seed, "tables": 1 + max(g["table"] for g in games), "games": games}


# ---------------------------------------------------------------- head_to_head


def test_even_games_give_share_one_half_and_zero_difference() -> None:
    games = [_game(t, (0, 3), (1, 2)) for t in range(4)] + [
        _game(t, (1, 2), (0, 3)) for t in range(4)
    ]
    h = head_to_head([_run(games)], "cand", "base")
    assert h["cand"]["win_rate"] == pytest.approx(0.25)
    assert h["base"]["win_rate"] == pytest.approx(0.25)
    assert h["share"] == pytest.approx(0.5)
    assert h["win_diff"] == pytest.approx(0.0)
    assert h["place_diff"] == pytest.approx(0.0)


def test_candidate_that_always_wins_has_full_share_and_positive_differences() -> None:
    games = [_game(t, (0, 1), (2, 3)) for t in range(10)]
    h = head_to_head([_run(games)], "cand", "base")
    assert h["share"] == pytest.approx(1.0)
    assert h["win_diff"] == pytest.approx(0.5)  # 1 of 2 seats wins vs 0 of 2
    assert h["place_diff"] > 0  # positive = candidate places better


def test_errored_games_are_excluded() -> None:
    games = [_game(0, (0, 1), (2, 3)), _game(0, (2, 3), (0, 1), error="engine rejection")]
    h = head_to_head([_run(games)], "cand", "base")
    assert h["games"] == 1
    assert h["errors"] == 1
    assert h["share"] == pytest.approx(1.0)


def test_paired_standard_error_uses_tables_not_seats() -> None:
    # tables alternate a clean candidate win and a clean baseline win:
    # per-table win_diff is +0.5 / -0.5, so SE = sd/sqrt(T) over 8 tables
    games = [_game(t, (0, 1), (2, 3)) if t % 2 == 0 else _game(t, (2, 3), (0, 1))
             for t in range(8)]
    h = head_to_head([_run(games)], "cand", "base")
    sd = math.sqrt(sum(0.25 for _ in range(8)) / 7)
    assert h["win_diff"] == pytest.approx(0.0)
    assert h["win_diff_se"] == pytest.approx(sd / math.sqrt(8))
    assert h["tables"] == 8


def test_seeds_pool_by_tables_and_keep_tables_distinct() -> None:
    run_a = _run([_game(t, (0, 1), (2, 3)) for t in range(3)], seed=1)
    run_b = _run([_game(t, (2, 3), (0, 1)) for t in range(3)], seed=2)
    h = head_to_head([run_a, run_b], "cand", "base")
    assert h["tables"] == 6  # table 0 of seed 1 is not table 0 of seed 2
    assert h["share"] == pytest.approx(0.5)
    assert h["seeds"] == [1, 2]


def test_unknown_agent_is_an_error() -> None:
    with pytest.raises(ValueError, match="nobody"):
        head_to_head([_run([_game(0, (0, 1), (2, 3))])], "nobody", "base")


def test_single_table_reports_no_standard_error() -> None:
    h = head_to_head([_run([_game(0, (0, 1), (2, 3))])], "cand", "base")
    assert h["win_diff_se"] is None
    assert h["win_diff_z"] is None


# ---------------------------------------------------------------- compare_runs


def test_compare_runs_groups_by_candidate_label() -> None:
    def labelled(label: str, seed: int, cand_wins: bool) -> dict:
        games = [_game(t, (0, 1), (2, 3)) if cand_wins else _game(t, (2, 3), (0, 1))
                 for t in range(4)]
        for g in games:
            g["seats"] = {f: (label if a == "cand" else "leg5b") for f, a in g["seats"].items()}
        return {**_run(games, seed=seed), "agents": [label, "leg5b"],
                "meta": {"approx_games_trained": 1000 * seed}}

    runs = [labelled("it1000", 1, True), labelled("it1000", 2, False),
            labelled("it3000", 1, True)]
    rows = compare_runs(runs, baseline="leg5b")
    assert [r["candidate"] for r in rows] == ["it1000", "it3000"]
    assert rows[0]["share"] == pytest.approx(0.5)
    assert rows[0]["seeds"] == [1, 2]
    assert rows[1]["share"] == pytest.approx(1.0)


def test_compare_runs_without_baseline_in_a_run_is_an_error() -> None:
    run = {**_run([_game(0, (0, 1), (2, 3))]), "agents": ["cand", "base"]}
    with pytest.raises(ValueError, match="leg5b"):
        compare_runs([run], baseline="leg5b")


def test_render_table_lists_every_candidate() -> None:
    games = [_game(t, (0, 1), (2, 3)) for t in range(4)]
    run = {**_run(games), "agents": ["cand", "base"]}
    text = render_table(compare_runs([run], baseline="base"))
    assert "cand" in text
    assert "share" in text


# ---------------------------------------------------------------- serialisation


class _Result:
    def __init__(self, seats, ranks, vps, error=None):
        self.seats, self.ranks, self.vps, self.error = seats, ranks, vps, error
        self.setup_game_id = "g"
        self.decisions = 1


class _Series:
    def __init__(self, results):
        self.results = tuple(results)
        self.n_errors = sum(1 for r in results if r.error)


def test_series_to_run_tags_tables_in_rotation_order() -> None:
    g = _game(0, (0, 1), (2, 3))
    results = [_Result(g["seats"], g["ranks"], g["vps"]) for _ in range(8)]
    run = series_to_run(_Series(results), seed=7, agents_spec="x", meta={"k": 1})
    assert [game["table"] for game in run["games"]] == [0, 0, 0, 0, 1, 1, 1, 1]
    assert run["seed"] == 7
    assert run["tables"] == 2
    assert run["agents"] == ["cand", "base"]
    assert run["meta"] == {"k": 1}


def test_series_to_run_rejects_a_partial_table() -> None:
    g = _game(0, (0, 1), (2, 3))
    results = [_Result(g["seats"], g["ranks"], g["vps"]) for _ in range(5)]
    with pytest.raises(ValueError, match="multiple of 4"):
        series_to_run(_Series(results), seed=7, agents_spec="x", meta={})


def test_load_run_round_trips_and_validates(tmp_path) -> None:
    path = tmp_path / "r.json"
    path.write_text(json.dumps(_run([_game(0, (0, 1), (2, 3))])))
    assert load_run(path)["seed"] == 1
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"seed": 1}))
    with pytest.raises(ValueError, match="games"):
        load_run(bad)


# ---------------------------------------------------------------- arena CLI


def test_arena_cli_writes_a_comparable_run(tmp_path) -> None:
    from bgai.arena.run import main

    out = tmp_path / "run.json"
    main(["--tables", "1", "--seed", "3", "--agents", "random,greedy",
          "--json", str(out), "--meta", "approx_games_trained=96000", "--meta", "label=x"])
    run = load_run(out)
    assert run["seed"] == 3
    assert len(run["games"]) == 4
    assert run["agents"] == ["random", "greedy"]
    assert run["meta"] == {"approx_games_trained": 96000, "label": "x"}
    rows = compare_runs([run], baseline="greedy")
    assert rows[0]["candidate"] == "random"


def test_arena_cli_rejects_malformed_meta(tmp_path) -> None:
    from bgai.arena.run import main

    with pytest.raises(SystemExit, match="meta"):
        main(["--tables", "1", "--agents", "random,greedy",
              "--json", str(tmp_path / "r.json"), "--meta", "novalue"])


def test_render_brief_is_one_line_per_candidate() -> None:
    from bgai.arena.compare import render_brief

    games = [_game(t, (0, 1), (2, 3)) if t % 2 else _game(t, (2, 3), (0, 1))
             for t in range(6)]
    run = {**_run(games), "agents": ["cand", "base"]}
    lines = render_brief(compare_runs([run], baseline="base")).splitlines()
    assert len(lines) == 1
    assert lines[0].startswith("cand vs base:")
    assert "share 0.500" in lines[0]
    assert "1 seed" in lines[0]
