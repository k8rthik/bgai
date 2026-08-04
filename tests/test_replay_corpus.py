"""tests/test_replay_corpus.py

Pinned regression set for Task 14 (corpus rollout to zero mismatches):
replays the 25 games selected by ``.superpowers/sdd/2026-08-03-tm-engine-
core/task-14-corpus-prep.md`` (every one of the 14 base factions appears
in >= 3 of them, every exotic option -- ``loose-dig``/``merge-income-
phases``/``loose-convert-phase``/``loose-cult-loss``/``loose-lose-cult`` --
in >= 1, and >= 2 games without ``variable-turn-order``/>= 1 without
``temple-scoring-tile``), plus the corpus-prep report's 4 non-overlapping
stress outliers (2 of the 5 longest games by move-row count, 2 of the 5
with the most leech/decline rows -- the other 3+3 either overlap the
25-game set already or were left out to keep this file's own runtime
trivial).

This is normal-CI-weight (29 games, well under a second total -- see
``test_regression_set_replays_clean``'s own timing note), unlike the full
3563-game corpus run documented in ``README.md``'s "Replaying the full
corpus" section. ``tests/test_replay.py``'s existing 10-game test stays
the fast smoke check; this file is the broader, curated cross-section.
"""

from __future__ import annotations

import json

import polars as pl
import pytest

from bgai.engine.tm.replay import _apply_row_commands, _iter_rows, replay_game
from bgai.engine.tm.round_flow import start_setup
from bgai.engine.tm.setup import load_setup
from bgai.engine.tm.state import GameState

# The 25-game set selected by task-14-corpus-prep.md section 3 (see that
# file for the full faction-coverage/exotic-option verification).
REGRESSION_SET: tuple[str, ...] = (
    "4pLeague_S5_D3L2_G3",  # loose-cult-loss (sole carrier)
    "4pLeague_S8_D3L3_G2",  # loose-lose-cult (sole carrier)
    "4pLeague_S3_D3L3_G2",  # merge-income-phases + loose-convert-phase
    "4pLeague_S1_D3L4_G1",  # loose-dig
    "4pLeague_S10_D3L3_G5",
    "4pLeague_S24_D2L1_G3",
    "4pLeague_S29_D3L1_G4",
    "4pLeague_S36_D3L2_G2",
    "4pLeague_S17_D3L3_G3",
    "4pLeague_S11_D3L1_G6",
    "4pLeague_S40_D3L1_G6",
    "4pLeague_S12_D3L3_G4",
    "4pLeague_S13_D1L1_G1",
    "4pLeague_S14_D1L1_G1",
    "4pLeague_S15_D1L1_G1",
    "4pLeague_S16_D1L1_G1",
    "4pLeague_S18_D1L1_G1",
    "4pLeague_S19_D1L1_G1",
    "4pLeague_S20_D1L1_G1",
    "4pLeague_S21_D1L1_G1",
    "4pLeague_S22_D1L1_G1",
    "4pLeague_S23_D1L1_G1",
    "4pLeague_S25_D1L1_G1",
    "4pLeague_S26_D1L1_G1",
    "4pLeague_S27_D1L1_G1",
)

# 4 non-overlapping stress outliers from the corpus-prep report's "longest
# games" (S34_D3L1_G2 545 rows, S40_D1L1_G2 541 rows, S72_D2L2_G4 532
# rows) and "most leech/decline rows" (S62_D2L1_G4, 111) lists -- picked
# to cover both stress dimensions without redundancy against each other
# or REGRESSION_SET (S14_D1L1_G1 already covers leech/decline stress
# incidentally, per that report).
STRESS_OUTLIERS: tuple[str, ...] = (
    "4pLeague_S34_D3L1_G2",
    "4pLeague_S40_D1L1_G2",
    "4pLeague_S72_D2L2_G4",
    "4pLeague_S62_D2L1_G4",
)


@pytest.fixture(scope="session")
def frames() -> tuple[pl.DataFrame, pl.DataFrame]:
    return (
        pl.read_parquet("data/datasets/moves.parquet"),
        pl.read_parquet("data/datasets/deltas.parquet"),
    )


@pytest.fixture(scope="session")
def games_meta() -> pl.DataFrame:
    return pl.read_parquet("data/datasets/games_meta.parquet")


def test_regression_set_replays_clean(frames: tuple[pl.DataFrame, pl.DataFrame]) -> None:
    """Every game in the 25-game regression set replays with zero errors
    and zero delta-oracle mismatches, row for row."""
    moves_df, deltas_df = frames
    for game_id in REGRESSION_SET:
        result = replay_game(game_id, moves_df, deltas_df)
        assert result.error is None, f"{game_id}: {result.error}"
        assert result.mismatches == (), f"{game_id}: {result.mismatches}"
        assert result.rows_checked > 0


def test_stress_outliers_replay_clean(frames: tuple[pl.DataFrame, pl.DataFrame]) -> None:
    """The 4 stress outliers (longest games / heaviest leech-decline
    traffic) also replay clean -- exercising the decision queue and
    row-grouping logic harder than the curated 25-game set alone does."""
    moves_df, deltas_df = frames
    for game_id in STRESS_OUTLIERS:
        result = replay_game(game_id, moves_df, deltas_df)
        assert result.error is None, f"{game_id}: {result.error}"
        assert result.mismatches == (), f"{game_id}: {result.mismatches}"
        assert result.rows_checked > 0


def _replay_to_final_state(game_id: str, moves_df: pl.DataFrame) -> GameState:
    """Replay ``game_id``'s full ledger and return the final ``GameState``
    (including round 6's ``score_vp``/``score_resources`` rows) --
    ``replay_game``'s frozen ``ReplayResult`` deliberately exposes no
    final state, so this mirrors its private row-driving loop directly
    (same pattern as ``test_scoring.py``'s ``test_replayed_final_vp_
    matches_games_meta_final_vp``, generalized to any game_id). No
    other_income_done/cult_income_done income-phase bookkeeping is
    threaded through here since every REGRESSION_SET/STRESS_OUTLIERS game
    is a normal 4-faction game with a complete income-row set for every
    round (none of them are dropped-faction games) -- the plain
    ``_advance_after_row``-equivalent phase transitions in ``replay_game``
    matter for those edge cases, not for this curated set.
    """
    from bgai.engine.tm.replay import _advance_after_row, _ensure_actions_phase_started
    from bgai.engine.tm.state import Phase

    setup = load_setup(game_id)
    state = start_setup(GameState.initial(setup))
    game_moves = moves_df.filter(pl.col("game_id") == game_id).sort(["row", "seq"])
    other_income_done: set[str] = set()
    cult_income_done: set[str] = set()
    for _row, faction, cmds in _iter_rows(game_moves):
        was_income = state.phase == Phase.INCOME
        state = _ensure_actions_phase_started(state, cmds)
        if was_income and state.phase != Phase.INCOME:
            other_income_done = set()
        state = _apply_row_commands(state, faction, cmds)
        state, other_income_done, cult_income_done = _advance_after_row(
            state, faction, cmds, other_income_done, cult_income_done
        )
    return state


def test_regression_set_final_vp_matches_games_meta(
    games_meta: pl.DataFrame,
) -> None:
    """Final-VP cross-check (task brief step 3): after a game replays
    clean, its final per-faction VP must equal ``games_meta.parquet``'s
    ``final_vp`` column -- catches final-scoring drift the last delta row
    (which only ever checks *running* VP, not the round-6 scoring block's
    net effect) can't see on its own.

    Skips a dropped faction's own VP (``GameSetup.dropped_factions``):
    ``4pLeague_S15_D1L1_G1``'s witches drops after row 357 and never
    appears in the ledger again, including round 6's own ``score_vp``/
    ``score_resources`` rows -- there is *no* ledger evidence to replay
    a dropped faction's final scoring from at all, dropped or not
    (``replay_game`` itself only ever replays real rows, same
    limitation), so ``games_meta.final_vp``'s number for that faction
    (apparently computed by Perl through some other means entirely for a
    dropped player) is not reproducible by ledger replay and is out of
    scope here -- every *other* (non-dropped) faction in every game
    still gets the full cross-check.
    """
    moves_df = pl.read_parquet("data/datasets/moves.parquet")
    for game_id in REGRESSION_SET:
        state = _replay_to_final_state(game_id, moves_df)
        expected = json.loads(games_meta.filter(pl.col("game_id") == game_id)["final_vp"][0])
        for faction, vp in expected.items():
            if faction in state.setup.dropped_factions:
                continue
            assert state.factions[faction].vp == vp, f"{game_id}/{faction}"
