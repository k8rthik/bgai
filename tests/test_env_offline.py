"""Offline-RL export from the logged human corpus.

Needs the gitignored ``data/datasets/*.parquet``, so it skips with a clear
message on a fresh clone rather than failing.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from bgai.env.config import MAX_CANDIDATES, RewardMode
from bgai.env.observation import GLOBAL_DIM, HEX_FEAT_DIM, HEXES, OBS_VERSION
from bgai.env.offline import (
    DEFAULT_DELTAS,
    DEFAULT_META,
    DEFAULT_MOVES,
    MASK_SOURCE,
    build,
)
from bgai.env.offline_shards import iter_transitions, load_shard
from bgai.env.reward import terminal_rewards
from bgai.training.encode_move import MOVE_FIELDS

pytestmark = pytest.mark.skipif(
    not (DEFAULT_MOVES.exists() and DEFAULT_DELTAS.exists() and DEFAULT_META.exists()),
    reason="offline export needs the gitignored corpus parquets in data/datasets/",
)

GAMES = 6


@pytest.fixture(scope="module")
def export(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("offline")
    stats = build(
        out,
        limit=GAMES,
        max_candidates=MAX_CANDIDATES,
        games_per_shard=GAMES,
        verbose=False,
    )
    assert stats.games == GAMES
    assert stats.shard_count == 1
    assert stats.failed_games == ()
    assert stats.record_count > 1000
    return out


def test_manifest_records_what_was_exported(export: Path) -> None:
    manifest = json.loads((export / "manifest.json").read_text())
    assert manifest["obs_version"] == OBS_VERSION
    assert manifest["games"] == GAMES
    assert manifest["max_candidates"] == MAX_CANDIDATES
    assert manifest["reward_mode"] == RewardMode.TERMINAL_VP_SHARE.value
    assert sum(s["records"] for s in manifest["shards"]) == manifest["record_count"]


def test_manifest_states_the_mask_source_honestly(export: Path) -> None:
    """The offline mask is the full legal set; the online mask is the
    driver's turn-protocol-filtered subset. That difference has to be on the
    record, not discovered later (docs/decisions.md C3 is what it costs).
    """
    manifest = json.loads((export / "manifest.json").read_text())
    assert manifest["mask_source"] == MASK_SOURCE
    assert "driver.decision" in manifest["mask_note"]


def test_every_logged_command_matched_a_legal_candidate(export: Path) -> None:
    """The containment sweep predicts zero unmatched; a nonzero count here
    is a ``legal_moves`` soundness regression, not data noise.
    """
    manifest = json.loads((export / "manifest.json").read_text())
    assert manifest["unmatched_commands"] == 0


def test_shard_arrays_have_the_env_observation_shapes(export: Path) -> None:
    manifest = json.loads((export / "manifest.json").read_text())
    shard = load_shard(export / manifest["shards"][0]["file"])
    n = shard["action"].shape[0]
    assert shard["hex_planes"].shape == (n, len(HEXES), HEX_FEAT_DIM)
    assert shard["globals"].shape == (n, GLOBAL_DIM)
    assert shard["cand_flat"].shape == (int(shard["cand_counts"].sum()), MOVE_FIELDS)
    assert shard["reward"].dtype == np.float32
    assert shard["done"].dtype == bool
    assert (shard["cand_counts"] >= 2).all()
    assert (shard["action"] < shard["cand_counts"]).all(), "chosen must be legal"


def test_one_terminal_transition_per_seat_per_game(export: Path) -> None:
    """Per-seat MDPs: each of the four seats has exactly one terminal
    transition in each game, so ``done`` count == seats x games.
    """
    manifest = json.loads((export / "manifest.json").read_text())
    shard = load_shard(export / manifest["shards"][0]["file"])
    assert int(shard["done"].sum()) == 4 * GAMES
    for game in range(GAMES):
        rows = shard["game"] == game
        assert int(shard["done"][rows].sum()) == 4
        assert set(shard["seat"][rows].tolist()) == {0, 1, 2, 3}


def test_next_index_points_at_the_same_seats_next_decision(export: Path) -> None:
    """The load-bearing property: ``next_obs`` must be *this* seat's next
    decision, not whatever seat happened to move next.
    """
    manifest = json.loads((export / "manifest.json").read_text())
    shard = load_shard(export / manifest["shards"][0]["file"])
    nxt = shard["next_index"]
    assert (nxt[shard["done"]] == -1).all()
    live = ~shard["done"]
    assert (nxt[live] >= 0).all()
    assert (shard["seat"][nxt[live]] == shard["seat"][live]).all()
    assert (shard["game"][nxt[live]] == shard["game"][live]).all()
    assert (nxt[live] > np.flatnonzero(live)).all(), "trajectories go forward"


def test_materialized_records_are_the_promised_tuple(export: Path) -> None:
    manifest = json.loads((export / "manifest.json").read_text())
    records = list(
        iter_transitions(export / manifest["shards"][0]["file"], MAX_CANDIDATES)
    )
    assert len(records) == manifest["record_count"]
    first = records[0]
    assert set(first) == {"obs", "action", "mask", "reward", "next_obs", "done"}
    assert set(first["obs"]) == {
        "hex_planes",
        "globals",
        "candidates",
        "action_mask",
    }
    assert first["mask"].shape == (MAX_CANDIDATES,)
    assert first["mask"].dtype == np.int8
    assert first["obs"]["candidates"].shape == (MAX_CANDIDATES, MOVE_FIELDS)
    assert np.array_equal(first["mask"], first["obs"]["action_mask"])


def test_materialized_observations_satisfy_the_gymnasium_space(export: Path) -> None:
    from bgai.env.spaces import observation_space

    space = observation_space(MAX_CANDIDATES)
    manifest = json.loads((export / "manifest.json").read_text())
    for index, record in enumerate(
        iter_transitions(export / manifest["shards"][0]["file"], MAX_CANDIDATES)
    ):
        obs = record["obs"]
        payload = {
            "observation": {
                "hex_planes": obs["hex_planes"],
                "globals": obs["globals"],
                "candidates": obs["candidates"],
            },
            "action_mask": obs["action_mask"],
        }
        assert space.contains(payload), f"record {index} outside the space"
        if index > 50:
            break


def test_chosen_action_is_always_inside_its_own_mask(export: Path) -> None:
    manifest = json.loads((export / "manifest.json").read_text())
    for record in iter_transitions(
        export / manifest["shards"][0]["file"], MAX_CANDIDATES
    ):
        assert record["mask"][record["action"]] == 1


def test_terminal_rewards_match_the_reward_module(export: Path) -> None:
    """The exporter must not grow its own copy of the reward rule."""
    import polars as pl

    from bgai.arena.setups import clean_game_ids

    manifest = json.loads((export / "manifest.json").read_text())
    shard = load_shard(export / manifest["shards"][0]["file"])
    ids = list(clean_game_ids())[:GAMES]
    meta = pl.read_parquet(DEFAULT_META).filter(pl.col("game_id").is_in(ids))
    for game_index, game_id in enumerate(ids):
        row = meta.filter(pl.col("game_id") == game_id)
        if row.height == 0:
            continue
        vps = {k: int(v) for k, v in json.loads(row["final_vp"][0]).items()}
        expected = terminal_rewards(RewardMode.TERMINAL_VP_SHARE, vps)
        rows = (shard["game"] == game_index) & shard["done"]
        got = sorted(round(float(r), 6) for r in shard["reward"][rows])
        assert got == sorted(round(v, 6) for v in expected.values())


def test_reward_mode_is_honoured(tmp_path: Path) -> None:
    stats = build(
        tmp_path,
        limit=2,
        reward_mode=RewardMode.TERMINAL_RANK,
        max_candidates=MAX_CANDIDATES,
        games_per_shard=2,
        verbose=False,
    )
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["reward_mode"] == RewardMode.TERMINAL_RANK.value
    shard = load_shard(tmp_path / manifest["shards"][0]["file"])
    terminals = shard["reward"][shard["done"]]
    assert terminals.min() >= -1.0 and terminals.max() <= 1.0
    assert stats.record_count > 0


def test_explicit_game_ids_are_honoured(tmp_path: Path) -> None:
    from bgai.arena.setups import clean_game_ids

    wanted = list(clean_game_ids())[:2]
    build(
        tmp_path,
        game_ids=wanted,
        max_candidates=MAX_CANDIDATES,
        games_per_shard=5,
        verbose=False,
    )
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["games"] == 2


def test_empty_game_list_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no games to export"):
        build(tmp_path, game_ids=[], max_candidates=MAX_CANDIDATES)


def test_iter_transitions_refuses_a_too_narrow_action_space(export: Path) -> None:
    manifest = json.loads((export / "manifest.json").read_text())
    with pytest.raises(ValueError, match="more than max_candidates"):
        list(iter_transitions(export / manifest["shards"][0]["file"], 2))


def test_cli_runs(tmp_path: Path, capsys) -> None:
    from bgai.env.offline import main

    main(["--out", str(tmp_path), "--limit", "2", "--games-per-shard", "2"])
    out = capsys.readouterr().out
    assert "transitions" in out
    assert "unmatched=0" in out


def test_shard_io_is_reachable_from_one_import() -> None:
    """``offline_shards`` is a separate module so the replay side and the
    I/O side need not import each other, but a consumer should still only
    have to import ``bgai.env.offline``.
    """
    import bgai.env.offline as offline
    import bgai.env.offline_shards as shards

    assert offline.iter_transitions is shards.iter_transitions
    assert offline.load_shard is shards.load_shard
    for name in offline.__all__:
        assert hasattr(offline, name), name


def test_the_mask_is_derived_from_cand_counts_not_stored(export: Path) -> None:
    """A stored mask would be a second, divergeable copy of cand_counts."""
    manifest = json.loads((export / "manifest.json").read_text())
    shard = load_shard(export / manifest["shards"][0]["file"])
    assert "action_mask" not in shard
    records = list(
        iter_transitions(export / manifest["shards"][0]["file"], MAX_CANDIDATES)
    )
    for row, record in enumerate(records[:40]):
        assert int(record["mask"].sum()) == int(shard["cand_counts"][row])
