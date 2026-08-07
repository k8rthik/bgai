"""Incremental ingest: idempotency, quarantine, and crawler-race safety."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import orjson
import polars as pl

from bgai.data.ingest import compact, parsed_ids, quarantined_ids, run_pass


def _write_game(raw_dir: Path, game_id: str, commands: list[str]) -> None:
    """A minimal crawled game: one ledger row per command string."""
    # every resource cell is {"value": ..., "delta": ...}, as in the real feed
    ledger = [
        {
            "faction": "witches",
            "commands": command,
            "VP": {"value": 20 + i, "delta": 1},
            "C": {"value": 15, "delta": 0},
            "W": {"value": 3, "delta": 0},
            "P": {"value": 0, "delta": 0},
            "PW": {"value": "5/7/0", "delta": 0},
            "CULT": {"value": "0/0/0/0", "delta": 0},
        }
        for i, command in enumerate(commands)
    ]
    payload = {
        "ledger": ledger,
        "factions": {"witches": {"VP": 100}},
        "players": [{"username": "a"}],
        "options": {"strict-leech": 1},
    }
    raw_dir.mkdir(parents=True, exist_ok=True)
    with gzip.open(raw_dir / f"{game_id}.json.gz", "wb") as handle:
        handle.write(orjson.dumps(payload))


def test_pass_parses_games_into_parts(tmp_path: Path) -> None:
    raw, parts = tmp_path / "raw", tmp_path / "parts"
    _write_game(raw, "g1", ["build A1", "pass BON3"])

    stats = run_pass(raw, parts)

    assert stats.parsed == 1 and stats.quarantined == 0 and stats.parts_written == 1
    assert parsed_ids(parts) == {"g1"}
    moves = pl.read_parquet(parts / "moves-0000.parquet")
    assert set(moves["verb"]) == {"build", "pass"}


def test_pass_is_idempotent(tmp_path: Path) -> None:
    """Re-running must do nothing -- the pass is looped every few minutes."""
    raw, parts = tmp_path / "raw", tmp_path / "parts"
    _write_game(raw, "g1", ["build A1"])
    run_pass(raw, parts)

    again = run_pass(raw, parts)

    assert again.parsed == 0 and again.parts_written == 0
    assert len(list(parts.glob("moves-*.parquet"))) == 1


def test_unparseable_game_is_quarantined_not_fatal(tmp_path: Path) -> None:
    """One odd log must not destroy a pass over tens of thousands of good ones."""
    raw, parts = tmp_path / "raw", tmp_path / "parts"
    _write_game(raw, "good", ["build A1"])
    _write_game(raw, "weird", ["frobnicate the gizmo"])

    stats = run_pass(raw, parts)

    assert stats.parsed == 1 and stats.quarantined == 1
    assert parsed_ids(parts) == {"good"}
    assert quarantined_ids(parts) == {"weird"}
    # the aggregate report is what tells us which grammar rule to write
    assert "frobnicate the gizmo" in stats.unknown_commands


def test_quarantined_game_is_retried_every_pass(tmp_path: Path) -> None:
    """Quarantine is derived, not a tombstone -- otherwise a grammar fix could
    never rescue the games that motivated it."""
    raw, parts = tmp_path / "raw", tmp_path / "parts"
    _write_game(raw, "weird", ["frobnicate the gizmo"])
    run_pass(raw, parts)

    again = run_pass(raw, parts)

    assert again.quarantined == 1, "a still-broken game must stay quarantined"
    assert quarantined_ids(parts) == {"weird"}


def test_quarantine_drains_once_the_game_can_be_parsed(tmp_path: Path) -> None:
    """The whole point of the loop: fix the grammar, re-run, the game lands."""
    raw, parts = tmp_path / "raw", tmp_path / "parts"
    _write_game(raw, "weird", ["frobnicate the gizmo"])
    assert run_pass(raw, parts).quarantined == 1

    # stand in for a grammar fix by making the game parseable
    _write_game(raw, "weird", ["build A1"])
    stats = run_pass(raw, parts)

    assert stats.parsed == 1 and stats.quarantined == 0
    assert quarantined_ids(parts) == set(), "quarantine file must not linger"
    assert "weird" in parsed_ids(parts)


def test_half_written_game_is_skipped_not_quarantined(tmp_path: Path) -> None:
    """The crawler writes concurrently; a truncated gzip is a race, not a defect."""
    raw, parts = tmp_path / "raw", tmp_path / "parts"
    raw.mkdir(parents=True)
    (raw / "inflight.json.gz").write_bytes(b"\x1f\x8b\x08\x00 truncated")

    stats = run_pass(raw, parts)

    assert stats.skipped == 1 and stats.quarantined == 0
    assert quarantined_ids(parts) == set(), "an in-flight write must stay retryable"


def test_completed_game_is_picked_up_on_a_later_pass(tmp_path: Path) -> None:
    raw, parts = tmp_path / "raw", tmp_path / "parts"
    raw.mkdir(parents=True)
    (raw / "g1.json.gz").write_bytes(b"\x1f\x8b\x08\x00 truncated")
    assert run_pass(raw, parts).skipped == 1

    _write_game(raw, "g1", ["build A1"])  # crawler finishes the write

    assert run_pass(raw, parts).parsed == 1


def test_parts_split_by_size(tmp_path: Path) -> None:
    raw, parts = tmp_path / "raw", tmp_path / "parts"
    for n in range(5):
        _write_game(raw, f"g{n}", ["build A1"])

    stats = run_pass(raw, parts, games_per_part=2)

    assert stats.parts_written == 3  # 2 + 2 + 1
    assert parsed_ids(parts) == {f"g{n}" for n in range(5)}


def test_compact_merges_parts_into_canonical_files(tmp_path: Path) -> None:
    raw, parts, out = tmp_path / "raw", tmp_path / "parts", tmp_path / "out"
    _write_game(raw, "g1", ["build A1"])
    _write_game(raw, "g2", ["pass BON3"])
    run_pass(raw, parts)
    out.mkdir()

    compact(parts, out)

    moves = pl.read_parquet(out / "moves.parquet")
    assert set(moves["game_id"].unique()) == {"g1", "g2"}
    assert (out / "deltas.parquet").exists() and (out / "games_meta.parquet").exists()


def test_compact_extends_rather_than_replaces_the_corpus(tmp_path: Path) -> None:
    """A later compaction must keep games an earlier one already merged in --
    otherwise each compaction would silently shrink the corpus to the newest
    parts."""
    raw, parts, out = tmp_path / "raw", tmp_path / "parts", tmp_path / "out"
    out.mkdir()
    _write_game(raw, "first", ["build A1"])
    run_pass(raw, parts)
    compact(parts, out)

    for part in parts.glob("*.parquet"):  # parts already merged; next batch arrives
        part.unlink()
    _write_game(raw, "second", ["pass BON3"])
    run_pass(raw, parts)
    compact(parts, out)

    ids = set(pl.read_parquet(out / "moves.parquet")["game_id"].unique())
    assert ids == {"first", "second"}


def test_quarantine_file_is_readable_jsonl(tmp_path: Path) -> None:
    raw, parts = tmp_path / "raw", tmp_path / "parts"
    _write_game(raw, "weird", ["frobnicate the gizmo"])
    run_pass(raw, parts)

    lines = (parts / "_quarantine.jsonl").read_text().splitlines()
    entry = json.loads(lines[0])
    assert entry["game_id"] == "weird" and "unparseable" in entry["reason"]
