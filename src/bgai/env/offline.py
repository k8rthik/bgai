"""Offline-RL export: logged human games -> (obs, action, mask, reward,
next_obs, done) records in TM-Env's observation layout.

    uv run python -m bgai.env.offline --out data/datasets/offline_rl --limit 200

Where the transitions come from
------------------------------
Straight through the engine: every clean corpus game is replayed with
``engine.tm.replay.replay_game``, whose ``on_decision`` hook hands over
the pre-apply ``GameState``, the acting faction, and the ledger command it
actually played -- the same path ``training/extract.py`` uses to build the
imitation shards, so the decision set is identical and comparable.

**Per-seat MDPs, not one interleaved stream.** Terra Mystica is
turn-based and non-zero-sum, so a seat's ``next_obs`` is *its own* next
decision, not whatever seat happened to move next. One game therefore
yields ``player_count`` trajectories, each terminal on that seat's last
decision. Interleaving them would make ``next_obs`` the wrong seat's
board and silently corrupt every bootstrapped value target.

Honest caveat about the mask
----------------------------
The mask here is ``canonical_moves(legal_moves_for(state, faction))`` --
the engine's full legal set for that faction at that state, which is what
the human was choosing among and what the imitation shards record. The
*online* env's mask is ``driver.decision``'s offer, which is a **subset**:
the driver enforces a turn protocol (one fresh main action, then
continuations, at most ``FREE_ACTIONS_PER_TURN`` free converts) that a
human ledger row does not expose, because a human submits a whole turn as
one row. So offline and online action sets differ in a specific,
documented way, and that difference is the same train/inference mismatch
that cost this project 30 VP once already (docs/decisions.md C3). It is
recorded in the manifest as ``mask_source`` rather than papered over.

On-disk shape
-------------
``next_obs`` is stored as an *index* (``next_index``, -1 when terminal)
rather than a second copy of the arrays: an observation is ~3.3 kB, so
duplicating it would double a multi-gigabyte export for no information.
:func:`load_shard` and :func:`iter_transitions` give back real
``(obs, action, mask, reward, next_obs, done)`` records.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from bgai.data.ledger_parser import ParsedCommand
from bgai.engine.tm.legal import legal_moves_for
from bgai.engine.tm.legal_match import match_index
from bgai.engine.tm.replay import replay_game
from bgai.engine.tm.state import GameState
from bgai.env.canonical import canonical_offer
from bgai.env.config import MAX_SEATS, RewardMode
from bgai.env.observation import (
    OBS_VERSION,
    Observation,
    encode_candidates,
    encode_globals,
    encode_hex_planes,
    seat_order,
)
from bgai.env.reward import dense_step_rewards, terminal_rewards
from bgai.training.encode_move import MOVE_FIELDS
from bgai.training.vocab import FACTION_INDEX

__all__ = [
    "ExportStats",
    "OfflineTransition",
    "build",
    "export_game",
    "iter_transitions",
    "load_shard",
    "main",
]

GAMES_PER_SHARD = 50
"""Smaller than ``dataset_build.GAMES_PER_SHARD`` (200) because an offline
record carries a reward, a done flag and a next-index on top of the
imitation record, and 50 games keeps a shard comfortably under 1 GB
decompressed."""

MASK_SOURCE = "legal_moves_for+canonical_offer"
DEFAULT_MOVES = Path("data/datasets/moves.parquet")
DEFAULT_DELTAS = Path("data/datasets/deltas.parquet")
DEFAULT_META = Path("data/datasets/games_meta.parquet")


@dataclass(frozen=True)
class OfflineTransition:
    """One logged human decision as an RL transition."""

    obs: Observation
    action: int
    reward: float
    next_obs: Observation | None
    done: bool
    faction: str
    seat: int
    game_id: str
    position: int
    """Index of this decision within its game's decision stream -- the
    stable key the shard writer builds ``next_index`` from (an identity or
    array comparison would be fragile; a position is not)."""
    next_position: int = -1

    @property
    def mask(self) -> np.ndarray:
        """The legal mask -- the observation's own, by construction."""
        return self.obs.action_mask


@dataclass(frozen=True)
class ExportStats:
    shard_count: int
    record_count: int
    games: int
    failed_games: tuple[str, ...]
    unmatched: int
    skipped_single_candidate: int


# --------------------------------------------------------------------------
# per-game export
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _RawDecision:
    faction: str
    hex_planes: np.ndarray
    globals: np.ndarray
    candidates: np.ndarray
    mask: np.ndarray
    action: int
    vp_before: dict[str, int]


def _encode_decision(
    state: GameState,
    faction: str,
    offer: tuple[ParsedCommand, ...],
    chosen: int,
    max_candidates: int,
) -> _RawDecision:
    order = seat_order(state, faction)
    seat_of = {name: i for i, name in enumerate(order)}
    candidates, mask = encode_candidates(state, faction, offer, max_candidates)
    return _RawDecision(
        faction=faction,
        hex_planes=encode_hex_planes(state, seat_of),
        globals=encode_globals(state, order, len(offer)),
        candidates=candidates,
        mask=mask,
        action=chosen,
        vp_before={f: state.factions[f].vp for f in state.setup.factions},
    )


def export_game(
    game_id: str,
    moves_df: pl.DataFrame,
    deltas_df: pl.DataFrame,
    final_vps: dict[str, int],
    *,
    reward_mode: RewardMode = RewardMode.TERMINAL_VP_SHARE,
    max_candidates: int,
) -> tuple[list[OfflineTransition], int, int]:
    """``(transitions, unmatched, skipped)`` for one logged game.

    ``unmatched`` counts ledger commands that no legal candidate matched (a
    soundness signal: the corpus containment sweep predicts 0).
    ``skipped`` counts decisions with fewer than 2 candidates -- no choice
    was made, so there is nothing to learn, exactly as ``extract.py`` does.
    """
    raw: list[_RawDecision] = []
    unmatched = 0
    skipped = 0

    def hook(state: GameState, faction: str, command: ParsedCommand) -> None:
        nonlocal unmatched, skipped
        offer, originals = canonical_offer(
            tuple(legal_moves_for(state, faction))
        )
        if len(offer) < 2:
            skipped += 1
            return
        # Match against the engine's own spelling first (that is what the
        # ledger rows were written in); fall back to the normalized copy for
        # an unordered hex pair the ledger happens to spell the other way.
        chosen = match_index(originals, command)
        if chosen is None:
            chosen = match_index(offer, command)
        if chosen is None:
            unmatched += 1
            return
        raw.append(_encode_decision(state, faction, offer, chosen, max_candidates))

    result = replay_game(game_id, moves_df, deltas_df, on_decision=hook)
    if result.error is not None:
        raise RuntimeError(f"{game_id} failed to replay during export: {result.error}")

    return (
        _assemble(raw, game_id, final_vps, reward_mode),
        unmatched,
        skipped,
    )


def _assemble(
    raw: Sequence[_RawDecision],
    game_id: str,
    final_vps: dict[str, int],
    reward_mode: RewardMode,
) -> list[OfflineTransition]:
    """Split one game's decision stream into per-seat trajectories."""
    if not raw:
        return []
    seats = tuple(final_vps)
    seat_index = {f: i for i, f in enumerate(seats)}
    per_seat: dict[str, list[int]] = {f: [] for f in seats}
    for position, decision in enumerate(raw):
        per_seat.setdefault(decision.faction, []).append(position)

    terminal = terminal_rewards(reward_mode, final_vps)
    observations = [
        Observation(
            hex_planes=d.hex_planes,
            globals=d.globals,
            candidates=d.candidates,
            action_mask=d.mask,
        )
        for d in raw
    ]

    out: list[OfflineTransition] = []
    for faction, positions in per_seat.items():
        for order, position in enumerate(positions):
            is_last = order == len(positions) - 1
            nxt = None if is_last else observations[positions[order + 1]]
            after = final_vps if is_last else raw[positions[order + 1]].vp_before
            reward = dense_step_rewards(
                reward_mode, raw[position].vp_before, after, seats
            )[faction]
            if is_last:
                reward += terminal[faction]
            out.append(
                OfflineTransition(
                    obs=observations[position],
                    action=raw[position].action,
                    reward=float(reward),
                    next_obs=nxt,
                    done=is_last,
                    faction=faction,
                    seat=seat_index.get(faction, -1),
                    game_id=game_id,
                    position=position,
                    next_position=(
                        -1 if is_last else positions[order + 1]
                    ),
                )
            )
    return out


# --------------------------------------------------------------------------
# shard writing
# --------------------------------------------------------------------------


def _pack(
    transitions: Sequence[OfflineTransition], games: Sequence[str]
) -> dict[str, np.ndarray]:
    """Flatten a shard. ``next_index`` points inside this same shard; -1 is
    terminal. Per-seat trajectories never cross a shard boundary because
    shards are cut on whole games.
    """
    game_index = {g: i for i, g in enumerate(games)}
    row_of = {(t.game_id, t.position): i for i, t in enumerate(transitions)}
    cand_counts = np.array([t.obs.n_legal for t in transitions], dtype=np.int32)
    next_index = np.array(
        [
            -1
            if t.next_position < 0
            else row_of.get((t.game_id, t.next_position), -1)
            for t in transitions
        ],
        dtype=np.int32,
    )
    return {
        "hex_planes": np.stack([t.obs.hex_planes for t in transitions]),
        "globals": np.stack([t.obs.globals for t in transitions]),
        "cand_flat": np.concatenate(
            [t.obs.candidates[: t.obs.n_legal] for t in transitions]
        ),
        "cand_counts": cand_counts,
        "action": np.array([t.action for t in transitions], dtype=np.int32),
        "reward": np.array([t.reward for t in transitions], dtype=np.float32),
        "done": np.array([t.done for t in transitions], dtype=bool),
        "next_index": next_index,
        "seat": np.array([t.seat for t in transitions], dtype=np.int8),
        "faction": np.array(
            [FACTION_INDEX[t.faction] for t in transitions], dtype=np.int8
        ),
        "game": np.array(
            [game_index[t.game_id] for t in transitions], dtype=np.int32
        ),
    }


def load_shard(path: Path) -> dict[str, np.ndarray]:
    """Read one shard's arrays. ``cand_flat`` is ragged; use
    :func:`iter_transitions` for assembled records.
    """
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def iter_transitions(
    path: Path, max_candidates: int
) -> Iterator[dict[str, np.ndarray | int | float | bool]]:
    """Yield materialized ``(obs, action, mask, reward, next_obs, done)``
    dicts from a shard, re-padding candidates to ``max_candidates``.
    """
    shard = load_shard(path)
    offsets = np.concatenate([[0], np.cumsum(shard["cand_counts"])]).astype(np.int64)

    def pad(row: int) -> tuple[np.ndarray, np.ndarray]:
        start, end = int(offsets[row]), int(offsets[row + 1])
        count = end - start
        if count > max_candidates:
            raise ValueError(
                f"row {row} has {count} candidates, more than max_candidates="
                f"{max_candidates}"
            )
        candidates = np.zeros((max_candidates, MOVE_FIELDS), dtype=np.int16)
        candidates[:count] = shard["cand_flat"][start:end]
        mask = np.zeros(max_candidates, dtype=np.int8)
        mask[:count] = 1
        return candidates, mask

    for row in range(len(shard["action"])):
        candidates, mask = pad(row)
        nxt = int(shard["next_index"][row])
        next_obs = None
        if nxt >= 0:
            n_candidates, n_mask = pad(nxt)
            next_obs = {
                "hex_planes": shard["hex_planes"][nxt],
                "globals": shard["globals"][nxt],
                "candidates": n_candidates,
                "action_mask": n_mask,
            }
        yield {
            "obs": {
                "hex_planes": shard["hex_planes"][row],
                "globals": shard["globals"][row],
                "candidates": candidates,
                "action_mask": mask,
            },
            "action": int(shard["action"][row]),
            "mask": mask,
            "reward": float(shard["reward"][row]),
            "next_obs": next_obs,
            "done": bool(shard["done"][row]),
        }


def _final_vps(game_id: str, meta_df: pl.DataFrame) -> dict[str, int]:
    row = meta_df.filter(pl.col("game_id") == game_id)
    if row.height == 0:
        raise KeyError(f"{game_id} has no row in games_meta")
    return {k: int(v) for k, v in json.loads(row["final_vp"][0]).items()}


def build(
    out: Path,
    *,
    limit: int | None = None,
    game_ids: Sequence[str] | None = None,
    reward_mode: RewardMode = RewardMode.TERMINAL_VP_SHARE,
    max_candidates: int,
    games_per_shard: int = GAMES_PER_SHARD,
    moves_path: Path = DEFAULT_MOVES,
    deltas_path: Path = DEFAULT_DELTAS,
    meta_path: Path = DEFAULT_META,
    verbose: bool = True,
) -> ExportStats:
    """Export offline transitions to ``out`` as npz shards + a manifest."""
    from bgai.arena.setups import clean_game_ids

    ids = list(game_ids) if game_ids is not None else list(clean_game_ids())
    if limit is not None:
        ids = ids[:limit]
    if not ids:
        raise ValueError("no games to export")

    out.mkdir(parents=True, exist_ok=True)
    moves_df = pl.read_parquet(moves_path).filter(pl.col("game_id").is_in(ids))
    deltas_df = pl.read_parquet(deltas_path).filter(pl.col("game_id").is_in(ids))
    meta_df = pl.read_parquet(meta_path).filter(pl.col("game_id").is_in(ids))

    shards: list[dict[str, object]] = []
    failed: list[str] = []
    total = unmatched_total = skipped_total = 0
    exported_games = 0

    for shard_index, start in enumerate(range(0, len(ids), games_per_shard)):
        batch = ids[start : start + games_per_shard]
        transitions: list[OfflineTransition] = []
        kept: list[str] = []
        for game_id in batch:
            try:
                records, unmatched, skipped = export_game(
                    game_id,
                    moves_df,
                    deltas_df,
                    _final_vps(game_id, meta_df),
                    reward_mode=reward_mode,
                    max_candidates=max_candidates,
                )
            except (RuntimeError, KeyError, ValueError) as exc:
                failed.append(f"{game_id}: {type(exc).__name__}: {exc}")
                continue
            unmatched_total += unmatched
            skipped_total += skipped
            if records:
                transitions.extend(records)
                kept.append(game_id)
        if not transitions:
            continue
        name = f"shard_{shard_index:04d}.npz"
        np.savez_compressed(out / name, **_pack(transitions, kept))
        shards.append(
            {"file": name, "records": len(transitions), "games": len(kept)}
        )
        total += len(transitions)
        exported_games += len(kept)
        if verbose:
            print(f"{name}: {len(transitions)} transitions from {len(kept)} games")

    manifest = {
        "obs_version": OBS_VERSION,
        "mask_source": MASK_SOURCE,
        "mask_note": (
            "full legal set for the acting faction; the online env's mask is "
            "driver.decision's turn-protocol-filtered subset -- see "
            "bgai/env/offline.py's module docstring"
        ),
        "reward_mode": reward_mode.value,
        "max_candidates": max_candidates,
        "max_seats": MAX_SEATS,
        "games_per_shard": games_per_shard,
        "record_count": total,
        "games": exported_games,
        "unmatched_commands": unmatched_total,
        "skipped_single_candidate": skipped_total,
        "failed_games": failed,
        "shards": shards,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return ExportStats(
        shard_count=len(shards),
        record_count=total,
        games=exported_games,
        failed_games=tuple(failed),
        unmatched=unmatched_total,
        skipped_single_candidate=skipped_total,
    )


def main(argv: list[str] | None = None) -> None:
    from bgai.env.config import MAX_CANDIDATES

    parser = argparse.ArgumentParser(
        description="Export logged human games as offline-RL transitions."
    )
    parser.add_argument("--out", type=Path, default=Path("data/datasets/offline_rl"))
    parser.add_argument("--limit", type=int, default=None, help="first N clean games")
    parser.add_argument("--game-id", action="append", default=None)
    parser.add_argument(
        "--reward",
        choices=[m.value for m in RewardMode],
        default=RewardMode.TERMINAL_VP_SHARE.value,
    )
    parser.add_argument("--games-per-shard", type=int, default=GAMES_PER_SHARD)
    parser.add_argument("--max-candidates", type=int, default=MAX_CANDIDATES)
    args = parser.parse_args(argv)

    stats = build(
        args.out,
        limit=args.limit,
        game_ids=args.game_id,
        reward_mode=RewardMode(args.reward),
        max_candidates=args.max_candidates,
        games_per_shard=args.games_per_shard,
    )
    print(
        f"{stats.record_count} transitions / {stats.games} games / "
        f"{stats.shard_count} shards -> {args.out}"
    )
    print(
        f"unmatched={stats.unmatched} "
        f"skipped_single_candidate={stats.skipped_single_candidate} "
        f"failed={len(stats.failed_games)}"
    )


if __name__ == "__main__":  # pragma: no cover -- CLI
    main()
