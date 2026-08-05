"""Decision extraction: replay capture -> DecisionRecords (plan Task 5).

For every ``Kind.DECISION`` command in a game's ledger, capture the
pre-apply state, compute the faction's canonical legal candidates, and
locate the replayed command among them **by encoded features** (raw-text
and color-alias differences make ``ParsedCommand`` equality unusable;
``encode_move`` normalizes both). Records with fewer than 2 candidates
carry no choice and are skipped silently; a replayed command that fails
to match any candidate is counted in ``skipped`` (a soundness signal --
the corpus containment sweep predicts 0).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

import numpy as np
import polars as pl

from bgai.data.ledger_parser import ParsedCommand
from bgai.engine.tm.legal import legal_moves_for
from bgai.engine.tm.legal_match import match_index
from bgai.engine.tm.replay import replay_game
from bgai.engine.tm.state import GameState
from bgai.arena.sim import canonical_moves
from bgai.training.encode_move import encode_move
from bgai.training.encode_state import encode_state
from bgai.training.vocab import FACTION_INDEX

_GAME_ID_RE = re.compile(r"4pLeague_S(\d+)_D(\d)L\d+_G\d+$")


@dataclass(frozen=True)
class DecisionRecord:
    hex_planes: np.ndarray  # (113, HEX_FEAT_DIM) int8
    globals: np.ndarray  # (GLOBAL_DIM,) int16
    candidates: np.ndarray  # (n_cand, MOVE_FIELDS) int16, canonical order
    chosen: int
    final_vps: np.ndarray  # (4,) int16, mover-relative seat order
    season: int
    division: int
    mover_faction_id: int


def _final_vps_by_faction(game_id: str, meta_df: pl.DataFrame) -> dict[str, int]:
    row = meta_df.filter(pl.col("game_id") == game_id)
    return {k: int(v) for k, v in json.loads(row["final_vp"][0]).items()}


def extract_game(
    game_id: str,
    moves_df: pl.DataFrame,
    deltas_df: pl.DataFrame,
    meta_df: pl.DataFrame | None = None,
) -> tuple[list[DecisionRecord], int]:
    """Returns (records, n_unmatched). Raises if the replay itself fails
    (clean games are expected to replay -- a failure here is a bug, not
    data noise).
    """
    if meta_df is None:
        meta_df = pl.read_parquet("data/datasets/games_meta.parquet")
    match = _GAME_ID_RE.match(game_id)
    if match is None:
        raise ValueError(f"unparseable game id {game_id!r}")
    season, division = int(match.group(1)), int(match.group(2))
    vps = _final_vps_by_faction(game_id, meta_df)

    records: list[DecisionRecord] = []
    unmatched = 0

    def hook(state: GameState, faction: str, cmd: ParsedCommand) -> None:
        nonlocal unmatched
        candidates = canonical_moves(legal_moves_for(state, faction))
        if len(candidates) < 2:
            return
        chosen = match_index(candidates, cmd)
        if chosen is None:
            unmatched += 1
            return
        cand_arr = np.stack([encode_move(c, state, faction) for c in candidates])
        seats = state.setup.factions
        pivot = seats.index(faction)
        order = seats[pivot:] + seats[:pivot]
        enc = encode_state(state, faction)
        records.append(
            DecisionRecord(
                hex_planes=enc.hex_planes,
                globals=enc.globals,
                candidates=cand_arr,
                chosen=chosen,
                final_vps=np.array([vps[f] for f in order], dtype=np.int16),
                season=season,
                division=division,
                mover_faction_id=FACTION_INDEX[faction],
            )
        )

    result = replay_game(game_id, moves_df, deltas_df, on_decision=hook)
    if result.error is not None:
        raise RuntimeError(f"{game_id} failed to replay during extraction: {result.error}")
    return records, unmatched
