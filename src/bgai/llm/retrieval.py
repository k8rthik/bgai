"""L3 retrieval: what did strong humans play here? (Phase 7)

Rung L3 of the ladder gives the LLM case-based evidence instead of
asking it to recall Terra Mystica theory. The index is built from the
same extraction the imitation net trains on, so "similar position" means
similar under the *encoder's* notion of similarity -- the one we already
validated by training a 55.7%-accurate policy on it.

**Design choice: hand-rolled nearest neighbour over the global feature
vector, not an embedding model.** The imitation net already produces a
256-d state embedding, and using it would couple the "pure LLM" class
(Class P in the master plan's fairness rules) to a trained engine-AI
net -- which would silently make the LLM a hybrid. Retrieval over raw
encoded features keeps L3 inside Class P: it uses the *corpus*, not our
model. That is a fairness decision, not a technical one.

Distance is L2 over the standardized global scalars only (resources,
buildings, cults, round). Hex planes are deliberately excluded: two
positions with the same economic shape but mirrored geography are
strategically similar, and including 1,921 sparse board features would
let geography dominate the metric.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bgai.engine.tm.state import GameState
from bgai.training.encode_state import encode_state
from bgai.training.vocab import FACTION_NAMES


@dataclass(frozen=True)
class Neighbour:
    """A retrieved historical decision."""

    faction: str
    move_summary: str
    final_share: float
    distance: float


class PositionIndex:
    """Nearest-neighbour index over extracted corpus decisions."""

    def __init__(
        self,
        globals_matrix: np.ndarray,
        factions: np.ndarray,
        chosen_summaries: list[str],
        final_shares: np.ndarray,
    ) -> None:
        if not (
            len(globals_matrix) == len(factions) == len(chosen_summaries) == len(final_shares)
        ):
            raise ValueError("index components must be the same length")
        self.factions = factions
        self.summaries = chosen_summaries
        self.final_shares = final_shares
        self.mean = globals_matrix.mean(axis=0)
        self.scale = globals_matrix.std(axis=0)
        self.scale[self.scale == 0] = 1.0
        self.matrix = (globals_matrix - self.mean) / self.scale

    def __len__(self) -> int:
        return len(self.summaries)

    def query(
        self, state: GameState, faction: str, k: int = 5, same_faction: bool = True
    ) -> list[Neighbour]:
        """The k most similar historical decisions. ``same_faction``
        restricts to the same faction -- usually right, because TM
        factions play differently enough that a Darklings precedent is
        poor advice for Swarmlings.
        """
        vector = (encode_state(state, faction).globals - self.mean) / self.scale
        mask = np.ones(len(self.matrix), dtype=bool)
        if same_faction:
            mask = self.factions == FACTION_NAMES.index(faction)
            if not mask.any():
                mask = np.ones(len(self.matrix), dtype=bool)
        candidates = np.flatnonzero(mask)
        distances = np.linalg.norm(self.matrix[candidates] - vector, axis=1)
        order = np.argsort(distances)[:k]
        return [
            Neighbour(
                faction=FACTION_NAMES[int(self.factions[candidates[i]])],
                move_summary=self.summaries[int(candidates[i])],
                final_share=float(self.final_shares[int(candidates[i])]),
                distance=float(distances[i]),
            )
            for i in order
        ]


def format_neighbours(neighbours: list[Neighbour]) -> str:
    """The retrieval block injected into the L3 prompt."""
    if not neighbours:
        return "No similar historical positions found."
    lines = ["In similar positions, strong human players chose:"]
    for n in neighbours:
        lines.append(
            f"  - {n.faction}: {n.move_summary} "
            f"(that player finished with {n.final_share:.0%} of the table's VP)"
        )
    lines.append(
        "These are precedents from Div 1-3 tournament games, not instructions."
    )
    return "\n".join(lines)


def build_index(shard_dir: Path, max_records: int = 50_000) -> PositionIndex:
    """Build an index from imitation shards. Capped by default: the full
    1.2M-decision corpus is ~1.4 GB of float32 globals, and retrieval
    quality saturates long before that.
    """
    from bgai.training.vocab import VERBS

    globals_rows: list[np.ndarray] = []
    factions: list[int] = []
    summaries: list[str] = []
    shares: list[float] = []
    for path in sorted(shard_dir.glob("shard_*.npz")):
        with np.load(path) as data:
            counts = data["cand_counts"]
            offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
            for row in range(len(counts)):
                if len(summaries) >= max_records:
                    break
                chosen = int(data["chosen"][row])
                cand = data["cand_flat"][offsets[row] + chosen]
                verb_id = int(cand[0]) - 1
                verb = VERBS[verb_id] if 0 <= verb_id < len(VERBS) else "?"
                globals_rows.append(data["globals"][row].astype(np.float32))
                factions.append(int(data["mover_faction"][row]))
                summaries.append(verb)
                total = float(data["final_vps"][row].sum())
                shares.append(float(data["final_vps"][row][0]) / total if total else 0.25)
        if len(summaries) >= max_records:
            break
    return PositionIndex(
        np.stack(globals_rows), np.array(factions), summaries, np.array(shares)
    )
