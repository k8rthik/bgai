"""One-ply greedy heuristic baseline (plan Task 6).

Deliberately crude: its only job is to be unambiguously stronger than
random so the arena's measuring instrument (mirrored rotation +
TrueSkill) demonstrably detects a known skill difference (master-plan
Phase 4 gate).
"""

from __future__ import annotations

import random

from bgai.data.ledger_parser import ParsedCommand
from bgai.engine.tm.apply import EngineError, apply
from bgai.engine.tm.factions_data import CULTS
from bgai.engine.tm.state import GameState

# Calibration constraints (checked against BASE_EXCHANGE_RATES):
# - resource->VP conversions must be net-negative (3 C -> 1 VP at
#   3*0.35 > 1.0) so greedy develops instead of hoarding VP early;
# - per-building-type credit must exceed construction cost under these
#   same weights (D: 2.0 > 1 W + 2 C = 1.25) so greedy actually builds;
# - workers > coins, priests > workers (their real exchange ordering).
WEIGHTS = {
    "vp": 1.0,
    "coins": 0.35,
    "workers": 0.55,
    "priests": 0.70,
    "bowl2": 0.03,
    "bowl3": 0.08,
    "cult": 0.25,
    "track": 0.4,
}

BUILDING_WEIGHTS = {"D": 2.0, "TP": 4.0, "TE": 5.0, "SH": 7.0, "SA": 7.0}


def evaluate(state: GameState, faction: str) -> float:
    """Static evaluation of ``faction``'s position (single scalar)."""
    fs = state.factions[faction]
    return (
        WEIGHTS["vp"] * fs.vp
        + WEIGHTS["coins"] * fs.coins
        + WEIGHTS["workers"] * fs.workers
        + WEIGHTS["priests"] * fs.priests
        + WEIGHTS["bowl2"] * fs.power.bowl2
        + WEIGHTS["bowl3"] * fs.power.bowl3
        + WEIGHTS["cult"] * sum(state.cults[faction][c] for c in CULTS)
        + sum(BUILDING_WEIGHTS[b] * len(hexes) for b, hexes in fs.buildings.items())
        + WEIGHTS["track"] * (fs.shipping + fs.dig_level)
    )


class GreedyAgent:
    """One-ply lookahead: apply each offered move, keep the argmax under
    :func:`evaluate` (stable -- first max wins, so equal-value moves
    resolve by the arena's canonical offer order). A move ``apply``
    rejects scores ``-inf`` (a soundness finding surfaces through arena
    fuzzing, not through a baseline crash).
    """

    def __init__(self, name: str = "greedy") -> None:
        self.name = name

    def choose(
        self,
        state: GameState,
        faction: str,
        offer: tuple[ParsedCommand, ...],
        rng: random.Random,
    ) -> ParsedCommand:
        best = offer[0]
        best_score = float("-inf")
        for move in offer:
            try:
                child = apply(state, faction, move)
            except EngineError:
                continue
            score = evaluate(child, faction)
            if score > best_score:
                best, best_score = move, score
        return best
