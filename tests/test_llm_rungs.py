"""L3 retrieval and L4 scaffolds (Phase 7), mock-tested end to end."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from bgai.arena.driver import decision, new_game
from bgai.engine.tm.setup import load_setup
from bgai.llm.provider import MockProvider
from bgai.llm.retrieval import PositionIndex, build_index, format_neighbours
from bgai.llm.scaffolds import BestOfNCritic, GamePlan
from bgai.training.dataset_build import build
from bgai.training.encode_state import GLOBAL_DIM
from bgai.training.vocab import FACTION_INDEX


def _position():
    sim = new_game(load_setup("4pLeague_S10_D1L1_G1"))
    faction, offer = decision(sim)
    return sim, faction, offer


def test_index_returns_same_faction_precedents(tmp_path: Path) -> None:
    build(tmp_path, game_ids=["4pLeague_S10_D1L1_G1", "4pLeague_S1_D1L1_G1"])
    index = build_index(tmp_path, max_records=2000)
    assert len(index) > 100

    sim, faction, _ = _position()
    hits = index.query(sim.game, faction, k=5)
    assert len(hits) == 5
    assert all(h.faction == faction for h in hits)
    assert all(h.distance >= 0 for h in hits)
    # distances come back sorted
    assert hits == sorted(hits, key=lambda h: h.distance)


def test_retrieval_block_is_framed_as_precedent_not_instruction() -> None:
    matrix = np.random.default_rng(0).normal(size=(20, GLOBAL_DIM)).astype(np.float32)
    index = PositionIndex(
        matrix,
        np.full(20, FACTION_INDEX["nomads"]),
        ["build"] * 20,
        np.full(20, 0.3, dtype=np.float32),
    )
    sim, _, _ = _position()
    text = format_neighbours(index.query(sim.game, "nomads", k=3))
    assert "strong human players chose" in text
    assert "not instructions" in text
    assert format_neighbours([]) == "No similar historical positions found."


def test_best_of_n_runs_propose_critique_decide() -> None:
    _, _, offer = _position()
    calls: list[str] = []

    def responder(messages):
        last = messages[-1].content
        calls.append(last)
        if "nominate up to" in last:
            return "0: take the corner\n2: contest the middle"
        if "Argue against it" in last:
            return "it is slow"
        return "2"

    critic = BestOfNCritic(provider=MockProvider(responder=responder), n_candidates=4)
    index, critiques = critic.choose_index("STATE", "MOVES", offer, "SYSTEM")
    assert index == 2
    assert len(critiques) == 2, "one critique per proposed candidate"
    assert sum("nominate up to" in c for c in calls) == 1
    assert sum("Argue against it" in c for c in calls) == 2


def test_best_of_n_falls_back_within_the_offer() -> None:
    _, _, offer = _position()
    critic = BestOfNCritic(provider=MockProvider(replies=["no numbers here"]))
    index, critiques = critic.choose_index("STATE", "MOVES", offer, "SYSTEM")
    assert 0 <= index < len(offer)
    assert critiques == []


def test_game_plan_updates_on_schedule() -> None:
    provider = MockProvider(replies=["plan: rush temples"])
    critic = BestOfNCritic(provider=provider, plan=GamePlan(update_every=5))
    critic.maybe_update_plan("STATE", "SYSTEM", decision_index=0)
    assert "rush temples" in critic.plan.text
    before = provider.usage.calls
    critic.maybe_update_plan("STATE", "SYSTEM", decision_index=2)  # too soon
    assert provider.usage.calls == before
    critic.maybe_update_plan("STATE", "SYSTEM", decision_index=9)
    assert provider.usage.calls == before + 1


def test_plan_block_is_empty_before_first_update() -> None:
    assert GamePlan().block() == ""
    assert "game plan" in GamePlan(text="x").block()
