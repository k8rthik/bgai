"""``Episode``: the immutable engine<->env glue both wrappers sit on."""

from __future__ import annotations

import random


import pytest

from bgai.engine.tm.apply import EngineError
from bgai.env.config import (
    ILLEGAL_ACTION_REWARD,
    MAX_PLAYERS,
    MIN_PLAYERS,
    EnvConfig,
    IllegalActionError,
    IllegalActionPolicy,
    RewardMode,
)
from bgai.env.core import Episode, vp_table

COUNTS = list(range(MIN_PLAYERS, MAX_PLAYERS + 1))


def _play_out(episode: Episode, seed: int = 0) -> tuple[Episode, int]:
    rng = random.Random(seed)
    steps = 0
    while not episode.finished:
        _, offer = episode.pending()
        episode, _ = episode.step(rng.randrange(len(offer)))
        steps += 1
    return episode, steps


@pytest.mark.parametrize("count", COUNTS)
def test_random_play_completes_every_player_count(count: int) -> None:
    """2/3/5 players are reference-rules-only (never replay-validated), so
    this is also the only evidence they run at all.
    """
    for seed in range(4):
        episode, steps = _play_out(Episode.start(seed, EnvConfig(player_count=count)), seed)
        assert episode.sim.finished, f"{count}p seed {seed}: {episode.engine_error}"
        assert episode.engine_error is None
        assert steps == episode.decisions
        assert len(episode.vps()) == count
        assert all(vp > 0 for vp in episode.vps().values())


def test_step_returns_a_new_episode_and_never_mutates() -> None:
    """Immutability is the property MCTS branching depends on (D6.1)."""
    episode = Episode.start(1, EnvConfig(player_count=4))
    before_sim = episode.sim
    before_decisions = episode.decisions
    nxt, _ = episode.step(0)
    assert nxt is not episode
    assert episode.sim is before_sim
    assert episode.decisions == before_decisions
    assert nxt.decisions == before_decisions + 1


def test_branching_from_one_position_gives_independent_futures() -> None:
    episode = Episode.start(2, EnvConfig(player_count=4))
    faction, offer = episode.pending()
    assert len(offer) >= 2
    left, _ = episode.step(0)
    right, _ = episode.step(1)
    assert left.sim != right.sim
    assert episode.pending() == (faction, offer)  # the parent is untouched


def test_seats_are_setup_order() -> None:
    episode = Episode.start(3, EnvConfig(player_count=5))
    assert episode.seats == episode.setup.factions
    assert list(vp_table(episode.game)) == list(episode.seats)


def test_offer_for_is_empty_for_a_waiting_seat() -> None:
    episode = Episode.start(4, EnvConfig(player_count=4))
    acting = episode.acting_seat()
    for faction in episode.seats:
        offer = episode.offer_for(faction)
        assert bool(offer) == (faction == acting)


def test_resolve_accepts_every_masked_index() -> None:
    episode = Episode.start(5, EnvConfig(player_count=4))
    _, offer = episode.pending()
    for index, move in enumerate(offer):
        # equality, not identity: ``pending()`` regenerates the offer each
        # call (the engine's legal-move generation is pure), so what matters
        # is that index i names the same *move*.
        assert episode.resolve(index) == move


@pytest.mark.parametrize("bad", [-1, 10_000])
def test_resolve_rejects_an_unmasked_index(bad: int) -> None:
    episode = Episode.start(6, EnvConfig(player_count=4))
    with pytest.raises(IllegalActionError, match="legal indices are"):
        episode.resolve(bad)


def test_resolve_rejects_a_non_integer_action() -> None:
    episode = Episode.start(6, EnvConfig(player_count=4))
    with pytest.raises(IllegalActionError, match="must be an integer index"):
        episode.resolve("build A1")  # type: ignore[arg-type]


def test_illegal_action_terminates_with_a_penalty_by_default() -> None:
    episode = Episode.start(7, EnvConfig(player_count=4))
    faction = episode.acting_seat()
    nxt, transition = episode.step(999)
    assert transition.terminated and not transition.truncated
    assert transition.illegal_action == 999
    assert transition.rewards[faction] == ILLEGAL_ACTION_REWARD
    assert all(
        transition.rewards[seat] == 0.0 for seat in episode.seats if seat != faction
    )
    assert nxt.finished
    assert nxt.illegal_action == 999
    with pytest.raises(IllegalActionError, match="finished episode"):
        nxt.step(0)


def test_illegal_action_raises_under_the_strict_policy() -> None:
    config = EnvConfig(player_count=4, illegal_action=IllegalActionPolicy.RAISE)
    episode = Episode.start(7, config)
    with pytest.raises(IllegalActionError, match="action_mask has exactly those bits"):
        episode.step(999)


def test_the_mask_is_never_repaired_into_a_different_legal_move() -> None:
    """An out-of-range index must not become ``offer[index % n]`` -- that
    would make every index playable and delete the mask's meaning.
    """
    episode = Episode.start(8, EnvConfig(player_count=4))
    _, offer = episode.pending()
    nxt, transition = episode.step(len(offer))  # one past the end
    assert transition.illegal_action == len(offer)
    assert nxt.sim.decisions == episode.sim.decisions, "no move was applied"


def test_resolve_on_a_finished_episode_explains_why() -> None:
    episode, _ = _play_out(Episode.start(9, EnvConfig(player_count=2)), 9)
    with pytest.raises(IllegalActionError, match="episode is over"):
        episode.resolve(0)


def test_truncation_fires_at_the_decision_budget() -> None:
    config = EnvConfig(player_count=4, max_decisions=5)
    episode = Episode.start(10, config)
    rng = random.Random(0)
    while True:
        _, offer = episode.pending()
        episode, transition = episode.step(rng.randrange(len(offer)))
        if transition.done:
            break
    assert transition.truncated and not transition.terminated
    assert episode.decisions == 5
    assert episode.finished
    assert episode.pending() is None


def test_engine_rejection_is_recorded_not_raised_by_default() -> None:
    """A move the engine offered and then refused is a legal_moves soundness
    finding; the repo's rule is visible-not-fatal (test_legal_soundness.py).
    """
    episode = Episode.start(11, EnvConfig(player_count=4))
    faction, offer = episode.pending()

    def boom(sim, choice):
        raise EngineError("synthetic rejection", state=sim.game, faction=faction, cmd=choice)

    import bgai.env.core as core

    original = core.advance
    core.advance = boom
    try:
        nxt, transition = episode.step(0)
    finally:
        core.advance = original

    assert transition.truncated and not transition.terminated
    assert transition.engine_error is not None
    assert "synthetic rejection" in transition.engine_error
    assert nxt.finished and nxt.engine_error == transition.engine_error
    assert all(value == 0.0 for value in transition.rewards.values())


def test_engine_rejection_can_be_made_fatal() -> None:
    config = EnvConfig(player_count=4, raise_on_engine_error=True)
    episode = Episode.start(11, config)
    faction, _ = episode.pending()

    def boom(sim, choice):
        raise EngineError("synthetic", state=sim.game, faction=faction, cmd=choice)

    import bgai.env.core as core

    original = core.advance
    core.advance = boom
    try:
        with pytest.raises(EngineError, match="synthetic"):
            episode.step(0)
    finally:
        core.advance = original


def test_start_rejects_a_setup_whose_seat_count_disagrees(monkeypatch) -> None:
    """A defensive guard: nothing in-tree can trip it, but a custom
    ``setup_source`` could, and a silent seat-count mismatch would show up
    much later as a wrong-width observation.
    """
    import bgai.env.core as core
    from bgai.env.table import synthetic_setup

    four = synthetic_setup(12, EnvConfig(player_count=4))
    monkeypatch.setattr(core, "build_setup", lambda seed, config: four)
    with pytest.raises(ValueError, match="setup has 4 seats"):
        Episode.start(12, EnvConfig(player_count=5))


@pytest.mark.parametrize("mode", list(RewardMode))
def test_every_reward_mode_plays_a_full_game(mode: RewardMode) -> None:
    config = EnvConfig(player_count=4, reward_mode=mode)
    episode, _ = _play_out(Episode.start(13, config), 13)
    assert episode.sim.finished
