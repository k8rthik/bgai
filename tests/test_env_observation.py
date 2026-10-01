"""Observation layout and -- the load-bearing one -- that the legal mask
is the engine's own legal move generation and nothing else.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from bgai.arena.driver import decision
from bgai.engine.tm.legal_shared import cmd
from bgai.env.canonical import normalize_move
from bgai.env.config import (
    MAX_PLAYERS,
    MAX_SEATS,
    MIN_PLAYERS,
    EnvConfig,
    OfferOverflowError,
)
from bgai.env.core import Episode
from bgai.env.observation import (
    GLOBAL_DIM,
    HEX_FEAT_DIM,
    HEXES,
    SEAT_BLOCK,
    encode_candidates,
    encode_observation,
    seat_order,
)
from bgai.training.encode_move import MOVE_FIELDS, encode_move

COUNTS = list(range(MIN_PLAYERS, MAX_PLAYERS + 1))


def _episode(count: int, seed: int = 101) -> Episode:
    return Episode.start(seed, EnvConfig(player_count=count))


def _walk(count: int, steps: int, seed: int = 101):
    """Yield every (episode, faction, offer) along a random trajectory."""
    episode = _episode(count, seed)
    rng = random.Random(seed)
    for _ in range(steps):
        if episode.finished:
            return
        faction, offer = episode.pending()
        yield episode, faction, offer
        episode, _ = episode.step(rng.randrange(len(offer)))


# --------------------------------------------------------------------------
# layout
# --------------------------------------------------------------------------


def test_layout_constants_are_self_consistent() -> None:
    assert HEX_FEAT_DIM == 7 + 1 + 5 + MAX_SEATS
    assert GLOBAL_DIM > MAX_SEATS * SEAT_BLOCK
    assert len(HEXES) == 113


@pytest.mark.parametrize("count", COUNTS)
def test_shapes_and_dtypes_are_fixed_across_player_counts(count: int) -> None:
    """One layout serves 2-5 players -- that is the whole point of
    MAX_SEATS seat blocks plus a presence mask.
    """
    config = EnvConfig(player_count=count)
    episode = Episode.start(5, config)
    faction = episode.acting_seat()
    obs = episode.observe(faction)
    assert obs.hex_planes.shape == (len(HEXES), HEX_FEAT_DIM)
    assert obs.hex_planes.dtype == np.int8
    assert obs.globals.shape == (GLOBAL_DIM,)
    assert obs.globals.dtype == np.int16
    assert obs.candidates.shape == (config.max_candidates, MOVE_FIELDS)
    assert obs.candidates.dtype == np.int16
    assert obs.action_mask.shape == (config.max_candidates,)
    assert obs.action_mask.dtype == np.int8


@pytest.mark.parametrize("count", COUNTS)
def test_absent_seat_blocks_are_zero(count: int) -> None:
    episode = _episode(count)
    obs = episode.observe(episode.acting_seat())
    tail = obs.globals[count * SEAT_BLOCK : MAX_SEATS * SEAT_BLOCK]
    assert not tail.any(), "an absent seat's block must be all zeros"


@pytest.mark.parametrize("count", COUNTS)
def test_seat_count_and_presence_are_encoded(count: int) -> None:
    episode = _episode(count)
    obs = episode.observe(episode.acting_seat())
    context = obs.globals[MAX_SEATS * SEAT_BLOCK :]
    seat_count_block = context[: MAX_PLAYERS - MIN_PLAYERS + 1]
    assert seat_count_block.sum() == 1
    assert seat_count_block[count - MIN_PLAYERS] == 1
    presence = context[
        MAX_PLAYERS - MIN_PLAYERS + 1 : MAX_PLAYERS - MIN_PLAYERS + 1 + MAX_SEATS
    ]
    assert presence.sum() == count
    assert presence[:count].all()


def test_observation_is_mover_relative() -> None:
    """Two seats at the same position see different vectors, and each sees
    *itself* in seat block 0 -- which is what makes a policy
    seat-equivariant.
    """
    episode = _episode(4)
    seats = episode.seats
    first = episode.observe(seats[0])
    second = episode.observe(seats[1])
    assert not np.array_equal(first.globals, second.globals)
    assert seat_order(episode.game, seats[1])[0] == seats[1]
    # the faction one-hot at the head of block 0 identifies the observer
    from bgai.training.vocab import FACTION_INDEX

    assert first.globals[FACTION_INDEX[seats[0]]] == 1
    assert second.globals[FACTION_INDEX[seats[1]]] == 1


def test_hex_owner_plane_is_a_mover_relative_one_hot() -> None:
    episode = _episode(4, seed=9)
    rng = random.Random(3)
    # play until someone owns a hex (setup dwellings place immediately)
    for _ in range(12):
        faction, offer = episode.pending()
        episode, _ = episode.step(rng.randrange(len(offer)))
    owners = {
        key: hx.owner for key, hx in episode.game.hexes.items() if hx.owner is not None
    }
    assert owners, "setup should have placed dwellings by now"
    seats = episode.seats
    owner_base = 7 + 1 + 5
    for observer in seats:
        obs = episode.observe(observer)
        order = seat_order(episode.game, observer)
        for key, owner in owners.items():
            row = HEXES.index(key)
            plane = obs.hex_planes[row, owner_base : owner_base + MAX_SEATS]
            assert plane.sum() == 1
            assert plane[order.index(owner)] == 1


# --------------------------------------------------------------------------
# the mask
# --------------------------------------------------------------------------


@pytest.mark.parametrize("count", COUNTS)
def test_mask_is_exactly_the_engine_offer(count: int) -> None:
    """The mask is not a second copy of the rules: bit i is set iff index i
    is in ``driver.decision``'s own canonical offer, for every decision
    along a whole trajectory.
    """
    for episode, faction, offer in _walk(count, 10_000):
        obs = episode.observe(faction)
        assert obs.n_legal == len(offer)
        assert obs.action_mask[: len(offer)].all()
        assert not obs.action_mask[len(offer) :].any()
        # and the episode's offer is a re-presentation of the driver's own --
        # same faction, same moves, one for one. The ORDER may differ (see
        # bgai.env.canonical: a bridge's hex pair is normalized), so this
        # compares the sets, and test_env_canonical.py pins the ordering.
        driver_faction, driver_offer = decision(episode.sim)
        assert driver_faction == faction
        assert len(driver_offer) == len(offer)
        assert {normalize_move(m) for m in driver_offer} == set(offer)


@pytest.mark.parametrize("count", COUNTS)
def test_candidate_rows_match_encode_move_in_offer_order(count: int) -> None:
    for episode, faction, offer in _walk(count, 60):
        obs = episode.observe(faction)
        for index, move in enumerate(offer):
            expected = encode_move(move, episode.game, faction)
            assert np.array_equal(obs.candidates[index], expected)
        assert not obs.candidates[len(offer) :].any(), "padding must stay zero"


def test_non_acting_seat_gets_an_all_zero_mask() -> None:
    """PettingZoo's "not your move" convention (what connect_four_v3 does)."""
    episode = _episode(4)
    acting = episode.acting_seat()
    for faction in episode.seats:
        obs = episode.observe(faction)
        if faction == acting:
            assert obs.n_legal > 0
        else:
            assert obs.n_legal == 0


def test_finished_episode_masks_everything_off() -> None:
    episode = _episode(2, seed=3)
    rng = random.Random(3)
    while not episode.finished:
        _, offer = episode.pending()
        episode, _ = episode.step(rng.randrange(len(offer)))
    for faction in episode.seats:
        assert episode.observe(faction).n_legal == 0


def test_observe_rejects_a_faction_not_at_the_table() -> None:
    episode = _episode(2)
    with pytest.raises(KeyError, match="not a seat at this table"):
        episode.observe("not_a_faction")


def test_offer_larger_than_the_action_space_raises_rather_than_truncating() -> None:
    """A truncated mask would delete legal moves -- a rules bug dressed up
    as a capacity limit.
    """
    episode = _episode(4)
    faction, offer = episode.pending()
    fake = tuple([cmd("done")] * 5)
    with pytest.raises(OfferOverflowError, match="action space holds 3"):
        encode_candidates(episode.game, faction, fake, 3)


def test_encode_observation_pads_to_the_configured_width() -> None:
    episode = _episode(4)
    faction, offer = episode.pending()
    obs = encode_observation(episode.game, faction, offer, 400)
    assert obs.candidates.shape == (400, MOVE_FIELDS)
    assert obs.n_legal == len(offer)


def test_observation_equality_compares_arrays_and_dtypes() -> None:
    episode = _episode(4)
    faction = episode.acting_seat()
    first = episode.observe(faction)
    second = episode.observe(faction)
    assert first == second
    assert hash(first) == hash(second)
    assert first != episode.observe(episode.seats[1])
    assert first.__eq__(object()) is NotImplemented
