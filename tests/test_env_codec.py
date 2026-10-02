"""State serialization: a paused game round-trips to a dict and to bytes
without losing anything the next offer depends on.
"""

from __future__ import annotations

import random

import orjson
import pytest

from bgai.arena.driver import decision
from bgai.env.codec import (
    CODEC_VERSION,
    CodecError,
    game_from_dict,
    game_to_dict,
    setup_from_dict,
    setup_to_dict,
    sim_from_bytes,
    sim_from_dict,
    sim_to_bytes,
    sim_to_dict,
)
from bgai.env.config import MAX_PLAYERS, MIN_PLAYERS, EnvConfig
from bgai.env.core import Episode
from bgai.env.table import synthetic_setup

COUNTS = list(range(MIN_PLAYERS, MAX_PLAYERS + 1))


def _states(count: int, steps: int, seed: int = 31):
    episode = Episode.start(seed, EnvConfig(player_count=count))
    rng = random.Random(seed)
    for _ in range(steps):
        if episode.finished:
            return
        yield episode
        _, offer = episode.pending()
        episode, _ = episode.step(rng.randrange(len(offer)))
    yield episode


@pytest.mark.parametrize("count", COUNTS)
def test_setup_round_trips(count: int) -> None:
    setup = synthetic_setup(9, EnvConfig(player_count=count))
    assert setup_from_dict(setup_to_dict(setup)) == setup


@pytest.mark.parametrize("count", COUNTS)
def test_game_state_round_trips_along_a_trajectory(count: int) -> None:
    for episode in _states(count, 45):
        assert game_from_dict(game_to_dict(episode.game)) == episode.game


@pytest.mark.parametrize("count", COUNTS)
def test_sim_state_round_trips_through_dict_and_bytes(count: int) -> None:
    for episode in _states(count, 45):
        assert sim_from_dict(sim_to_dict(episode.sim)) == episode.sim
        assert sim_from_bytes(sim_to_bytes(episode.sim)) == episode.sim


@pytest.mark.parametrize("count", COUNTS)
def test_encoding_is_canonical(count: int) -> None:
    """Equal states must give byte-identical payloads -- frozenset iteration
    order is not stable, so sets are written sorted. This is what lets the
    bytes double as a transposition key.
    """
    for episode in _states(count, 30):
        once = sim_to_bytes(episode.sim)
        twice = sim_to_bytes(sim_from_bytes(once))
        assert once == twice


def test_restored_state_offers_the_same_moves() -> None:
    """The real contract: a restored snapshot is the *same game*, which
    means the driver's next offer is identical -- including the turn
    bookkeeping (fresh_taken / free_used) that is not on the board.
    """
    episode = Episode.start(404, EnvConfig(player_count=4))
    rng = random.Random(404)
    for _ in range(80):
        if episode.finished:
            break
        restored = sim_from_bytes(sim_to_bytes(episode.sim))
        assert decision(restored) == decision(episode.sim)
        assert restored.fresh_taken == episode.sim.fresh_taken
        assert restored.free_used == episode.sim.free_used
        assert restored.prev_verb == episode.sim.prev_verb
        assert restored.income_marker == episode.sim.income_marker
        _, offer = episode.pending()
        episode, _ = episode.step(rng.randrange(len(offer)))


def test_restored_state_continues_identically() -> None:
    """Snapshot mid-game, replay the same actions from both copies, and the
    trajectories must coincide move for move.
    """
    episode = Episode.start(808, EnvConfig(player_count=3))
    rng = random.Random(1)
    for _ in range(40):
        _, offer = episode.pending()
        episode, _ = episode.step(rng.randrange(len(offer)))

    from bgai.arena.driver import advance

    actions = [random.Random(2).randrange(3) for _ in range(30)]
    live = episode.sim
    copy = sim_from_bytes(sim_to_bytes(episode.sim))
    for action in actions:
        live_decision, copy_decision = decision(live), decision(copy)
        if live_decision is None:
            assert copy_decision is None
            break
        assert live_decision == copy_decision
        move = live_decision[1][action % len(live_decision[1])]
        live, copy = advance(live, move), advance(copy, move)
    assert live == copy


def test_payload_is_json() -> None:
    episode = Episode.start(5, EnvConfig(player_count=2))
    raw = orjson.loads(sim_to_bytes(episode.sim))
    assert raw["codec_version"] == CODEC_VERSION
    assert set(raw) >= {"game", "fresh_taken", "free_used", "decisions"}


def test_wrong_codec_version_is_refused() -> None:
    episode = Episode.start(5, EnvConfig(player_count=2))
    payload = sim_to_dict(episode.sim)
    payload["codec_version"] = CODEC_VERSION + 1
    with pytest.raises(CodecError, match="codec v"):
        sim_from_dict(payload)


def test_missing_key_names_itself() -> None:
    episode = Episode.start(5, EnvConfig(player_count=2))
    payload = sim_to_dict(episode.sim)
    del payload["fresh_taken"]
    with pytest.raises(CodecError, match="missing required key 'fresh_taken'"):
        sim_from_dict(payload)


def test_unknown_phase_is_refused() -> None:
    episode = Episode.start(5, EnvConfig(player_count=2))
    payload = game_to_dict(episode.game)
    payload["phase"] = "NOT_A_PHASE"
    with pytest.raises(CodecError, match="unknown phase"):
        game_from_dict(payload)


def test_unknown_option_field_is_refused() -> None:
    episode = Episode.start(5, EnvConfig(player_count=2))
    payload = setup_to_dict(episode.setup)
    payload["options"]["warp_drive"] = True
    with pytest.raises(CodecError, match="unknown GameOptions fields"):
        setup_from_dict(payload)


@pytest.mark.parametrize("payload", [b"not json", b"[1, 2, 3]"])
def test_garbage_bytes_are_refused(payload: bytes) -> None:
    with pytest.raises(CodecError):
        sim_from_bytes(payload)


def test_episode_round_trips_with_its_config() -> None:
    episode = Episode.start(606, EnvConfig(player_count=5))
    rng = random.Random(606)
    for _ in range(25):
        _, offer = episode.pending()
        episode, _ = episode.step(rng.randrange(len(offer)))
    restored = Episode.from_bytes(episode.to_bytes())
    assert restored.sim == episode.sim
    assert restored.config == episode.config
    assert restored.seed == episode.seed
    assert restored.seats == episode.seats
    assert restored.pending() == episode.pending()


def test_episode_payload_without_config_is_refused() -> None:
    with pytest.raises(CodecError, match="missing its 'config' object"):
        Episode.from_dict({"seed": 1, "sim": {}})
