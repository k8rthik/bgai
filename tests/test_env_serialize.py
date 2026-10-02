"""Observation serialization: dict, bytes and JSON forms all round-trip
losslessly, and a decoded observation still satisfies the Gymnasium space.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from bgai.env.config import MAX_PLAYERS, MIN_PLAYERS, EnvConfig
from bgai.env.core import Episode
from bgai.env.observation import OBS_VERSION
from bgai.env.serialize import (
    ObservationCodecError,
    obs_from_bytes,
    obs_from_dict,
    obs_from_jsonable,
    obs_to_bytes,
    obs_to_dict,
    obs_to_jsonable,
)

COUNTS = list(range(MIN_PLAYERS, MAX_PLAYERS + 1))


def _observations(count: int, steps: int = 20):
    episode = Episode.start(17, EnvConfig(player_count=count))
    rng = random.Random(count)
    for _ in range(steps):
        if episode.finished:
            return
        faction, offer = episode.pending()
        yield episode.observe(faction)
        episode, _ = episode.step(rng.randrange(len(offer)))


@pytest.mark.parametrize("count", COUNTS)
def test_bytes_round_trip(count: int) -> None:
    for obs in _observations(count):
        assert obs_from_bytes(obs_to_bytes(obs)) == obs


@pytest.mark.parametrize("count", COUNTS)
def test_dict_round_trip(count: int) -> None:
    for obs in _observations(count):
        assert obs_from_dict(obs_to_dict(obs)) == obs


@pytest.mark.parametrize("count", COUNTS)
def test_jsonable_round_trip(count: int) -> None:
    for obs in _observations(count):
        assert obs_from_jsonable(obs_to_jsonable(obs)) == obs


def test_jsonable_form_is_actually_json_serializable() -> None:
    import json

    obs = next(_observations(4))
    payload = json.loads(json.dumps(obs_to_jsonable(obs)))
    assert obs_from_jsonable(payload) == obs


def test_round_trip_preserves_dtypes() -> None:
    """Dtype survival is the load-bearing part: ``action_mask`` must stay
    int8 or ``Discrete.sample(mask)`` rejects it, and PettingZoo's
    ``api_test`` asserts the observation's dtypes match its space.
    """
    obs = next(_observations(4))
    for decoded in (
        obs_from_bytes(obs_to_bytes(obs)),
        obs_from_dict(obs_to_dict(obs)),
        obs_from_jsonable(obs_to_jsonable(obs)),
    ):
        assert decoded.hex_planes.dtype == np.int8
        assert decoded.globals.dtype == np.int16
        assert decoded.candidates.dtype == np.int16
        assert decoded.action_mask.dtype == np.int8


def test_decoded_observation_is_still_in_the_declared_space() -> None:
    from bgai.env.spaces import observation_space

    config = EnvConfig(player_count=4)
    space = observation_space(config.max_candidates)
    obs = next(_observations(4))
    assert space.contains(obs_to_dict(obs))
    assert space.contains(obs_to_dict(obs_from_bytes(obs_to_bytes(obs))))


def test_dict_form_is_the_shape_the_envs_hand_back() -> None:
    obs = next(_observations(4))
    payload = obs_to_dict(obs)
    assert set(payload) == {"observation", "action_mask"}
    assert set(payload["observation"]) == {"hex_planes", "globals", "candidates"}


def test_missing_top_level_keys_are_refused() -> None:
    with pytest.raises(ObservationCodecError, match="needs 'observation' and"):
        obs_from_dict({"action_mask": np.zeros(4, dtype=np.int8)})


def test_missing_inner_keys_are_refused() -> None:
    with pytest.raises(ObservationCodecError, match="missing \\['globals'\\]"):
        obs_from_dict(
            {
                "observation": {
                    "hex_planes": np.zeros((113, 18), dtype=np.int8),
                    "candidates": np.zeros((4, 12), dtype=np.int16),
                },
                "action_mask": np.zeros(4, dtype=np.int8),
            }
        )


def test_wrong_hex_plane_shape_is_refused() -> None:
    obs = next(_observations(4))
    payload = obs_to_dict(obs)
    payload["observation"] = dict(payload["observation"])
    payload["observation"]["hex_planes"] = np.zeros((2, 2), dtype=np.int8)
    with pytest.raises(ObservationCodecError, match="hex_planes must have shape"):
        obs_from_dict(payload)


def test_mask_length_must_match_candidate_rows() -> None:
    obs = next(_observations(4))
    payload = obs_to_dict(obs)
    payload["action_mask"] = np.zeros(7, dtype=np.int8)
    with pytest.raises(ObservationCodecError, match="does not match candidates rows"):
        obs_from_dict(payload)


def test_wrong_candidate_width_is_refused() -> None:
    obs = next(_observations(4))
    payload = obs_to_dict(obs)
    payload["observation"] = dict(payload["observation"])
    payload["observation"]["candidates"] = np.zeros((4, 3), dtype=np.int16)
    payload["action_mask"] = np.zeros(4, dtype=np.int8)
    with pytest.raises(ObservationCodecError, match="candidates must be"):
        obs_from_dict(payload)


def test_foreign_layout_version_is_refused() -> None:
    obs = next(_observations(4))
    payload = obs_to_jsonable(obs)
    payload["obs_version"] = OBS_VERSION + 1
    with pytest.raises(ObservationCodecError, match="observation layout v"):
        obs_from_jsonable(payload)


def test_unstamped_bytes_are_refused() -> None:
    import io

    buffer = io.BytesIO()
    np.savez(buffer, hex_planes=np.zeros((113, 18), dtype=np.int8))
    with pytest.raises(ObservationCodecError, match="no obs_version stamp"):
        obs_from_bytes(buffer.getvalue())


def test_garbage_bytes_are_refused() -> None:
    with pytest.raises(ObservationCodecError, match="not a valid observation payload"):
        obs_from_bytes(b"definitely not an npz")


def test_incomplete_payload_is_refused() -> None:
    import io

    buffer = io.BytesIO()
    np.savez(
        buffer,
        obs_version=np.array(OBS_VERSION, dtype=np.int32),
        hex_planes=np.zeros((113, 18), dtype=np.int8),
    )
    with pytest.raises(ObservationCodecError, match="payload is missing"):
        obs_from_bytes(buffer.getvalue())


def test_incomplete_jsonable_payload_is_refused() -> None:
    with pytest.raises(ObservationCodecError, match="payload is missing"):
        obs_from_jsonable({"obs_version": OBS_VERSION, "globals": []})
