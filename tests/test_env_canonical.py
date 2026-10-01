"""Hash-independent offer ordering.

The bug this guards against is documented in ``bgai.env.canonical``:
``legal_actions.bridge_moves`` unpacks a ``frozenset`` of two hexes, so the
same bridge is offered with its endpoints in either order depending on the
process's string-hash seed. Cross-process reproducibility of every action
index depends on normalizing that away.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from bgai.engine.tm.legal_shared import cmd
from bgai.env.canonical import (
    UNORDERED_PAIR_VERBS,
    canonical_offer,
    normalize_move,
    sort_key,
)


def test_only_bridge_has_an_unordered_hex_pair() -> None:
    assert UNORDERED_PAIR_VERBS == frozenset({"bridge"})


def test_bridge_endpoints_are_spelled_low_to_high() -> None:
    forward = cmd("bridge", loc="E4", loc2="G1")
    backward = cmd("bridge", loc="G1", loc2="E4")
    assert normalize_move(forward) == forward
    assert normalize_move(backward) == forward
    assert normalize_move(backward).loc == "E4"


def test_directional_verbs_are_left_alone() -> None:
    """A blanket loc/loc2 swap would silently reorder a verb where the
    orientation means something, so the normalization is verb-scoped.
    """
    move = cmd("connect", loc="G1", loc2="E4")
    assert normalize_move(move) is move
    build = cmd("build", loc="Z9")
    assert normalize_move(build) is build


def test_normalize_is_a_no_op_when_already_canonical() -> None:
    move = cmd("bridge", loc="A1", loc2="B2")
    assert normalize_move(move) is move


def test_normalize_handles_a_missing_endpoint() -> None:
    assert normalize_move(cmd("bridge", loc="A1")) == cmd("bridge", loc="A1")


def test_either_spelling_produces_the_same_order() -> None:
    """The property that matters: two processes that disagree about how the
    engine spelled a bridge still agree about every index.
    """
    base = [
        cmd("build", loc="C3"),
        cmd("burn", n1=1),
        cmd("done"),
        cmd("convert", res1="PW", res2="C", n1=1, n2=1),
    ]
    forward = tuple([*base, cmd("bridge", loc="E4", loc2="G1")])
    backward = tuple([*base, cmd("bridge", loc="G1", loc2="E4")])
    first, _ = canonical_offer(forward)
    second, _ = canonical_offer(backward)
    assert first == second


def test_canonical_offer_keeps_originals_index_aligned() -> None:
    """Index i of the normalized tuple must be the same move as index i of
    the originals -- the originals are what ``driver.advance`` receives.
    """
    offer = (
        cmd("bridge", loc="G1", loc2="E4"),
        cmd("done"),
        cmd("build", loc="A1"),
    )
    normalized, originals = canonical_offer(offer)
    assert len(normalized) == len(originals) == 3
    for norm, original in zip(normalized, originals, strict=True):
        assert norm == normalize_move(original)
    assert set(originals) == set(offer)


def test_order_is_independent_of_input_order() -> None:
    offer = (
        cmd("done"),
        cmd("build", loc="A1"),
        cmd("bridge", loc="E4", loc2="G1"),
    )
    forward, _ = canonical_offer(offer)
    reversed_, _ = canonical_offer(tuple(reversed(offer)))
    assert forward == reversed_


def test_sort_key_never_compares_none_against_a_number() -> None:
    """``None`` in any field would raise on comparison; the key maps it to a
    sentinel instead.
    """
    moves = [
        cmd("build", loc="A1"),
        cmd("burn", n1=3),
        cmd("convert", res1="PW", res2="C", n1=1, n2=1),
        cmd("done"),
    ]
    keys = [sort_key(m) for m in moves]
    assert sorted(keys) == sorted(keys)  # comparison does not raise
    assert all(isinstance(k[10], int) for k in keys)


def test_sort_key_distinguishes_reason_and_kind() -> None:
    """``driver.canonical_moves`` stops at n2; two commands differing only in
    ``reason`` would tie there and be left in set-iteration order. The env's
    key goes further so the order is total.
    """
    from bgai.data.ledger_parser import Kind

    first = cmd("lose_marker", reason="FREE_D")
    second = cmd("lose_marker", reason="FREE_TP")
    assert sort_key(first) != sort_key(second)
    third = replace(first, kind=Kind.BOOKKEEPING)
    assert sort_key(first) != sort_key(third)


def test_empty_offer_is_handled() -> None:
    assert canonical_offer(()) == ((), ())


@pytest.mark.parametrize("count", [2, 4])
def test_live_offers_are_already_sorted_by_the_key(count: int) -> None:
    """Sanity: what the env hands out really is in this order."""
    import random

    from bgai.env.config import EnvConfig
    from bgai.env.core import Episode

    episode = Episode.start(321, EnvConfig(player_count=count))
    rng = random.Random(321)
    for _ in range(60):
        if episode.finished:
            break
        _, offer = episode.pending()
        keys = [sort_key(m) for m in offer]
        assert keys == sorted(keys)
        episode, _ = episode.step(rng.randrange(len(offer)))
