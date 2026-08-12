"""Arena agent specs.

The arena CLI could only build `random` and `greedy`, so the trained nets
could not be measured against each other -- which is the only measurement
that says whether better imitation makes better *play*. Specs let a run
name a checkpoint and search settings.
"""

from __future__ import annotations

import pytest

from bgai.arena.run import parse_spec


def test_bare_name_has_no_options() -> None:
    assert parse_spec("greedy") == ("greedy", "greedy", {})


def test_options_are_parsed_and_typed() -> None:
    name, label, opts = parse_spec("mcts:ckpt=/tmp/a.pt,sims=128,c_puct=1.25")
    assert (name, label) == ("mcts", "mcts")
    assert opts == {"ckpt": "/tmp/a.pt", "sims": 128, "c_puct": 1.25}


def test_label_lets_two_configs_of_one_agent_coexist() -> None:
    """Two MCTS agents with different checkpoints need distinct rating
    keys, or the series collapses them into one competitor."""
    name, label, opts = parse_spec("mcts@new:ckpt=/tmp/new.pt")
    assert (name, label) == ("mcts", "new")
    assert opts["ckpt"] == "/tmp/new.pt"


def test_bool_option() -> None:
    _, _, opts = parse_spec("mcts:blend=0.5")
    assert opts["blend"] == 0.5


def test_unknown_agent_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown agent"):
        parse_spec("nonesuch:ckpt=/tmp/x.pt")


def test_malformed_option_is_rejected() -> None:
    with pytest.raises(ValueError, match="malformed option"):
        parse_spec("greedy:justakey")


def test_specs_split_on_agent_boundaries_not_option_commas() -> None:
    """Options are comma-separated too, so the splitter must tell a
    continuation (`sims=64`) from a new agent that happens to carry `=`."""
    from bgai.arena.run import _split_specs

    assert _split_specs("greedy,mcts@new:ckpt=a.pt,sims=64,imitation@old:ckpt=b.pt") == [
        "greedy",
        "mcts@new:ckpt=a.pt,sims=64",
        "imitation@old:ckpt=b.pt",
    ]
