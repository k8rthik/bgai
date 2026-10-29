"""MCTSAgent(vp_weight): each seat's objective is share + w * VP/150."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from bgai.agents.mcts import MCTSAgent
from bgai.arena.driver import new_game
from bgai.engine.tm.setup import load_setup
from bgai.training.model import ModelConfig, PolicyValueNet


def _net(aux: bool) -> PolicyValueNet:
    torch.manual_seed(0)
    cfg = ModelConfig(hidden=64, embed=32, move_hidden=32, dropout=0.0,
                      value_simplex=True, aux_head=aux)
    return PolicyValueNet(cfg)


def _sim():
    return new_game(load_setup("4pLeague_S10_D1L1_G1"))


def test_zero_weight_is_the_share_objective() -> None:
    agent = MCTSAgent(net=_net(aux=True))
    _, value = agent._evaluate(*_pending(agent))
    assert abs(float(value.sum()) - 1.0) < 1e-5


def test_leaf_value_adds_weighted_aux_vp() -> None:
    net = _net(aux=True)
    plain, weighted = MCTSAgent(net=net), MCTSAgent(net=net, vp_weight=0.5)
    game, faction, offer = _pending(plain)
    _, v0 = plain._evaluate(game, faction, offer)
    _, v1 = weighted._evaluate(game, faction, offer)
    from bgai.training.encode_move import encode_move
    from bgai.training.encode_state import encode_state
    from bgai.training.vocab import FACTION_INDEX

    enc = encode_state(game, faction)
    cand = np.stack([encode_move(m, game, faction) for m in offer])
    with torch.no_grad():
        _, _, aux = net.forward_with_aux(
            torch.from_numpy(enc.hex_planes.astype(np.float32))[None],
            torch.from_numpy(enc.globals.astype(np.float32))[None],
            torch.tensor([FACTION_INDEX[faction]]),
            torch.from_numpy(cand.astype(np.int64))[None],
            torch.ones((1, len(offer)), dtype=torch.bool),
        )
    np.testing.assert_allclose(v1, v0 + 0.5 * aux[0, :4].numpy(), atol=1e-5)


def test_terminal_value_adds_weighted_vp() -> None:
    agent = MCTSAgent(net=_net(aux=True), vp_weight=0.3)
    sim = _sim()
    vps = np.array([sim.game.factions[f].vp for f in sim.game.setup.factions], dtype=np.float32)
    expected = vps / vps.sum() + 0.3 * vps / 150.0
    np.testing.assert_allclose(agent._terminal_value(sim), expected, atol=1e-6)


def test_needs_an_aux_head() -> None:
    with pytest.raises(ValueError, match="aux_head"):
        MCTSAgent(net=_net(aux=False), vp_weight=0.3)


def _pending(agent: MCTSAgent):
    from bgai.arena.driver import decision

    sim = _sim()
    faction, offer = decision(sim)
    return sim.game, faction, offer
