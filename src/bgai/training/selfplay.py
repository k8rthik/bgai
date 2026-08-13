"""Human-regularized self-play (Phase 6b).

Self-play games are generated with MCTS, and the net is fine-tuned on
three signals at once:

1. **policy** toward the search's visit distribution (search is stronger
   than the raw policy, so its choices are a better target);
2. **value** toward the game's actual final VP shares;
3. **KL toward the frozen human policy** -- the Cicero/piKL move. Pure
   self-play in Terra Mystica is known to go badly (the Digidiced
   post-mortem in the master plan: pure self-play lost to a heuristic
   MCTS), because a 4-player economic game has enormous strategy space
   and self-play drifts into conventions no human would punish. Anchoring
   to the human policy keeps the agent inside the region the corpus
   actually covers, which is also the region it will be evaluated in
   (against humans, in Phase 8).

``lambda_kl`` trades strength against drift: 0 is pure self-play, large
values pin the policy to imitation. It is a knob to sweep, not a constant
to guess, so it lives in the config.

**Scale reality (measured, not assumed).** One 4-seat MCTS self-play
game at 64 simulations costs ~14 s on an M3 Pro. A meaningful RL run is
10^5 games -- ~16 CPU-days here. This module is therefore written to be
correct and cluster-portable (checkpoint/resume, config-driven, no local
paths), demonstrated at small scale locally, and left ready for the
MSI Agate request the master plan anticipates. The alternative -- running
a token 200-game "RL phase" locally and reporting a number -- would be
measuring noise.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from bgai.agents.mcts import MCTSAgent
from bgai.arena.driver import advance, decision, new_game
from bgai.arena.setups import sample_setup
from bgai.engine.tm.setup import GameSetup
from bgai.training.encode_move import encode_move
from bgai.training.encode_state import encode_state
from bgai.training.model import PolicyValueNet
from bgai.training.rank_loss import pairwise_rank_loss
from bgai.training.vocab import FACTION_INDEX


@dataclass(frozen=True)
class SelfPlayRecord:
    """One decision from a self-play game."""

    hex_planes: np.ndarray
    globals: np.ndarray
    candidates: np.ndarray
    visits: np.ndarray  # MCTS visit counts, the policy target
    faction_id: int
    seat: int  # mover's absolute seat, to read its own final share
    final_shares: np.ndarray | None = None  # filled in when the game ends


@dataclass(frozen=True)
class SelfPlayConfig:
    """Search settings mirror the arena sweep's winning deep-search
    config (top_k=8, max_depth=48; docs/decisions.md C6 and the Aug 13
    sweep): the whole premise of self-play is distilling a searcher that
    is *stronger* than the raw policy, and that margin was only ever
    measured with pruning and depth extension on."""

    simulations: int = 512
    top_k: int = 8
    max_depth: int = 48
    leaf_batch: int = 16
    temperature: float = 1.0
    temp_decisions: int = 30
    """Sample moves ∝ visits for this many recorded decisions per game,
    then switch to argmax (AlphaZero's temperature cutoff). Early
    sampling diversifies openings; late argmax keeps the value target
    (realised final shares) close to what best play would have produced."""
    max_decisions: int = 5000
    lambda_kl: float = 1.0
    rank_weight: float = 1.0
    """Weight on the pairwise ranking term over the value head. Not
    optional in spirit: the first RL leg trained value on MSE alone and
    measurably traded ordering for magnitude (held-out pairwise 0.753 ->
    0.715, winner-top1 0.601 -> 0.534 while MSE improved 46x) -- and
    search consumes orderings, so RL search LOST to pre-RL search 21.7%
    to 30.8% even as the RL policy beat its predecessor. Same
    dissociation rank_loss.py documents for the supervised phase."""
    lr: float = 1e-4


def play_game(
    agent: MCTSAgent, setup: GameSetup, rng: random.Random, cfg: SelfPlayConfig
) -> list[SelfPlayRecord]:
    """One self-play game; returns every decision with its search
    distribution and the game's realised final VP shares."""
    sim = new_game(setup)
    records: list[SelfPlayRecord] = []
    seats = setup.factions
    recorded = 0
    while (pending := decision(sim)) is not None:
        if sim.decisions >= cfg.max_decisions:
            break
        faction, offer = pending
        if not offer:
            # Engine edge: a pending decision with zero legal moves (the
            # iter-5 crash; suspected ACTC double-turn corner). The game
            # cannot proceed, and a truncated game would put wrong
            # final-share labels on every record -- drop it loudly.
            print(
                f"WARN: dropping self-play game {setup.game_id}: "
                f"zero-move decision for {faction} at decision {sim.decisions}",
                flush=True,
            )
            return []
        if len(offer) == 1:
            sim = advance(sim, offer[0])
            continue
        root = agent.search(sim)
        assert root is not None  # decision(sim) was non-None with a real offer
        visits = root.visits.astype(np.float32)
        enc = encode_state(sim.game, faction)
        records.append(
            SelfPlayRecord(
                hex_planes=enc.hex_planes,
                globals=enc.globals,
                candidates=np.stack([encode_move(m, sim.game, faction) for m in offer]),
                visits=visits,
                faction_id=FACTION_INDEX[faction],
                seat=seats.index(faction),
            )
        )
        recorded += 1
        total = visits.sum()
        explore = cfg.temperature > 0 and recorded <= cfg.temp_decisions
        if total <= 0:
            choice = offer[int(np.argmax(root.priors))]
        elif explore:
            probs = visits / total
            choice = offer[int(rng.choices(range(len(offer)), weights=probs, k=1)[0])]
        else:
            choice = offer[int(np.argmax(visits))]
        sim = advance(sim, choice)

    vps = np.array([sim.game.factions[f].vp for f in seats], dtype=np.float32)
    total_vp = float(vps.sum())
    shares = vps / total_vp if total_vp > 0 else np.full(len(seats), 0.25, dtype=np.float32)
    return [
        SelfPlayRecord(
            hex_planes=r.hex_planes,
            globals=r.globals,
            candidates=r.candidates,
            visits=r.visits,
            faction_id=r.faction_id,
            seat=r.seat,
            final_shares=np.roll(shares, -r.seat),  # mover-relative, like the net
        )
        for r in records
    ]


def regularized_loss(
    net: PolicyValueNet,
    frozen: PolicyValueNet,
    batch: dict[str, torch.Tensor],
    lambda_kl: float,
    rank_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Search-policy cross-entropy + value MSE + pairwise rank + KL(new || human).

    ``frozen`` is the imitation net, kept in eval mode and never updated
    -- it is the anchor, so it must not drift with the policy it is
    anchoring.
    """
    logits, value = net(
        batch["hex_planes"], batch["globals"], batch["faction"],
        batch["candidates"], batch["cand_mask"],
    )
    with torch.no_grad():
        ref_logits, _ = frozen(
            batch["hex_planes"], batch["globals"], batch["faction"],
            batch["candidates"], batch["cand_mask"],
        )

    # Padding slots carry -inf logits, so log_softmax gives -inf there and
    # the soft targets give 0 -- and 0 * -inf is NaN, which silently
    # poisons the whole batch. (The imitation loop never hit this: its
    # cross_entropy reads only the chosen index.) Zero those terms out
    # explicitly rather than relying on the multiply.
    mask = batch["cand_mask"]
    log_probs = F.log_softmax(logits, dim=-1)
    safe_log_probs = log_probs.masked_fill(~mask, 0.0)
    target = batch["visits"] / batch["visits"].sum(dim=-1, keepdim=True).clamp(min=1e-6)
    policy_loss = -(target * safe_log_probs).sum(dim=-1).mean()
    value_loss = F.mse_loss(value, batch["value"])

    ref_log_probs = F.log_softmax(ref_logits, dim=-1)
    kl_terms = (log_probs.exp() * (safe_log_probs - ref_log_probs.masked_fill(~mask, 0.0)))
    kl = kl_terms.masked_fill(~mask, 0.0).sum(dim=-1).mean()
    rank = pairwise_rank_loss(value, batch["value"])
    loss = policy_loss + 0.5 * value_loss + rank_weight * rank + lambda_kl * kl
    return loss, {
        "policy": float(policy_loss.item()),
        "value": float(value_loss.item()),
        "rank": float(rank.item()),
        "kl": float(kl.item()),
        "total": float(loss.item()),
    }


def generate(
    checkpoint: Path,
    games: int,
    seed: int = 0,
    cfg: SelfPlayConfig | None = None,
    device: str = "cpu",
) -> list[SelfPlayRecord]:
    """Generate ``games`` self-play games from a checkpoint. Setups are
    corpus-sampled, exactly as the arena does, so self-play stays on the
    distribution the agent is evaluated on."""
    cfg = cfg or SelfPlayConfig()
    agent = MCTSAgent(
        checkpoint,
        simulations=cfg.simulations,
        max_depth=cfg.max_depth,
        top_k=cfg.top_k,
        leaf_batch=cfg.leaf_batch,
        device=device,
    )
    rng = random.Random(seed)
    out: list[SelfPlayRecord] = []
    for _ in range(games):
        out.extend(play_game(agent, sample_setup(rng), rng, cfg))
    return out
