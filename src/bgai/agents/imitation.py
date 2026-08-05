"""Imitation-net agent (plan Task 9).

Wraps a trained policy/value checkpoint in the arena's ``Agent``
protocol. The offer the arena passes is already in canonical order --
the same order the training pipeline indexed candidates in -- so the
net's argmax over the offer is directly the predicted expert move.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch

from bgai.data.ledger_parser import ParsedCommand
from bgai.engine.tm.state import GameState
from bgai.training.encode_move import encode_move
from bgai.training.encode_state import encode_state
from bgai.training.model import ModelConfig, PolicyValueNet
from bgai.training.vocab import ENCODING_VERSION, FACTION_INDEX


DEFAULT_TEMPERATURE = 1.0
"""Sample from the policy rather than taking its argmax.

Measured, not assumed (80 mirrored games per setting vs the greedy
baseline, seed 4242, checkpoint imitation_v1):

    T=0.0  imitation 1.562 vs greedy 1.413   (-0.150)
    T=0.3  imitation 1.531 vs greedy 1.381   (-0.150)
    T=0.5  imitation 1.419 vs greedy 1.488   (+0.069)
    T=0.7  imitation 1.394 vs greedy 1.525   (+0.131)
    T=1.0  imitation 1.363 vs greedy 1.531   (+0.169)

Greedy *beats* the argmax policy and loses to the sampled one, and the
trend is monotonic in temperature. A behavior-cloned policy's argmax is
brittle: it commits to the single most-imitated move in states the
expert corpus never contains, and repeats that commitment every time the
same state recurs. Sampling at the trained distribution keeps the
diversity the human data actually had. T=1.0 is the honest default --
it is the policy as trained, with no sharpening.
"""


class ImitationAgent:
    """Temperature-sampled (or, at T=0, argmax) policy over the offer."""

    def __init__(
        self,
        checkpoint_path: Path | None = None,
        name: str = "imitation",
        temperature: float = DEFAULT_TEMPERATURE,
        device: str = "cpu",
        net: PolicyValueNet | None = None,
    ) -> None:
        self.name = name
        self.temperature = temperature
        self.device = torch.device(device)
        if net is not None:
            self.net = net
        else:
            if checkpoint_path is None:
                raise ValueError("either checkpoint_path or net must be given")
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            if ckpt.get("encoding_version") != ENCODING_VERSION:
                raise ValueError(
                    f"checkpoint encoding v{ckpt.get('encoding_version')} != code "
                    f"v{ENCODING_VERSION} -- retrain or check out the matching commit"
                )
            self.net = PolicyValueNet(ModelConfig())
            self.net.load_state_dict(ckpt["model"])
        self.net.to(self.device).eval()

    @torch.no_grad()
    def choose(
        self,
        state: GameState,
        faction: str,
        offer: tuple[ParsedCommand, ...],
        rng: random.Random,
    ) -> ParsedCommand:
        enc = encode_state(state, faction)
        candidates = np.stack([encode_move(m, state, faction) for m in offer])
        hex_planes = torch.from_numpy(enc.hex_planes.astype(np.float32))[None].to(self.device)
        globals_ = torch.from_numpy(enc.globals.astype(np.float32))[None].to(self.device)
        cand = torch.from_numpy(candidates.astype(np.int64))[None].to(self.device)
        mask = torch.ones((1, len(offer)), dtype=torch.bool, device=self.device)
        faction_id = torch.tensor([FACTION_INDEX[faction]], device=self.device)

        logits, _ = self.net(hex_planes, globals_, faction_id, cand, mask)
        if self.temperature <= 0:
            return offer[int(logits[0].argmax().item())]
        probs = (logits[0] / self.temperature).softmax(dim=-1).cpu().numpy()
        # draw from the arena's seeded rng so games stay reproducible
        threshold = rng.random()
        cumulative = float(np.cumsum(probs)[-1])
        target = threshold * cumulative
        running = 0.0
        for index, p in enumerate(probs):
            running += float(p)
            if running >= target:
                return offer[index]
        return offer[-1]
