"""max^n MCTS guided by the imitation net (Phase 6).

Terra Mystica is deterministic and perfect-information once the setup is
fixed (score tiles and the bonus pool are public from the start), so the
search needs no chance nodes and no determinization -- the tree is the
game tree.

Four-player games break the minimax/negamax assumption that one player's
gain is another's loss, so each node carries a **value vector**, one
component per mover-relative seat (Petosa & Balch's max^n, the shape the
master plan calls for). Selection maximizes the *acting* seat's own
component; backup adds the whole vector along the path. A scalar
"win probability" would force the search to pretend TM is zero-sum
between two of the four seats.

Leaf evaluation comes from the imitation net's value head (a predicted
final-VP share per seat), so there are no random rollouts: a rollout in
TM is ~200 decisions of noise, while the value head was trained on
1.2M real positions. The same net supplies the policy prior for PUCT.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from bgai.agents.leaf_eval import blend, computed_value
from bgai.arena.driver import SimState, advance, decision
from bgai.data.ledger_parser import ParsedCommand
from bgai.engine.tm.apply import EngineError
from bgai.engine.tm.state import GameState
from bgai.training.encode_move import MOVE_FIELDS, encode_move
from bgai.training.encode_state import encode_state
from bgai.training.model import (
    ModelConfig,
    PolicyValueNet,
    model_config_from_checkpoint,
)
from bgai.training.vocab import ENCODING_VERSION, FACTION_INDEX


@dataclass
class _Node:
    """One decision point in the search tree."""

    sim: SimState
    faction: str
    offer: tuple[ParsedCommand, ...]
    priors: np.ndarray
    seat_of: dict[str, int]
    children: dict[int, "_Node | None"] = field(default_factory=dict)
    visits: np.ndarray | None = None
    action_value: np.ndarray | None = None  # (n_actions, 4) summed value vectors
    total_visits: int = 0
    allowed: np.ndarray | None = None  # cached top-k prior indices

    def __post_init__(self) -> None:
        n = len(self.offer)
        self.visits = np.zeros(n, dtype=np.int32)
        self.action_value = np.zeros((n, 4), dtype=np.float32)


class MCTSAgent:
    """Imitation-guided max^n search.

    ``simulations`` is per decision. Each simulation walks PUCT-selected
    edges to a leaf, evaluates the leaf with the net's value head, and
    backs the vector up. With ``simulations=0`` the agent degenerates to
    sampling the raw policy (useful as a control in ablations).

    ``temperature`` applies to the final visit counts. It is 0 (argmax
    over visits), the usual choice. It was briefly 1.0 on the strength of
    D6.6, which turned out to share a root cause with D5.6: the driver's
    uncapped free-action loop, not the selection rule. Fixed in
    arena/driver.py; see docs/decisions.md D5.6/D6.6.

    ``value_blend_w`` (D6.9, diagnostic A) blends each leaf's learned
    value with ``leaf_eval.computed_value``, an engine-computed VP-share
    projection robust off-distribution by construction:
    ``(1-w)*learned + w*computed``. Default 0.0 reproduces the learned
    value head exactly -- unchanged behaviour until a sweep result
    justifies otherwise; see docs/decisions.md D6.9.
    """

    def __init__(
        self,
        checkpoint_path: Path | None = None,
        name: str = "mcts",
        simulations: int = 64,
        c_puct: float = 1.5,
        temperature: float = 0.0,
        device: str = "cpu",
        net: PolicyValueNet | None = None,
        max_depth: int = 24,
        value_blend_w: float = 0.0,
        leaf_batch: int = 0,
        top_k: int = 0,
        late_sims: int = 0,
        root_choice: str = "visits",
        evaluator=None,
    ) -> None:
        self.name = name
        self.simulations = simulations
        self.c_puct = c_puct
        self.temperature = temperature
        self.max_depth = max_depth
        self.value_blend_w = value_blend_w
        self.leaf_batch = leaf_batch
        """Leaves evaluated per forward pass. 0/1 keeps the sequential
        path exactly. Higher batches amortise the network call that
        otherwise runs once per simulation, at the cost of virtual-loss
        approximation: descents in the same batch cannot see each other's
        results."""
        self.top_k = top_k
        """Consider only this many highest-prior moves per node (0 = all).
        Branching is 23 on average, so an unpruned tree at 512 sims is
        about two levels deep; pruning spends the budget deeper instead."""
        self.root_choice = root_choice
        """Final move rule: "visits" (robust child, the classical and
        current default) or "q" (max child: highest own-seat mean value
        among sufficiently-visited moves). With the winner-CE-trained
        simplex head, a move's Q approximates winner-mass -- "q" is the
        seek-the-win rule, willing to prefer a lower-visit line whose
        value says it flips placement."""
        self.late_sims = late_sims
        """Simulation budget for rounds 5-6 (0 = use ``simulations``
        throughout). Every diagnostic since C1 shows the same signature:
        the agent develops well and converts badly, and conversion
        happens in the endgame -- where the tree is also smallest, so
        deep budgets go furthest. Lean-early/deep-late spends the same
        average compute where the measured weakness lives."""
        self._reuse_root: _Node | None = None
        """Tree reuse (opt-in via note_advance): the subtree under the
        played move carries its visits into the next decision instead of
        being discarded, and ``search`` tops the budget up to
        ``simulations`` total root visits. Only a driver that reports
        EVERY advance (self-play's play_game) can use this -- an arena
        agent sees only its own seats' decisions, so unreported opponent
        moves would silently desynchronize the tree. The (game_id,
        decisions, faction, offer) match in ``search`` is the safety net:
        any mismatch falls back to a fresh expansion."""
        self.evaluator = evaluator
        """Optional remote evaluator (agents/inference_server.py): called
        with pre-encoded numpy batches instead of the local net. When
        set, this process never runs torch forward passes -- a central
        server batches every worker's leaves into one GPU call."""
        self.device = torch.device(device)
        if evaluator is not None:
            self.net = None  # type: ignore[assignment]
            return
        if net is not None:
            self.net = net
        else:
            if checkpoint_path is None:
                raise ValueError("either checkpoint_path or net must be given")
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            if ckpt.get("encoding_version") != ENCODING_VERSION:
                raise ValueError(
                    f"checkpoint encoding v{ckpt.get('encoding_version')} != code "
                    f"v{ENCODING_VERSION}"
                )
            self.net = PolicyValueNet(model_config_from_checkpoint(ckpt))
            self.net.load_state_dict(ckpt["model"])
        self.net.to(self.device).eval()

    # -- net access -------------------------------------------------------

    @torch.no_grad()
    def _evaluate(
        self, game: GameState, faction: str, offer: tuple[ParsedCommand, ...]
    ) -> tuple[np.ndarray, np.ndarray]:
        """(policy over ``offer``, value vector in mover-relative seat
        order) for one position."""
        if self.evaluator is not None:
            return self._evaluate_many([(game, faction, offer)])[0]
        enc = encode_state(game, faction)
        cand = np.stack([encode_move(m, game, faction) for m in offer])
        logits, value = self.net(
            torch.from_numpy(enc.hex_planes.astype(np.float32))[None].to(self.device),
            torch.from_numpy(enc.globals.astype(np.float32))[None].to(self.device),
            torch.tensor([FACTION_INDEX[faction]], device=self.device),
            torch.from_numpy(cand.astype(np.int64))[None].to(self.device),
            torch.ones((1, len(offer)), dtype=torch.bool, device=self.device),
        )
        return (
            logits[0].softmax(dim=-1).cpu().numpy(),
            value[0].cpu().numpy(),
        )

    @torch.no_grad()
    def _evaluate_many(
        self, items: Sequence[tuple[GameState, str, tuple[ParsedCommand, ...]]]
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """``_evaluate`` for several positions in one forward pass.

        Search costs one pass per simulation, so a 512-sim move runs 512
        sequential passes of a 3.8M-parameter net -- the reason a 120-game
        512-sim series takes over an hour. Batching amortises that.

        Candidate sets are ragged (mean 23, max 216), so rows are padded to
        the batch maximum and masked. The mask matters: ``model.forward``
        fills masked logits with -inf before the softmax, so padding
        contributes no probability mass and each row's priors still sum
        to 1 over its own candidates.
        """
        if not items:
            return []
        width = max(len(offer) for _, _, offer in items)
        n = len(items)
        hex_planes, globals_, factions = [], [], []
        cand = np.zeros((n, width, MOVE_FIELDS), dtype=np.int64)
        mask = np.zeros((n, width), dtype=bool)
        for i, (game, faction, offer) in enumerate(items):
            enc = encode_state(game, faction)
            hex_planes.append(enc.hex_planes.astype(np.float32))
            globals_.append(enc.globals.astype(np.float32))
            factions.append(FACTION_INDEX[faction])
            moves = np.stack([encode_move(m, game, faction) for m in offer])
            cand[i, : len(offer)] = moves
            mask[i, : len(offer)] = True
        if self.evaluator is not None:
            priors, values = self.evaluator(
                np.stack(hex_planes), np.stack(globals_),
                np.asarray(factions, dtype=np.int64), cand, mask,
            )
            return [
                (priors[i, : len(items[i][2])], values[i]) for i in range(n)
            ]
        logits, value = self.net(
            torch.from_numpy(np.stack(hex_planes)).to(self.device),
            torch.from_numpy(np.stack(globals_)).to(self.device),
            torch.tensor(factions, device=self.device),
            torch.from_numpy(cand).to(self.device),
            torch.from_numpy(mask).to(self.device),
        )
        priors = logits.softmax(dim=-1).cpu().numpy()
        values = value.cpu().numpy()
        return [
            (priors[i, : len(items[i][2])], values[i]) for i in range(n)
        ]

    def _seat_map(self, game: GameState, faction: str) -> dict[str, int]:
        """Absolute faction -> index into a value vector encoded from
        ``faction``'s point of view. Every node's vector is re-based to
        the ROOT mover's seat order before backup, so the components mean
        the same thing all the way up the tree.
        """
        seats = game.setup.factions
        pivot = seats.index(faction)
        order = seats[pivot:] + seats[:pivot]
        return {name: i for i, name in enumerate(order)}

    # -- tree -------------------------------------------------------------

    def _expand(self, sim: SimState) -> tuple[_Node | None, np.ndarray | None]:
        pending = decision(sim)
        if pending is None:
            return None, self._terminal_value(sim)
        faction, offer = pending
        if not offer:
            # Engine edge (selfplay_deep iter-5 crash): a pending decision
            # with zero legal moves -- seen in rare search lines, suspected
            # ACTC double-turn corner. The position can't be searched or
            # net-evaluated (no candidates), so value it with the engine's
            # own VP projection and stop descending.
            return None, computed_value(sim.game)
        priors, value = self._evaluate(sim.game, faction, offer)
        node = _Node(
            sim=sim,
            faction=faction,
            offer=offer,
            priors=priors,
            seat_of=self._seat_map(sim.game, faction),
        )
        return node, self._leaf_value(value, node.seat_of, faction, sim.game)

    def _terminal_value(self, sim: SimState) -> np.ndarray:
        """Final VP shares, in absolute seat order of the setup."""
        seats = sim.game.setup.factions
        vps = np.array([sim.game.factions[f].vp for f in seats], dtype=np.float32)
        total = float(vps.sum())
        return vps / total if total > 0 else np.full(len(seats), 0.25, dtype=np.float32)

    def _rebase(
        self,
        value_rel: np.ndarray,
        seat_of: dict[str, int],
        faction: str,
        game: GameState,
    ) -> np.ndarray:
        """Convert a mover-relative value vector into absolute seat order
        (``setup.factions``), the common frame used inside the tree."""
        seats = game.setup.factions
        out = np.zeros(len(seats), dtype=np.float32)
        for name, rel in seat_of.items():
            out[seats.index(name)] = value_rel[rel]
        return out

    def _leaf_value(
        self,
        value_rel: np.ndarray,
        seat_of: dict[str, int],
        faction: str,
        game: GameState,
    ) -> np.ndarray:
        """Rebase a mover-relative net value to absolute seat order, then
        blend in the engine-computed projection per ``value_blend_w``
        (D6.9; a no-op at the default 0.0)."""
        abs_value = self._rebase(value_rel, seat_of, faction, game)
        if self.value_blend_w <= 0:
            return abs_value
        return blend(abs_value, game, self.value_blend_w)

    def _allowed(self, node: _Node) -> np.ndarray | None:
        """Indices selection may consider, or None when every move is open.

        Mean branching is 23 (max 216), so 512 simulations spread over
        every legal move leave the tree about two levels deep -- far short
        of what a 6-round game needs. Restricting to the top ``top_k``
        priors spends the same budget deeper. Safe only because the policy
        is good enough to put the right move in that set: top1 is 58% and
        top3 84%.
        """
        if not self.top_k or len(node.offer) <= self.top_k:
            return None
        if node.allowed is None:
            node.allowed = np.argsort(-node.priors)[: self.top_k]
        return node.allowed

    def _select(self, node: _Node) -> int:
        """PUCT over the acting seat's own value component."""
        seat = node.sim.game.setup.factions.index(node.faction)
        visits = node.visits
        q = np.zeros(len(node.offer), dtype=np.float32)
        visited = visits > 0
        q[visited] = node.action_value[visited, seat] / visits[visited]
        # unvisited edges inherit the parent's own current estimate, so a
        # single bad first result cannot permanently bury a move
        if visited.any():
            q[~visited] = q[visited].mean()
        else:
            q[:] = 0.25
        u = self.c_puct * node.priors * math.sqrt(max(node.total_visits, 1)) / (1 + visits)
        score = q + u
        allowed = self._allowed(node)
        if allowed is not None:
            masked = np.full(len(node.offer), -np.inf, dtype=np.float32)
            masked[allowed] = score[allowed]
            score = masked
        return int(np.argmax(score))

    def _simulate(self, root: _Node) -> None:
        path: list[tuple[_Node, int]] = []
        node = root
        depth = 0
        while True:
            action = self._select(node)
            path.append((node, action))
            child = node.children.get(action)
            if child is None:
                try:
                    child_sim = advance(node.sim, node.offer[action])
                except EngineError:
                    # A move the engine rejects is worth nothing; record a
                    # zero vector so the search stops trying it.
                    self._backup(path, np.zeros(4, dtype=np.float32))
                    node.children[action] = None
                    return
                new_node, value = self._expand(child_sim)
                node.children[action] = new_node
                self._backup(path, value)
                return
            node = child
            depth += 1
            if depth >= self.max_depth:
                _, value = self._evaluate(node.sim.game, node.faction, node.offer)
                self._backup(path, self._leaf_value(value, node.seat_of, node.faction, node.sim.game))
                return

    def _backup(self, path: list[tuple[_Node, int]], value: np.ndarray) -> None:
        for node, action in path:
            node.visits[action] += 1
            node.total_visits += 1
            node.action_value[action] += value

    def _apply_virtual_loss(self, path: list[tuple[_Node, int]]) -> None:
        """Count a visit with no value along the path being explored.

        PUCT is deterministic, so without this every descent in a batch
        follows the same optimal path to the same leaf and batching buys
        nothing. Adding a visit but no value drops that edge's Q
        (``action_value / visits``), so the next descent in the same batch
        prefers a different move. Reverted before the real backup.
        """
        for node, action in path:
            node.visits[action] += 1
            node.total_visits += 1

    def _revert_virtual_loss(self, path: list[tuple[_Node, int]]) -> None:
        for node, action in path:
            node.visits[action] -= 1
            node.total_visits -= 1

    def _descend_batch(
        self, root: _Node, count: int
    ) -> list[tuple[list[tuple[_Node, int]], _Node, int, SimState]]:
        """Walk ``count`` paths to unexpanded leaves, holding virtual loss.

        Paths that resolve without a network evaluation -- an engine
        rejection, or hitting ``max_depth`` -- are backed up immediately
        and not returned; only leaves needing a forward pass come back, so
        the caller can evaluate them in one batch.
        """
        pending: list[tuple[list[tuple[_Node, int]], _Node, int, SimState]] = []
        for _ in range(count):
            path: list[tuple[_Node, int]] = []
            node = root
            depth = 0
            while True:
                action = self._select(node)
                path.append((node, action))
                self._apply_virtual_loss([(node, action)])
                child = node.children.get(action)
                if child is None:
                    try:
                        child_sim = advance(node.sim, node.offer[action])
                    except EngineError:
                        self._revert_virtual_loss(path)
                        self._backup(path, np.zeros(4, dtype=np.float32))
                        node.children[action] = None
                        break
                    pending.append((path, node, action, child_sim))
                    break
                node = child
                depth += 1
                if depth >= self.max_depth:
                    _, value = self._evaluate(node.sim.game, node.faction, node.offer)
                    self._revert_virtual_loss(path)
                    self._backup(
                        path, self._leaf_value(value, node.seat_of, node.faction, node.sim.game)
                    )
                    break
        return pending

    def _simulate_batch(self, root: _Node, count: int) -> None:
        """One batched round: descend ``count`` paths, evaluate their
        leaves together, then expand and back up."""
        pending = self._descend_batch(root, count)
        if not pending:
            return
        items = []
        terminal: list[int] = []
        stuck: list[int] = []
        for i, (_path, _parent, _action, sim) in enumerate(pending):
            decided = decision(sim)
            if decided is None:
                terminal.append(i)
            elif not decided[1]:
                # zero-move decision: same engine edge _expand guards
                stuck.append(i)
            else:
                faction, offer = decided
                items.append((i, sim, faction, offer))

        results = self._evaluate_many([(sim.game, f, o) for _i, sim, f, o in items])
        for (i, sim, faction, offer), (priors, value_rel) in zip(items, results):
            path, parent, action, _ = pending[i]
            seat_of = self._seat_map(sim.game, faction)
            node = _Node(
                sim=sim,
                faction=faction,
                offer=offer,
                priors=priors,
                seat_of=seat_of,
            )
            parent.children[action] = node
            self._revert_virtual_loss(path)
            self._backup(path, self._leaf_value(value_rel, seat_of, faction, sim.game))
        for i in terminal:
            path, parent, action, sim = pending[i]
            parent.children[action] = None
            self._revert_virtual_loss(path)
            self._backup(path, self._terminal_value(sim))
        for i in stuck:
            path, parent, action, sim = pending[i]
            parent.children[action] = None
            self._revert_virtual_loss(path)
            self._backup(path, computed_value(sim.game))

    # -- Agent protocol ---------------------------------------------------

    def choose(
        self,
        state: GameState,
        faction: str,
        offer: tuple[ParsedCommand, ...],
        rng: random.Random,
    ) -> ParsedCommand:
        raise NotImplementedError(
            "MCTSAgent needs the driver's SimState, not just a GameState -- "
            "use choose_sim(); bgai.arena.sim.run_game routes through "
            "SimAgent when the agent exposes choose_sim."
        )

    def reset_tree(self) -> None:
        """Drop any retained subtree (call at game boundaries)."""
        self._reuse_root = None

    def note_advance(self, choice: ParsedCommand) -> None:
        """Walk the retained tree down the edge actually played.

        Call once per driver ``advance`` -- searched decisions and forced
        single-move steps alike. A missing/unexpanded child (or no
        retained tree) resets reuse; the next ``search`` starts fresh.
        """
        root = self._reuse_root
        self._reuse_root = None
        if root is None:
            return
        try:
            index = root.offer.index(choice)
        except ValueError:
            return
        child = root.children.get(index)
        if isinstance(child, _Node):
            self._reuse_root = child

    def _reusable(self, sim: SimState) -> _Node | None:
        node = self._reuse_root
        if node is None:
            return None
        if node.sim.decisions != sim.decisions:
            return None
        if node.sim.game.setup.game_id != sim.game.setup.game_id:
            return None
        pending = decision(sim)
        if pending is None or pending[0] != node.faction or pending[1] != node.offer:
            return None
        return node

    def search(self, sim: SimState) -> _Node | None:
        """Expand the root and spend the simulation budget on it,
        honouring ``leaf_batch``/``top_k``/``max_depth`` as configured.

        With a retained subtree matching ``sim`` (see ``note_advance``),
        its visits carry over and the budget tops up to ``simulations``
        total root visits -- same strength target, fewer fresh descents.

        Returns the searched root (None when no decision is pending).
        ``choose_sim`` argmaxes its visits; self-play reads the whole
        visit distribution as a policy target.
        """
        root = self._reusable(sim)
        if root is None:
            root, _ = self._expand(sim)
        if root is None:
            return None
        budget = (
            self.late_sims
            if self.late_sims and sim.game.round >= 5
            else self.simulations
        )
        remaining = max(0, budget - root.total_visits)
        if self.leaf_batch > 1:
            done = 0
            while done < remaining:
                step = min(self.leaf_batch, remaining - done)
                self._simulate_batch(root, step)
                done += step
        else:
            for _ in range(remaining):
                self._simulate(root)
        self._reuse_root = root
        return root

    def choose_sim(
        self, sim: SimState, faction: str, offer: tuple[ParsedCommand, ...], rng: random.Random
    ) -> ParsedCommand:
        if len(offer) == 1:
            return offer[0]
        root = self.search(sim)
        assert root is not None
        if root.total_visits == 0:
            return offer[int(np.argmax(root.priors))]
        counts = root.visits.astype(np.float64)
        if self.temperature <= 0:
            if self.root_choice == "q":
                # max child over own-seat Q, restricted to moves with at
                # least a materiality floor of visits (an unexplored
                # move's Q is noise, not a discovery)
                seat = root.sim.game.setup.factions.index(root.faction)
                floor = max(1, int(0.05 * root.total_visits))
                eligible = counts >= floor
                if eligible.any():
                    q = np.full(len(counts), -np.inf)
                    q[eligible] = (
                        root.action_value[eligible, seat] / counts[eligible]
                    )
                    return offer[int(np.argmax(q))]
            return offer[int(np.argmax(counts))]
        weights = counts ** (1.0 / self.temperature)
        total = weights.sum()
        if total <= 0:
            return offer[int(np.argmax(root.priors))]
        target = rng.random() * total
        running = 0.0
        for index, w in enumerate(weights):
            running += float(w)
            if running >= target:
                return offer[index]
        return offer[-1]
