"""LLM player, rungs L0-L2 (Phase 7).

The master plan's ladder is L0 bare -> L1 knowledge-loaded -> L2 engine
tools -> L3 retrieval -> L4 test-time search -> L5 LLM-guided MCTS ->
L6 fine-tuned. This module implements the first three rungs and the
scaffolding the rest hang off:

- **L0** serialized state + numbered legal moves, answer with an index.
- **L1** adds a knowledge block (rules digest, faction notes, the
  current round's scoring pressure) to the system prompt.
- **L2** adds engine tools the model can call before answering --
  `apply_and_show` (what-if a candidate), `neighbors`, `income_preview`.
  These are answered by the real engine, so the model can *calculate*
  instead of estimate.

Rungs are additive and selected by ``level``, so the arena can measure a
capability-vs-strength curve rather than one number.

**Answer protocol.** The model replies with the index of its chosen move.
An unparseable or out-of-range answer falls back to the last valid index
seen, then to 0 -- never to an illegal move, because the model never
names a move, only picks one. Fallbacks are counted in
``LLMAgent.fallbacks`` so a rung that cannot follow the protocol is
visible in the report rather than silently scoring as a weak player.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

from bgai.arena.driver import SimState, advance
from bgai.data.ledger_parser import ParsedCommand
from bgai.engine.tm.apply import EngineError
from bgai.engine.tm.factions_data import FACTIONS
from bgai.engine.tm.state import GameState
from bgai.llm.provider import Message, Provider
from bgai.llm.serialize import describe_move, neighbors, serialize_offer, serialize_state

_TOOL_RE = re.compile(r"TOOL\s+(\w+)\s*(.*)")

_L0_SYSTEM = """You are playing Terra Mystica (4 players, base map, base factions).
You will be shown the public game state and a numbered list of the legal moves
available to you right now. Choose the move that best improves your final score.

Answer with ONLY the number of your chosen move. No explanation."""

_L1_KNOWLEDGE = """Terra Mystica essentials:
- You score VP mainly from: the round's scoring tile, town founding (a
  connected group of >=4 buildings with >=7 power value), cult tracks at
  the end (8/4/2 for 1st/2nd/3rd on each of FIRE/WATER/EARTH/AIR), and
  final network size (18/12/6 for the largest connected group).
- Buildings: D (dwelling) -> TP (trading post) -> TE (temple) or SH
  (stronghold); TE -> SA (sanctuary). Income comes from buildings, so
  building early compounds. A TP next to an opponent gives THEM leech
  power; taking leech costs you 1 VP per power beyond the first.
- Terraforming costs spades; your dig level reduces the worker cost.
  Building far from your existing buildings requires shipping or digging
  toward them - reachability is the main constraint on where you can go.
- Passing early gives you the first-player slot next round and a bonus
  tile; passing late gets you more actions this round. The pass bonus
  tile you take matters as much as when you pass.
- Priests are the scarcest resource (7 total, and cult tracks eat them).
  Spending a priest for 1 cult step is usually bad; for 2-3 it is often
  right when it wins a cult majority or unlocks a favor tile.
Play the position in front of you, not a fixed opening."""

_L2_TOOLS = """Before answering you MAY call tools, one per line, to check facts:
  TOOL apply <index>      - show the state after playing move <index>
  TOOL neighbors <hex>    - list hexes adjacent to <hex>
After any tool output you will be asked again. When ready, answer with ONLY
the move number."""


@dataclass
class LLMAgent:
    """An LLM at a given rung of the capability ladder."""

    provider: Provider
    name: str = "llm"
    level: int = 0
    max_tool_calls: int = 3
    fallbacks: int = 0
    tool_calls: int = 0
    _last_valid: int = field(default=0, repr=False)

    # -- prompt construction ---------------------------------------------

    def _system(self, faction: str) -> str:
        blocks = [_L0_SYSTEM]
        if self.level >= 1:
            data = FACTIONS[faction]
            blocks.append(_L1_KNOWLEDGE)
            blocks.append(
                f"You are {faction} ({data.color}). Home terrain {data.color}; "
                f"starting shipping {data.shipping.level}, dig {data.dig.level}."
            )
        if self.level >= 2:
            blocks.append(_L2_TOOLS)
        return "\n\n".join(blocks)

    def _user(self, state: GameState, faction: str, offer: tuple[ParsedCommand, ...]) -> str:
        return (
            f"{serialize_state(state, faction)}\n\n"
            f"Your legal moves:\n{serialize_offer(offer)}\n\n"
            f"Choose one move (0-{len(offer) - 1})."
        )

    # -- answer parsing ---------------------------------------------------

    def _parse_index(self, text: str, n: int) -> int | None:
        """The last whole integer in the reply, range-checked.

        Whole integers only: an earlier version matched at most 3 digits,
        so "9999" silently became 9 and a model that ignored the protocol
        scored as though it had chosen move 9. A malformed answer must
        read as malformed -- it is counted as a fallback and shows up in
        the report, rather than quietly polluting the measurement.
        """
        numbers = re.findall(r"\d+", text)
        if not numbers:
            return None
        value = int(numbers[-1])
        return value if 0 <= value < n else None

    # -- tools (L2) -------------------------------------------------------

    def _run_tool(
        self, line: str, sim: SimState | None, faction: str, offer: tuple[ParsedCommand, ...]
    ) -> str | None:
        match = _TOOL_RE.search(line.strip())
        if match is None:
            return None
        tool, arg = match.group(1).lower(), match.group(2).strip()
        self.tool_calls += 1
        if tool == "neighbors":
            adj = neighbors(arg.upper())
            return f"neighbors of {arg.upper()}: {' '.join(adj) or 'none'}"
        if tool == "apply":
            if sim is None:
                return "apply is unavailable at this decision"
            try:
                index = int(arg)
            except ValueError:
                return f"bad index {arg!r}"
            if not 0 <= index < len(offer):
                return f"index {index} out of range"
            try:
                after = advance(sim, offer[index])
            except (EngineError, ValueError) as exc:
                return f"move {index} could not be applied: {exc}"
            return (
                f"after {describe_move(offer[index])}:\n"
                f"{serialize_state(after.game, faction)}"
            )
        return f"unknown tool {tool!r}"

    # -- Agent protocol ---------------------------------------------------

    def choose(
        self,
        state: GameState,
        faction: str,
        offer: tuple[ParsedCommand, ...],
        rng: random.Random,
    ) -> ParsedCommand:
        return self._decide(state, faction, offer, sim=None)

    def choose_sim(
        self, sim: SimState, faction: str, offer: tuple[ParsedCommand, ...], rng: random.Random
    ) -> ParsedCommand:
        return self._decide(sim.game, faction, offer, sim=sim)

    def _decide(
        self,
        state: GameState,
        faction: str,
        offer: tuple[ParsedCommand, ...],
        sim: SimState | None,
    ) -> ParsedCommand:
        if len(offer) == 1:
            return offer[0]
        messages = [
            Message("system", self._system(faction)),
            Message("user", self._user(state, faction, offer)),
        ]
        for _ in range(self.max_tool_calls + 1 if self.level >= 2 else 1):
            reply = self.provider.complete(messages)
            if self.level >= 2:
                tool_output = self._run_tool(reply, sim, faction, offer)
                if tool_output is not None:
                    messages.append(Message("assistant", reply))
                    messages.append(
                        Message("user", f"{tool_output}\n\nNow answer with ONLY the move number.")
                    )
                    continue
            index = self._parse_index(reply, len(offer))
            if index is not None:
                self._last_valid = index
                return offer[index]
            break
        self.fallbacks += 1
        return offer[min(self._last_valid, len(offer) - 1)]
