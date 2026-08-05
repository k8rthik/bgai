"""L4 test-time search scaffolds (Phase 7).

Rung L4 spends 10-100x the tokens of L0 on a single move. Two scaffolds,
both provider-agnostic and both usable with or without engine tools:

- **propose -> critique -> pick.** The model nominates N candidate moves
  with reasons, a critic pass tries to *refute* each, and a final pass
  chooses. Independent criticism is what makes best-of-N better than
  temperature sampling: without it, N samples of the same bias just
  reproduce the bias.

- **persistent game plan.** A short scratchpad (opening intent, read on
  each opponent, what changed) carried across the ~150 decisions of a
  game and updated every few turns. Terra Mystica punishes incoherence
  more than it punishes any single weak move; without memory an LLM
  re-derives a strategy every turn and drifts.

Both are deliberately *outside* the model: the state, the legal moves,
and the what-if results all come from the engine. The LLM supplies
judgement, never facts. That split is what keeps the comparison against
the engine AI meaningful (master plan, fairness Class P).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from bgai.data.ledger_parser import ParsedCommand
from bgai.llm.provider import Message, Provider
from bgai.llm.serialize import describe_move

_PROPOSE = """From the legal moves listed, nominate up to {n} that are worth serious
consideration. For each, give the move number and one line on what it is
trying to achieve. Format each line as: <number>: <reason>"""

_CRITIQUE = """You are reviewing a candidate move in Terra Mystica. Argue against it:
what does it cost, what does it give opponents, what does it fail to set up?
If it is genuinely the best available, say so plainly. Be brief."""

_DECIDE = """Given the candidates and the criticism of each, choose the single best move.
Answer with ONLY the move number."""

_PLAN_UPDATE = """Update your game plan in at most 5 short lines. Keep what still holds,
change what the last few turns disproved. Cover: your scoring route, your
main constraint (resources / reachability / cults), and anything an
opponent is doing that threatens it."""


@dataclass
class GamePlan:
    """The per-game scratchpad carried across decisions."""

    text: str = ""
    updated_at_decision: int = -1
    update_every: int = 8

    def due(self, decision_index: int) -> bool:
        """Always due before the first update: the opening is exactly
        where a plan matters most, and a plain interval check would have
        left the first ``update_every`` decisions -- the whole faction
        setup and first round -- played with no plan at all.
        """
        if self.updated_at_decision < 0:
            return True
        return decision_index - self.updated_at_decision >= self.update_every

    def block(self) -> str:
        return f"Your current game plan:\n{self.text}" if self.text else ""


@dataclass
class BestOfNCritic:
    """propose -> critique -> pick, over the engine's legal moves."""

    provider: Provider
    n_candidates: int = 4
    plan: GamePlan = field(default_factory=GamePlan)

    def _parse_candidates(self, reply: str, n_moves: int) -> list[int]:
        out: list[int] = []
        for line in reply.splitlines():
            head = line.strip().split(":", 1)[0].strip()
            digits = "".join(ch for ch in head if ch.isdigit())
            if not digits:
                continue
            index = int(digits)
            if 0 <= index < n_moves and index not in out:
                out.append(index)
        return out[: self.n_candidates]

    def choose_index(
        self,
        state_text: str,
        offer_text: str,
        offer: tuple[ParsedCommand, ...],
        system: str,
    ) -> tuple[int, list[str]]:
        """Returns (chosen index, critique transcript). Falls back to the
        first proposed candidate, then to 0 -- never to an illegal move.
        """
        base = [
            Message("system", system),
            Message("user", f"{self.plan.block()}\n\n{state_text}\n\n{offer_text}"),
        ]
        proposal = self.provider.complete(
            base + [Message("user", _PROPOSE.format(n=self.n_candidates))]
        )
        candidates = self._parse_candidates(proposal, len(offer))
        if not candidates:
            return 0, []

        critiques: list[str] = []
        for index in candidates:
            critique = self.provider.complete(
                base
                + [
                    Message("assistant", proposal),
                    Message(
                        "user",
                        f"{_CRITIQUE}\n\nCandidate: {index}. {describe_move(offer[index])}",
                    ),
                ]
            )
            critiques.append(f"{index}: {critique}")

        verdict = self.provider.complete(
            base
            + [
                Message("assistant", proposal),
                Message("user", "Criticism of each candidate:\n" + "\n".join(critiques)),
                Message("user", _DECIDE),
            ]
        )
        digits = [int(d) for d in "".join(
            ch if ch.isdigit() else " " for ch in verdict
        ).split()]
        for value in reversed(digits):
            if 0 <= value < len(offer):
                return value, critiques
        return candidates[0], critiques

    def maybe_update_plan(
        self, state_text: str, system: str, decision_index: int
    ) -> None:
        if not self.plan.due(decision_index):
            return
        self.plan.text = self.provider.complete(
            [
                Message("system", system),
                Message("user", f"{self.plan.block()}\n\n{state_text}\n\n{_PLAN_UPDATE}"),
            ],
            max_tokens=300,
        ).strip()
        self.plan.updated_at_decision = decision_index
