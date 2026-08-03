"""Power bowl arithmetic (the three-bowl token cycle).

Rules (reference: resources.pm in jsnell/terra-mystica):
- gain n: tokens move bowl1 -> bowl2 one at a time; once bowl1 is empty,
  bowl2 -> bowl3. Gains beyond capacity are lost.
- burn n: remove n tokens from bowl2 permanently and move n more from
  bowl2 to bowl3 (costs 2 bowl2-tokens per usable power).
- spend n: move n tokens from bowl3 back to bowl1.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Power:
    bowl1: int
    bowl2: int
    bowl3: int

    def __post_init__(self) -> None:
        if self.bowl1 < 0 or self.bowl2 < 0 or self.bowl3 < 0:
            raise ValueError(f"negative power bowl: {self}")

    @property
    def total(self) -> int:
        return self.bowl1 + self.bowl2 + self.bowl3

    @property
    def usable(self) -> int:
        return self.bowl3

    def gainable(self) -> int:
        """Maximum power gain that has any effect."""
        return self.bowl1 * 2 + self.bowl2

    def gain(self, n: int) -> Power:
        if n < 0:
            raise ValueError(f"gain must be >= 0, got {n}")
        from_bowl1 = min(n, self.bowl1)
        from_bowl2 = min(n - from_bowl1, self.bowl2 + from_bowl1)
        return Power(
            bowl1=self.bowl1 - from_bowl1,
            bowl2=self.bowl2 + from_bowl1 - from_bowl2,
            bowl3=self.bowl3 + from_bowl2,
        )

    def burn(self, n: int) -> Power:
        if n < 0 or self.bowl2 < 2 * n:
            raise ValueError(f"cannot burn {n} with bowl2={self.bowl2}")
        return Power(bowl1=self.bowl1, bowl2=self.bowl2 - 2 * n, bowl3=self.bowl3 + n)

    def spend(self, n: int) -> Power:
        if n < 0 or self.bowl3 < n:
            raise ValueError(f"cannot spend {n} with bowl3={self.bowl3}")
        return Power(bowl1=self.bowl1 + n, bowl2=self.bowl2, bowl3=self.bowl3 - n)

    def as_str(self) -> str:
        """Snellman ledger format, e.g. "5/7/0"."""
        return f"{self.bowl1}/{self.bowl2}/{self.bowl3}"

    @classmethod
    def from_str(cls, text: str) -> Power:
        parts = text.split("/")
        if len(parts) != 3:
            raise ValueError(f"bad power string: {text!r}")
        return cls(*(int(p) for p in parts))
