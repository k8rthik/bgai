"""Session configuration: JSON at $BGAI_TM_SESSION, or interactive defaults."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

_ENV_VAR = "BGAI_TM_SESSION"

_FIELD_TYPES: dict[str, type | tuple[type, ...]] = {
    "seed": int,
    "llm_faction_index": int,
    "opponents": str,
    "rungs": list,
    "result_path": str,
    "max_commands": int,
    "factions": list,
}


@dataclass(frozen=True)
class SessionConfig:
    seed: int = 0
    llm_faction_index: int = 0
    """Seat index 0-3 for the external (LLM) seat; -1 = every seat external
    (interactive analysis / corpus-style driving)."""
    opponents: str = "greedy"  # "random" | "greedy"
    rungs: tuple[int, ...] = (1, 2, 3)
    result_path: str | None = None
    max_commands: int = 10000
    factions: tuple[str, str, str, str] | None = None

    def __post_init__(self) -> None:
        if self.llm_faction_index not in (-1, 0, 1, 2, 3):
            raise ValueError(f"llm_faction_index must be -1..3, got {self.llm_faction_index}")
        if self.opponents not in ("random", "greedy"):
            raise ValueError(f"opponents must be 'random' or 'greedy', got {self.opponents!r}")
        if not self.rungs or set(self.rungs) - {1, 2, 3} or 1 not in self.rungs:
            raise ValueError(f"rungs must be a subset of (1,2,3) including 1, got {self.rungs}")
        if self.factions is not None and len(self.factions) != 4:
            raise ValueError(f"factions must name exactly 4 seats, got {self.factions}")


def load_config() -> SessionConfig:
    """SessionConfig from the JSON file named by $BGAI_TM_SESSION (fail fast
    on unknown keys or wrong types), else interactive defaults.
    """
    path_text = os.environ.get(_ENV_VAR)
    if not path_text:
        return SessionConfig()
    raw = json.loads(Path(path_text).read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{path_text}: session config must be a JSON object")
    unknown = set(raw) - set(_FIELD_TYPES)
    if unknown:
        raise ValueError(f"{path_text}: unknown session config keys {sorted(unknown)}")
    for key, expected in _FIELD_TYPES.items():
        if key in raw and raw[key] is not None and not isinstance(raw[key], expected):
            raise ValueError(
                f"{path_text}: config key {key!r} must be {expected}, got {type(raw[key])}"
            )
    if isinstance(raw.get("rungs"), list):
        raw["rungs"] = tuple(raw["rungs"])
    if isinstance(raw.get("factions"), list):
        raw["factions"] = tuple(raw["factions"])
    return SessionConfig(**raw)
