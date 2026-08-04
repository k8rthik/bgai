"""Parser for snellman ledger command text → typed move records.

Grammar reference: ``usage.org`` in jsnell/terra-mystica plus vocabulary
observed in real tournament ledgers. Commands arrive dot-separated within a
ledger row (``"convert 1P to 1W. burn 3. action ACT6. dig 1. build F7"``).

Design: one compiled regex per verb family, dispatched from a single table.
``parse_command`` returns ``None`` for anything unknown — callers collect
unknowns and fail loudly at corpus level rather than silently skipping.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum


class Kind(Enum):
    DECISION = "decision"        # a player choice (trainable)
    INCOME = "income"            # engine-generated income bookkeeping
    SCORING = "scoring"          # engine-generated scoring rows
    BOOKKEEPING = "bookkeeping"  # other engine-generated state changes
    ANNOTATION = "annotation"    # informational bracket rows


@dataclass(frozen=True)
class ParsedCommand:
    verb: str
    kind: Kind
    raw: str
    loc: str | None = None       # map hex, uppercased (e.g. "G5")
    loc2: str | None = None      # bridge second hex
    building: str | None = None  # TP/TE/SH/SA
    tile: str | None = None      # BONx/FAVx/TWx/ACTx/ACTS
    cult: str | None = None      # FIRE/WATER/EARTH/AIR
    color: str | None = None     # transform target color
    target: str | None = None    # other faction (leech/decline)
    reason: str | None = None    # scoring reason (cult name / "network" / ...)
    res1: str | None = None      # convert: from-resource
    res2: str | None = None      # convert: to-resource
    n1: int | None = None
    n2: int | None = None


_CULTS = r"(?:FIRE|WATER|EARTH|AIR)"

_BUILDING_NAMES = {
    "tp": "TP", "te": "TE", "sh": "SH", "sa": "SA",
    "trading post": "TP", "temple": "TE", "stronghold": "SH", "sanctuary": "SA",
}

_RESOURCE_NAMES = {
    "pw": "PW", "power": "PW", "vp": "VP", "c": "C", "w": "W", "p": "P",
    "coin": "C", "coins": "C", "worker": "W", "workers": "W",
    "priest": "P", "priests": "P",
}

_RES = r"(?:PW|VP|[CWP]|power|coins?|workers?|priests?)"

_Maker = Callable[[re.Match, str], ParsedCommand]


def _normalize_hex_ref(text: str) -> str:
    """Land hexes are keyed uppercase ("A1".."I13"); river hexes are keyed
    lowercase ("r0".."r35", ``board.py``'s own docstring). Every other
    location-bearing rule here only ever matches land hexes, so a blanket
    ``.upper()`` was correct for them -- but ``connect``'s grammar accepts
    a bare river-hex reference too (``r\\d+``), case-insensitively per
    ``re.IGNORECASE``, and blindly uppercasing that produced "R1" instead
    of the board's actual "r1" key, a lookup miss (task-13 report,
    reference-game row 208: ``connect r1`` -- "unknown hex 'R1'").
    """
    return text.lower() if text[:1] in ("r", "R") and text[1:].isdigit() else text.upper()


def _rule(pattern: str, maker: _Maker) -> tuple[re.Pattern, _Maker]:
    return re.compile(pattern, re.IGNORECASE), maker


_RULES: list[tuple[re.Pattern, _Maker]] = [
    _rule(
        r"^build\s+([a-z]\d+)$",
        lambda m, raw: ParsedCommand("build", Kind.DECISION, raw, loc=m[1].upper()),
    ),
    _rule(
        r"^upgrade\s+([a-z]\d+)\s+to\s+"
        r"(TP|TE|SH|SA|trading\s+post|temple|stronghold|sanctuary)$",
        lambda m, raw: ParsedCommand(
            "upgrade", Kind.DECISION, raw, loc=m[1].upper(),
            building=_BUILDING_NAMES[" ".join(m[2].lower().split())],
        ),
    ),
    _rule(
        r"^action\s+(\w+)$",
        lambda m, raw: ParsedCommand("action", Kind.DECISION, raw, tile=m[1].upper()),
    ),
    _rule(
        r"^pass(?:\s+(\w+))?$",
        lambda m, raw: ParsedCommand(
            "pass", Kind.DECISION, raw, tile=m[1].upper() if m[1] else None
        ),
    ),
    _rule(
        # Source faction is optional in early-era logs ("leech 2"); bare
        # "decline" declines the outstanding offer.
        r"^(leech|decline)(?:\s+(\d+))?(?:\s+from\s+(\w+))?$",
        lambda m, raw: ParsedCommand(
            m[1].lower(), Kind.DECISION, raw,
            n1=int(m[2]) if m[2] else None,
            target=m[3].lower() if m[3] else None,
        ),
    ),
    _rule(
        # Amounts default to 1 ("convert pw to c"); VP is a resource for
        # Alchemists; early logs spell resources out ("Convert 1 power to 1c").
        rf"^convert\s+(\d+)?\s*({_RES})\s+to\s+(\d+)?\s*({_RES})$",
        lambda m, raw: ParsedCommand(
            "convert", Kind.DECISION, raw,
            n1=int(m[1]) if m[1] else 1, res1=_RESOURCE_NAMES[m[2].lower()],
            n2=int(m[3]) if m[3] else 1, res2=_RESOURCE_NAMES[m[4].lower()],
        ),
    ),
    _rule(
        r"^burn\s+(\d+)$",
        lambda m, raw: ParsedCommand("burn", Kind.DECISION, raw, n1=int(m[1])),
    ),
    _rule(
        r"^dig\s+(\d+)(?:\s+([a-z]\d+))?$",  # rare "DIG 1 I8" carries a location
        lambda m, raw: ParsedCommand(
            "dig", Kind.DECISION, raw,
            n1=int(m[1]), loc=m[2].upper() if m[2] else None,
        ),
    ),
    _rule(
        rf"^send\s+(?:p|priest)\s+to\s+({_CULTS})(?:\s+for\s+(\d+))?$",
        lambda m, raw: ParsedCommand(
            "send", Kind.DECISION, raw,
            cult=m[1].upper(), n1=int(m[2]) if m[2] else None,
        ),
    ),
    _rule(
        r"^transform\s+([a-z]\d+)(?:\s+to\s+(\w+))?$",
        lambda m, raw: ParsedCommand(
            "transform", Kind.DECISION, raw,
            loc=m[1].upper(), color=m[2].lower() if m[2] else None,
        ),
    ),
    _rule(
        r"^bridge\s+([a-z]\d+):([a-z]\d+)$",
        lambda m, raw: ParsedCommand(
            "bridge", Kind.DECISION, raw, loc=m[1].upper(), loc2=m[2].upper()
        ),
    ),
    _rule(
        # Optional explicit target level in early logs: "advance ship to 1".
        r"^advance\s+(ship|shipping|dig|digging)(?:\s+(?:to\s+)?(\d+))?$",
        lambda m, raw: ParsedCommand(
            "advance", Kind.DECISION, raw,
            reason={"shipping": "ship", "digging": "dig"}.get(m[1].lower(), m[1].lower()),
            n1=int(m[2]) if m[2] else None,
        ),
    ),
    _rule(
        # Mermaids town connection: single river hex, or an early-era hex pair.
        r"^connect\s+([a-z]\d+|r\d+)(?::([a-z]\d+|r\d+))?$",
        lambda m, raw: ParsedCommand(
            "connect", Kind.DECISION, raw,
            loc=_normalize_hex_ref(m[1]), loc2=_normalize_hex_ref(m[2]) if m[2] else None,
        ),
    ),
    _rule(
        r"^(wait|done|resign)$",
        lambda m, raw: ParsedCommand(m[1].lower(), Kind.DECISION, raw),
    ),
    _rule(
        r"^\+(FAV\d+)$",
        lambda m, raw: ParsedCommand("gain_favor", Kind.DECISION, raw, tile=m[1].upper()),
    ),
    _rule(
        r"^\+(\d+)?(TW\d+)$",
        lambda m, raw: ParsedCommand(
            "gain_town", Kind.DECISION, raw,
            tile=m[2].upper(), n1=int(m[1]) if m[1] else 1,
        ),
    ),
    _rule(
        rf"^\+(\d+)?({_CULTS})$",
        lambda m, raw: ParsedCommand(
            "gain_cult", Kind.BOOKKEEPING, raw,
            cult=m[2].upper(), n1=int(m[1]) if m[1] else 1,
        ),
    ),
    _rule(
        rf"^-(\d+)?({_CULTS})$",
        lambda m, raw: ParsedCommand(
            "lose_cult", Kind.BOOKKEEPING, raw,
            cult=m[2].upper(), n1=int(m[1]) if m[1] else 1,
        ),
    ),
    _rule(
        # Rare engine bookkeeping marker, e.g. "-3CONVERT_W_TO_P" (Darklings SH).
        r"^[+-](\d+)?CONVERT_(PW|VP|[CWP])_TO_(PW|VP|[CWP])$",
        lambda m, raw: ParsedCommand(
            "convert_marker", Kind.BOOKKEEPING, raw,
            n1=int(m[1]) if m[1] else 1, res1=m[2].upper(), res2=m[3].upper(),
        ),
    ),
    _rule(
        r"^\+(\d+)vp\s+for\s+(\w+)$",
        lambda m, raw: ParsedCommand(
            "score_vp", Kind.SCORING, raw, n1=int(m[1]), reason=m[2].upper()
        ),
    ),
    _rule(
        r"^(other_income_for_faction|cult_income_for_faction|all_income_for_faction)$",
        lambda m, raw: ParsedCommand(m[1].lower(), Kind.INCOME, raw),
    ),
    _rule(
        r"^-(\d+)?spade$",
        lambda m, raw: ParsedCommand(
            "lose_spade", Kind.BOOKKEEPING, raw, n1=int(m[1]) if m[1] else 1
        ),
    ),
    _rule(
        r"^-(FREE_D|FREE_TP|FREE_TF|BRIDGE)$",
        lambda m, raw: ParsedCommand(
            "lose_marker", Kind.BOOKKEEPING, raw, reason=m[1].upper()
        ),
    ),
    _rule(
        rf"^-(\d+)\s*({_RES})$",
        lambda m, raw: ParsedCommand(
            "lose_resource", Kind.BOOKKEEPING, raw,
            n1=int(m[1]), res1=_RESOURCE_NAMES[m[2].lower()],
        ),
    ),
    _rule(
        r"^score_resources$",
        lambda m, raw: ParsedCommand("score_resources", Kind.SCORING, raw),
    ),
    _rule(
        r"^setup$",
        lambda m, raw: ParsedCommand("setup", Kind.BOOKKEEPING, raw),
    ),
    _rule(
        # Cultists' (Game/Factions/Cultists.pm `leech_effect.not_taken`)
        # "all opponents declined" bonus: `commands.pm`'s
        # `cultist_maybe_gain_power` (558-586) logs this via
        # `$ledger->add_row_for_effect` -- an *unbuffered*, immediate
        # ledger-array push that lands **before** the resolving
        # `decline`/capped-`leech` row's own (buffered, `finish_row`-flushed)
        # summary row in final ledger order, even though it fires
        # synchronously *during* that row's command processing
        # (`leech.py`'s module docstring has the full ordering citation).
        # Parsed as its own verb (not a generic ``annotation``) so
        # ``leech.py`` can grant the faction's ``leech_effect.not_taken``
        # gain directly off this row instead of trying to infer it from the
        # *following* row's decline command, which would land the state
        # change one row late relative to this checkpoint.
        r"^\[all opponents declined power\]$",
        lambda m, raw: ParsedCommand("cultist_leech_bonus", Kind.BOOKKEEPING, raw),
    ),
    _rule(
        r"^\[.*\]$",
        lambda m, raw: ParsedCommand("annotation", Kind.ANNOTATION, raw),
    ),
]


def split_commands(commands: str) -> list[str]:
    """Split a ledger row's dot-separated command string into single commands."""
    parts = (p.strip() for p in commands.split(". "))
    return [p.rstrip(".").strip() for p in parts if p.rstrip(".").strip()]


def parse_command(text: str) -> ParsedCommand | None:
    """Parse one command; None if the vocabulary doesn't cover it."""
    stripped = text.strip()
    for pattern, maker in _RULES:
        match = pattern.match(stripped)
        if match is not None:
            return maker(match, stripped)
    return None


def parse_commands(commands: str) -> tuple[list[ParsedCommand], list[str]]:
    """Parse a full ledger row command string; returns (moves, unknown_commands)."""
    moves: list[ParsedCommand] = []
    unknown: list[str] = []
    for part in split_commands(commands):
        parsed = parse_command(part)
        if parsed is None:
            unknown.append(part)
        else:
            moves.append(parsed)
    return moves, unknown
