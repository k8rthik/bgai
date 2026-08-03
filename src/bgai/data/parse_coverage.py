"""Report parser coverage over all crawled game logs.

Run: ``python -m bgai.data.parse_coverage [raw_games_dir]``
Prints unknown-command counts (empty output section = full coverage).
"""

from __future__ import annotations

import gzip
import sys
from collections import Counter
from pathlib import Path

import orjson

from bgai.data.ledger_parser import parse_commands


def check_coverage(raw_dir: Path) -> Counter:
    unknown: Counter = Counter()
    n_games = n_rows = n_moves = 0
    for path in sorted(raw_dir.glob("*.json.gz")):
        game = orjson.loads(gzip.decompress(path.read_bytes()))
        n_games += 1
        for row in game.get("ledger", []):
            commands = row.get("commands")
            if not commands:
                continue
            n_rows += 1
            moves, unknown_cmds = parse_commands(commands)
            n_moves += len(moves)
            for u in unknown_cmds:
                unknown[u] += 1
    print(f"games: {n_games}, command rows: {n_rows}, parsed moves: {n_moves}")
    print(f"unknown commands: {sum(unknown.values())} occurrences, {len(unknown)} distinct")
    for text, count in unknown.most_common():
        print(f"  {count:6d}  {text!r}")
    return unknown


if __name__ == "__main__":
    raw_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/raw/games")
    check_coverage(raw_dir)
