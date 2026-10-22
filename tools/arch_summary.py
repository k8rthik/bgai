"""Summarise arch-sweep runs: final held-out metrics + the val curve.

    uv run python tools/arch_summary.py data/checkpoints/arch
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

KEYS = ("top1", "top3", "pairwise", "winner_top1", "policy", "value")


def val_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [r for r in rows if "top1" in r and "pairwise" in r]


def main(root: str) -> None:
    runs = sorted(p for p in Path(root).iterdir() if (p / "metrics.jsonl").exists())
    print(f"{'run':28s} {'step':>6s} " + " ".join(f"{k:>11s}" for k in KEYS) + "  top1 curve")
    for run in runs:
        rows = val_rows(run / "metrics.jsonl")
        if not rows:
            continue
        last = rows[-1]
        curve = " ".join(f"{r['top1'] * 100:.1f}" for r in rows)
        print(f"{run.name:28s} {last['step']:6d} "
              + " ".join(f"{last[k]:11.4f}" for k in KEYS) + f"  {curve}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/checkpoints/arch")
