"""Generate the compendium statistics tables:
`uv run python -m bgai.knowledge --out docs/knowledge/tm-compendium/tables/`."""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from bgai.knowledge.stats import write_compendium_tables


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Mine win-conditioned TM corpus statistics.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--moves", type=Path, default=Path("data/datasets/moves.parquet"))
    parser.add_argument("--deltas", type=Path, default=Path("data/datasets/deltas.parquet"))
    args = parser.parse_args(argv)

    moves = pl.read_parquet(args.moves)
    deltas = pl.read_parquet(args.deltas)
    write_compendium_tables(moves, deltas, args.out)
    print(f"tables written to {args.out}")


if __name__ == "__main__":
    main()
