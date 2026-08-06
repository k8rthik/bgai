# Generating the rung-4 strategy playbooks

The compendium has two layers:

1. **Statistics tables** (`tables/`, generated, committed):
   `uv run python -m bgai.knowledge --out docs/knowledge/tm-compendium/tables/`
   mines the full corpus into win-conditioned tables per faction. Numbers
   only — no positions, no moves.
2. **Prose playbooks** (`<faction>-playbook.md`, LLM-written, human-reviewed):
   principles with effect sizes, derived from the tables *alone*. These are
   what the headless runner substitutes into the pilot prompt when `--rungs`
   includes 4.

## The playbook prompt

Run in Claude Code, once per faction:

> From the win-conditioned statistics tables in
> `docs/knowledge/tm-compendium/tables/<faction>.md` and `_overview.md`
> ALONE, write a playbook of strategic principles for playing <faction> in
> 4-player Terra Mystica. Requirements:
> - ≤600 words, markdown, titled "# <faction> playbook".
> - Every claim must cite a number from the tables (e.g. "winners average
>   5.2 priests sent by round 4; losers 3.1").
> - Principles with effect sizes only — no move prescriptions, no board
>   positions, no hex references, no invented facts.
> - Save to `docs/knowledge/tm-compendium/<faction>-playbook.md`.

## Review checklist (before committing a playbook)

- [ ] Every quantitative claim traceable to a table cell.
- [ ] No board positions, hexes, or concrete move sequences.
- [ ] No facts that are not in the tables (no outside TM theory).
- [ ] ≤600 words.

## Why this boundary

Position-level retrieval and outside theory were rejected by design (spec,
2026-08-04): the compendium teaches *what expert games show*, with effect
sizes, and the pilot applies the principles with its own judgment. Keeping
the prose generation table-grounded and reviewed preserves that.
