"""Headless LLM arena: spawn `claude -p` (Claude Code CLI) per game against
the repo's TM MCP server, collect results, aggregate win rate / VP / cost.

Usage:
    uv run python scripts/arena_llm.py --games 4 --seed 100 \
        --opponents heuristic --rungs 1,2,3 --out runs/llm_r123/ [--dry-run]

Per game g: writes <out>/game_<g>/{session.json,mcp.json}, spawns claude
with the pilot prompt, then reads back result.json (written by the MCP
session at game end). A game whose result never appears is retried up to
2 times with the same seed, then recorded invalid and excluded from stats.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PROMPT_PATH = REPO_ROOT / "prompts" / "tm_pilot.md"
DEFAULT_COMPENDIUM_DIR = REPO_ROOT / "docs" / "knowledge" / "tm-compendium"
RETRIES = 2
GAME_TIMEOUT_S = 3600

_RUNG2_TEXT = (
    "## Factual analysis tools (enabled)\n\n"
    "`preview_move` shows a move's exact deltas without committing;\n"
    "`score_projection` shows mechanically guaranteed scoring arithmetic.\n"
    "Use them for the arithmetic -- the judgment is yours."
)
_RUNG3_TEXT = (
    "## Branch tools (enabled)\n\n"
    "`branch`/`branch_play_move`/`branch_state`/`discard_branch` fork the\n"
    "game into scratch copies for lookahead. You play ALL seats in a\n"
    "branch -- model the opponents' likely replies yourself, read the\n"
    "resulting factual states, and judge the lines with your own reasoning."
)


def pilot_prompt(rungs: tuple[int, ...], compendium_dir: Path | None = None) -> str:
    text = PROMPT_PATH.read_text()
    text = text.replace("{{RUNG2}}", _RUNG2_TEXT if 2 in rungs else "")
    text = text.replace("{{RUNG3}}", _RUNG3_TEXT if 3 in rungs else "")
    compendium = ""
    directory = compendium_dir or DEFAULT_COMPENDIUM_DIR
    if 4 in rungs and directory.is_dir():
        playbooks = sorted(directory.glob("*-playbook.md"))
        if playbooks:
            compendium = "## Strategy compendium (corpus-derived principles)\n\n" + "\n\n".join(
                p.read_text() for p in playbooks
            )
    text = text.replace("{{COMPENDIUM}}", compendium)
    return "\n".join(line for line in text.splitlines() if line is not None).strip() + "\n"


def build_session_config(seed: int, game_index: int, opponents: str, rungs: tuple[int, ...],
                         result_path: Path) -> dict:
    return {
        "seed": seed,
        "llm_faction_index": game_index % 4,
        "opponents": opponents,
        "rungs": [r for r in rungs if r in (1, 2, 3)],  # rung 4 is prompt-side only
        "result_path": str(result_path),
    }


def build_mcp_config(session_path: Path) -> dict:
    return {
        "mcpServers": {
            "tm": {
                "command": "uv",
                "args": [
                    "run", "--directory", str(REPO_ROOT), "python", "-m", "bgai.mcp.server",
                ],
                "env": {"BGAI_TM_SESSION": str(session_path)},
            }
        }
    }


def spawn_game(game_dir: Path, prompt: str, args: argparse.Namespace) -> dict | None:
    """Run one headless claude game; returns the claude CLI's JSON output
    (or None on timeout/crash). The game result itself lands in
    result.json via the MCP session."""
    command = [
        args.claude_bin, "-p", prompt,
        "--mcp-config", str(game_dir / "mcp.json"),
        "--allowedTools", "mcp__tm__*",
        # Measurement integrity: the pilot may use ONLY the tm tools -- the
        # rung-1 smoke run showed a pilot probing the repo's engine source
        # with Bash/Write when left in the repo cwd with default tools.
        "--disallowedTools", "Bash,Read,Write,Edit,Glob,Grep,WebFetch,WebSearch,Task,Agent",
        "--permission-mode", "default",  # a plan-mode user default would block the MCP tools
        "--max-turns", "600",
        "--output-format", "json",
    ]
    if args.model:
        command += ["--model", args.model]
    try:
        proc = subprocess.run(
            command, capture_output=True, text=True, timeout=GAME_TIMEOUT_S,
            cwd=game_dir,  # not the repo: the pilot gets no incidental repo access
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        (game_dir / "transcript.txt").write_text(f"spawn failed: {exc}")
        return None
    (game_dir / "transcript.txt").write_text(proc.stderr or "")
    (game_dir / "claude.json").write_text(proc.stdout or "")
    return _result_record(proc.stdout)


def _result_record(stdout: str) -> dict | None:
    """The claude CLI's summary record: --output-format json emits either a
    single object or (verbose variants) a list whose 'type'=='result'
    element carries total_cost_usd/duration_ms."""
    try:
        payload = json.loads(stdout)
    except (json.JSONDecodeError, TypeError):
        return None
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, list):
        for item in reversed(payload):
            if isinstance(item, dict) and item.get("type") == "result":
                return item
    return None


def _run_one_game(g: int, args: argparse.Namespace, out: Path, prompt: str) -> dict:
    game_dir = out / f"game_{g}"
    game_dir.mkdir(parents=True, exist_ok=True)
    result_path = game_dir / "result.json"
    session_path = game_dir / "session.json"
    session_path.write_text(
        json.dumps(
            build_session_config(args.seed + g, g, args.opponents, args.rungs, result_path),
            indent=2,
        )
    )
    (game_dir / "mcp.json").write_text(json.dumps(build_mcp_config(session_path), indent=2))
    if args.dry_run:
        return {"status": "dry-run"}

    for attempt in range(1 + RETRIES):
        if result_path.exists():
            result_path.unlink()
        started = time.monotonic()
        claude_meta = spawn_game(game_dir, prompt, args)
        duration = time.monotonic() - started
        if result_path.exists():
            try:
                payload = json.loads(result_path.read_text())
            except json.JSONDecodeError:
                continue
            payload["status"] = "ok"
            payload["duration_s"] = duration
            payload["cost_usd"] = (claude_meta or {}).get("total_cost_usd")
            return payload
        print(f"  game {g} attempt {attempt + 1}: no result.json, retrying")
    return {"status": "invalid"}


def aggregate(results: list[dict]) -> dict:
    valid = [r for r in results if r.get("status") == "ok"]
    invalid = [r for r in results if r.get("status") != "ok"]
    summary: dict = {
        "games_valid": len(valid),
        "games_invalid": len(invalid),
    }
    if valid:
        wins = sum(1 for r in valid if r["llm_faction"] in r["winners"])
        vps = [r["vp"][r["llm_faction"]] for r in valid]
        margins = [
            r["vp"][r["llm_faction"]]
            - max(v for f, v in r["vp"].items() if f != r["llm_faction"])
            for r in valid
        ]
        costs = [r["cost_usd"] for r in valid if r.get("cost_usd") is not None]
        durations = [r["duration_s"] for r in valid if r.get("duration_s") is not None]
        summary.update(
            {
                "llm_win_rate": wins / len(valid),
                "llm_mean_vp": sum(vps) / len(vps),
                "llm_mean_margin_vs_best_opponent": sum(margins) / len(margins),
                "mean_commands": sum(r.get("commands_used", 0) for r in valid) / len(valid),
                "mean_cost_usd": sum(costs) / len(costs) if costs else None,
                "mean_duration_s": sum(durations) / len(durations) if durations else None,
            }
        )
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--opponents", choices=("random", "heuristic"), default="heuristic")
    parser.add_argument("--rungs", default="1,2,3",
                        help="comma-separated subset of 1,2,3,4 (4 = compendium in prompt)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--claude-bin", default="claude")
    parser.add_argument("--dry-run", action="store_true",
                        help="write configs and print commands without spawning claude")
    args = parser.parse_args(argv)
    args.rungs = tuple(sorted(int(r) for r in str(args.rungs).split(",") if r.strip()))

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    prompt = pilot_prompt(args.rungs)

    results = []
    for g in range(args.games):
        print(f"game {g}: seed {args.seed + g}, llm seat {g % 4}, rungs {args.rungs}")
        results.append(_run_one_game(g, args, out, prompt))

    if args.dry_run:
        print(f"dry run: configs written under {out}")
        return
    summary = aggregate(results)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
