"""Arena match runner, rotation, and reports (LLM-harness plan, task 6)."""

import pytest

from bgai.arena.driver import GameResult
from bgai.arena.match import MatchOutcome, MatchSpec, build_agent, rotation, run_match
from bgai.arena.report import format_table, summarize, to_json


def test_rotation_is_seat_fair():
    names = ("random", "random2", "random3", "heuristic")
    specs = rotation(names, games=8, base_seed=50)
    assert len(specs) == 8
    assert [s.seed for s in specs] == list(range(50, 58))
    for name in names:
        for seat in range(4):
            count = sum(1 for s in specs if s.agent_names[seat] == name)
            assert count == 2  # 8 games / 4 seats


def test_build_agent_registry():
    assert build_agent("random", seed=1).name == "random"
    assert build_agent("heuristic", seed=1).name == "heuristic"
    with pytest.raises(ValueError, match="unknown agent"):
        build_agent("stockfish", seed=1)


def _fake_outcome(seed: int, agent_names, vps) -> MatchOutcome:
    factions = ("nomads", "darklings", "engineers", "mermaids")
    vp = dict(zip(factions, vps, strict=True))
    top = max(vp.values())
    winners = tuple(f for f in factions if vp[f] == top)
    result = GameResult(
        game_id=f"arena_{seed}", vp=vp, winners=winners, commands_applied=100
    )
    return MatchOutcome(
        spec=MatchSpec(seed=seed, agent_names=tuple(agent_names)),
        result=result,
        agent_by_faction=dict(zip(factions, agent_names, strict=True)),
    )


def test_summarize_win_rate_and_mean_vp():
    outcomes = [
        _fake_outcome(0, ("a", "b", "b", "b"), (90, 40, 50, 60)),
        _fake_outcome(1, ("a", "b", "b", "b"), (80, 40, 50, 60)),
        _fake_outcome(2, ("b", "a", "b", "b"), (30, 100, 50, 60)),
        _fake_outcome(3, ("b", "a", "b", "b"), (30, 20, 50, 60)),  # a loses this one
    ]
    report = summarize(outcomes)
    a = report.agents["a"]
    assert a.games == 4
    assert a.wins == 3
    assert a.win_rate == pytest.approx(0.75)
    assert a.mean_vp == pytest.approx((90 + 80 + 100 + 20) / 4)
    b = report.agents["b"]
    assert b.games == 4  # per-game participation, not per-seat
    assert b.wins == 1


def test_dominant_agent_gets_higher_trueskill():
    outcomes = [
        _fake_outcome(g, ("a", "b", "b", "b"), (90, 40, 50, 60)) for g in range(6)
    ]
    report = summarize(outcomes)
    assert report.agents["a"].trueskill_mu > report.agents["b"].trueskill_mu


def test_report_serialization_roundtrip():
    outcomes = [_fake_outcome(0, ("a", "b", "b", "b"), (90, 40, 50, 60))]
    report = summarize(outcomes)
    text = to_json(report)
    assert '"win_rate"' in text
    table = format_table(report)
    assert "a" in table and "win" in table.lower()


def test_run_match_real_short():
    spec = MatchSpec(seed=77, agent_names=("random", "random", "random", "heuristic"))
    outcome = run_match(spec)
    assert set(outcome.agent_by_faction.values()) == {"random", "heuristic"}
    assert set(outcome.result.vp) == set(outcome.agent_by_faction)
