"""Seeded fresh-game setups for arena/MCP play (LLM-harness plan, task 2)."""

from bgai.arena.setup_factory import fresh_setup
from bgai.engine.tm.factions_data import FACTIONS
from bgai.engine.tm.legal import BLOCKING_PENDING_KINDS


def test_blocking_pending_kinds_is_public():
    assert "leech" in BLOCKING_PENDING_KINDS


def test_fresh_setup_shape():
    s = fresh_setup(seed=42)
    assert s.player_count == 4
    assert len(s.factions) == 4
    assert len(s.score_tiles) == 6
    assert len(s.bonus_tiles) == 7  # player_count + 3
    assert s.game_id == "arena_42"
    colors = [FACTIONS[f].color for f in s.factions]
    assert len(set(colors)) == 4  # one faction per color


def test_fresh_setup_deterministic_and_seed_sensitive():
    assert fresh_setup(seed=1) == fresh_setup(seed=1)
    assert fresh_setup(seed=1) != fresh_setup(seed=2)


def test_fresh_setup_respects_explicit_factions():
    factions = ("nomads", "darklings", "engineers", "mermaids")
    assert fresh_setup(seed=3, factions=factions).factions == factions


def test_fresh_setup_excludes_option_gated_tiles():
    for seed in range(30):
        s = fresh_setup(seed=seed)
        assert "BON10" not in s.bonus_tiles  # shipping-bonus option
        assert all(t is not None for t in s.score_tiles)
