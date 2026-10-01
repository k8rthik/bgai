"""TM-Env configuration validation (the env's outermost input boundary)."""

from __future__ import annotations

import pytest

from bgai.env.config import (
    ILLEGAL_ACTION_REWARD,
    MAX_CANDIDATES,
    MAX_PLAYERS,
    MIN_PLAYERS,
    EnvConfig,
    IllegalActionPolicy,
    RewardMode,
    SetupSource,
)


def test_defaults_are_a_four_player_terminal_share_table() -> None:
    config = EnvConfig()
    assert config.player_count == 4
    assert config.reward_mode is RewardMode.TERMINAL_VP_SHARE
    assert config.setup_source is SetupSource.SYNTHETIC
    assert config.illegal_action is IllegalActionPolicy.TERMINATE
    assert config.max_candidates == MAX_CANDIDATES
    assert config.factions is None


@pytest.mark.parametrize("count", range(MIN_PLAYERS, MAX_PLAYERS + 1))
def test_every_supported_player_count_constructs(count: int) -> None:
    assert EnvConfig(player_count=count).player_count == count


@pytest.mark.parametrize("count", [0, 1, 6, -1])
def test_unsupported_player_counts_are_rejected(count: int) -> None:
    with pytest.raises(ValueError, match="player_count must be in"):
        EnvConfig(player_count=count)


def test_config_is_frozen() -> None:
    config = EnvConfig()
    with pytest.raises(Exception):  # FrozenInstanceError
        config.player_count = 2  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("reward_mode", "terminal_vp_share", "reward_mode must be a RewardMode"),
        ("setup_source", "synthetic", "setup_source must be a SetupSource"),
        ("illegal_action", "raise", "illegal_action must be an IllegalActionPolicy"),
    ],
)
def test_enum_fields_refuse_bare_strings(field: str, value: str, match: str) -> None:
    """A bare string would compare equal to the StrEnum member but would not
    carry the enum's identity, and the env dispatches with ``is``.
    """
    with pytest.raises(TypeError, match=match):
        EnvConfig(**{field: value})  # type: ignore[arg-type]


@pytest.mark.parametrize(("field", "value"), [("max_decisions", 0), ("max_candidates", 0)])
def test_positive_budgets_required(field: str, value: int) -> None:
    with pytest.raises(ValueError, match=f"{field} must be >= 1"):
        EnvConfig(**{field: value})


def test_corpus_setups_are_four_player_only() -> None:
    with pytest.raises(ValueError, match="4-player only"):
        EnvConfig(player_count=3, setup_source=SetupSource.CORPUS)
    # ... and fine at four
    assert EnvConfig(player_count=4, setup_source=SetupSource.CORPUS).player_count == 4


def test_explicit_lineup_must_match_player_count() -> None:
    with pytest.raises(ValueError, match="player_count is 4"):
        EnvConfig(player_count=4, factions=("witches", "nomads"))


def test_explicit_lineup_rejects_unknown_faction() -> None:
    with pytest.raises(ValueError, match="unknown factions"):
        EnvConfig(player_count=2, factions=("witches", "not_a_faction"))


def test_explicit_lineup_rejects_duplicates() -> None:
    with pytest.raises(ValueError, match="duplicate factions"):
        EnvConfig(player_count=2, factions=("witches", "witches"))


def test_explicit_lineup_rejects_colour_clash() -> None:
    """Auren and Witches are both green: an illegal Terra Mystica lineup."""
    with pytest.raises(ValueError, match="share a colour"):
        EnvConfig(player_count=2, factions=("auren", "witches"))


def test_legal_lineup_accepted() -> None:
    config = EnvConfig(player_count=2, factions=("witches", "nomads"))
    assert config.factions == ("witches", "nomads")


def test_illegal_action_reward_is_not_better_than_losing() -> None:
    """Rank mode bottoms out at -1.0; illegal play must not beat that."""
    assert ILLEGAL_ACTION_REWARD <= -1.0
