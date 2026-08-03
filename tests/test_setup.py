"""Game options + per-game setup loader tests."""

from bgai.engine.tm.setup import GameOptions, load_setup


def test_options_from_csv() -> None:
    opts = GameOptions.from_csv(
        "email-notify,errata-cultist-power,strict-leech,mini-expansion-1"
    )
    assert opts.errata_cultist_power and opts.strict_leech and opts.mini_expansion_1
    assert not opts.variable_turn_order


def test_load_reference_game() -> None:
    s = load_setup("4pLeague_S10_D1L1_G1")
    assert s.player_count == 4
    assert s.factions == ("engineers", "darklings", "nomads", "mermaids")
    assert len(s.score_tiles) == 6
    assert s.score_tiles[0].cult == "WATER" and s.score_tiles[0].req == 4
    assert set(s.bonus_tiles) == {"BON1", "BON3", "BON4", "BON5", "BON7", "BON9", "BON10"}
    assert s.options.strict_leech
