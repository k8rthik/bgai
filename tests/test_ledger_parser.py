"""Tests for the snellman ledger command parser (network-free, real command examples)."""

import pytest

from bgai.data.ledger_parser import Kind, parse_command, parse_commands, split_commands


def cmd(text: str):
    parsed = parse_command(text)
    assert parsed is not None, f"failed to parse: {text!r}"
    return parsed


def test_split_commands_dot_separated() -> None:
    assert split_commands("convert 1P to 1W. burn 3. action ACT6. dig 1. build F7") == [
        "convert 1P to 1W",
        "burn 3",
        "action ACT6",
        "dig 1",
        "build F7",
    ]


def test_build_and_upgrade() -> None:
    move = cmd("build G5")
    assert (move.verb, move.loc, move.kind) == ("build", "G5", Kind.DECISION)
    move = cmd("upgrade G5 to TP")
    assert (move.verb, move.loc, move.building) == ("upgrade", "G5", "TP")


def test_case_insensitive_verbs() -> None:
    assert cmd("Pass BON1").verb == "pass"
    assert cmd("Pass BON1").tile == "BON1"
    assert cmd("Leech 1 from darklings").target == "darklings"
    assert cmd("Bridge G4:H5").loc2 == "H5"


def test_leech_decline_amounts() -> None:
    leech = cmd("Leech 3 from witches")
    assert (leech.verb, leech.n1, leech.target) == ("leech", 3, "witches")
    decline = cmd("Decline 4 from witches")
    assert (decline.verb, decline.n1, decline.target) == ("decline", 4, "witches")


def test_convert_resources() -> None:
    move = cmd("convert 3PW to 1W")
    assert (move.n1, move.res1, move.n2, move.res2) == (3, "PW", 1, "W")
    move = cmd("convert 1P to 1W")
    assert (move.n1, move.res1, move.n2, move.res2) == (1, "P", 1, "W")


def test_send_priest_to_cult() -> None:
    move = cmd("send p to EARTH")
    assert (move.verb, move.cult) == ("send", "EARTH")
    move = cmd("send p to WATER for 2")
    assert (move.verb, move.cult, move.n1) == ("send", "WATER", 2)


def test_transform_dig_burn_advance_action() -> None:
    assert cmd("transform G3 to gray").color == "gray"
    assert cmd("dig 1").n1 == 1
    assert cmd("burn 2").n1 == 2
    assert cmd("advance ship").verb == "advance"
    assert cmd("action ACT6").tile == "ACT6"
    assert cmd("action FAV6").tile == "FAV6"


def test_engine_generated_rows() -> None:
    assert cmd("other_income_for_faction").kind == Kind.INCOME
    assert cmd("cult_income_for_faction").kind == Kind.INCOME
    assert cmd("score_resources").kind == Kind.SCORING
    vp = cmd("+8vp for FIRE")
    assert (vp.kind, vp.n1, vp.reason) == (Kind.SCORING, 8, "FIRE")
    assert cmd("+18vp for network").n1 == 18


def test_gain_rows_are_decisions_for_tiles_and_bookkeeping_for_cults() -> None:
    fav = cmd("+FAV11")
    assert (fav.verb, fav.tile, fav.kind) == ("gain_favor", "FAV11", Kind.DECISION)
    town = cmd("+TW7")
    assert (town.verb, town.tile, town.kind) == ("gain_town", "TW7", Kind.DECISION)
    cult = cmd("+AIR")
    assert (cult.verb, cult.cult, cult.kind) == ("gain_cult", "AIR", Kind.BOOKKEEPING)


def test_annotations_and_misc() -> None:
    assert cmd("[opponent accepted power]").kind == Kind.ANNOTATION
    assert cmd("wait").kind == Kind.DECISION
    assert cmd("setup").kind == Kind.BOOKKEEPING
    assert cmd("done").kind == Kind.DECISION


def test_cultist_leech_bonus_bracket_is_its_own_verb_not_a_generic_annotation() -> None:
    """Task-13 report follow-up (``4pLeague_S10_D1L1_G5`` row 208):
    ``"[all opponents declined power]"`` carries a real state change
    (Cultists' ``leech_effect.not_taken``) that lands on this exact ledger
    row -- unlike other bracket text, it can't be discarded as a
    no-op ``annotation`` (``leech.py``'s module docstring has the full
    ordering citation for why)."""
    move = cmd("[all opponents declined power]")
    assert (move.verb, move.kind) == ("cultist_leech_bonus", Kind.BOOKKEEPING)


def test_real_world_variants() -> None:
    move = cmd("Convert pw to c")
    assert (move.n1, move.res1, move.n2, move.res2) == (1, "PW", 1, "C")
    move = cmd("convert 3pw to w")
    assert (move.n1, move.res1, move.n2, move.res2) == (3, "PW", 1, "W")
    move = cmd("Convert vp to c")  # Alchemists
    assert (move.res1, move.res2) == ("VP", "C")
    assert cmd("Advance shipping").reason == "ship"
    assert cmd("Advance digging").reason == "dig"
    assert (cmd("+2FIRE").cult, cmd("+2FIRE").n1) == ("FIRE", 2)
    assert (cmd("+2TW3").tile, cmd("+2TW3").n1) == ("TW3", 2)
    assert (cmd("-water").verb, cmd("-water").cult) == ("lose_cult", "WATER")
    assert cmd("-3CONVERT_W_TO_P").verb == "convert_marker"


def test_early_era_variants() -> None:
    leech = cmd("leech 2")
    assert (leech.verb, leech.n1, leech.target) == ("leech", 2, None)
    assert cmd("Decline").verb == "decline"
    assert cmd("all_income_for_faction").kind == Kind.INCOME
    assert cmd("Upgrade e7 to Trading Post").building == "TP"
    assert cmd("Upgrade B3 to Trading post").building == "TP"
    assert cmd("Upgrade E7 to Sanctuary").building == "SA"
    assert cmd("Upgrade f4 to Temple").building == "TE"
    assert cmd("Upgrade G1 to Stronghold").building == "SH"
    conv = cmd("Convert 1 power to 1c")
    assert (conv.res1, conv.res2, conv.n2) == ("PW", "C", 1)
    assert cmd("send priest to earth").cult == "EARTH"
    assert cmd("Send Priest to Earth for 2").n1 == 2
    assert cmd("advance ship to 1").n1 == 1
    assert cmd("Advance dig to 1").reason == "dig"
    assert cmd("advance dig 1").n1 == 1
    dig = cmd("DIG 1 I8")
    assert (dig.n1, dig.loc) == (1, "I8")
    conn = cmd("connect e4:f2")
    assert (conn.loc, conn.loc2) == ("E4", "F2")
    # River hexes are keyed lowercase on the board ("r0".."r35",
    # board.py's own docstring) -- unlike land hexes, a bare river-hex
    # `connect` reference must stay lowercase, not get the blanket
    # `.upper()` land hexes need (task-13 report, reference-game row 208:
    # "connect r1" produced "R1", a board.hexes lookup miss).
    assert cmd("connect r1").loc == "r1"
    assert cmd("connect R1").loc == "r1"
    assert cmd("-SPADE").verb == "lose_spade"
    assert cmd("-2SPADE").n1 == 2
    assert cmd("-FREE_D").reason == "FREE_D"
    assert cmd("-BRIDGE").reason == "BRIDGE"
    lost = cmd("-2c")
    assert (lost.verb, lost.n1, lost.res1) == ("lose_resource", 2, "C")


def test_unknown_command_returns_none() -> None:
    assert parse_command("frobnicate X9") is None


def test_parse_commands_collects_unknowns() -> None:
    moves, unknown = parse_commands("build A1. frobnicate. burn 2")
    assert [m.verb for m in moves] == ["build", "burn"]
    assert unknown == ["frobnicate"]


@pytest.mark.parametrize(
    "text",
    [
        "build G5",
        "upgrade C4 to SH",
        "action ACTS",
        "action BON2",
        "convert 2W to 2C",
        "Leech 2 from nomads",
        "pass BON10",
        "transform F6 to green",
        "send p to FIRE",
        "advance dig",
        "bridge f5:g6",
        "+FAV5",
        "+2vp for WATER",
        "connect r9",
    ],
)
def test_broad_vocabulary_parses(text: str) -> None:
    assert parse_command(text) is not None
