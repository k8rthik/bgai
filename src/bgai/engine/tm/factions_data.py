"""Base-game faction definitions for Terra Mystica (14 factions, no Fire & Ice).

Ported from the reference implementation (jsnell/terra-mystica, MIT):

- ``src/Game/Factions/<Name>.pm`` for each of the 14 base factions (each file
  is a single ``Readonly our $<name> => {...}`` hash, lines 6 to end of file):
  Alchemists, Auren, Chaosmagicians, Cultists, Darklings, Dwarves, Engineers,
  Fakirs, Giants, Halflings, Mermaids, Nomads, Swarmlings, Witches.
- ``src/Game/Factions.pm`` ``initialize_faction`` (lines 41-97): default P/P1/
  P2/P3 of 0, MAX_P 7, building max levels (D 8, TP 4, TE 3, SH 1, SA 1,
  lines 63-67), default GAIN_FAVOR on TE builds 1-3 and the SA build (lines
  69-72), TOWN_SIZE 7 / BRIDGE_COUNT 3 (lines 79-80), and base exchange rates
  (lines 83-88).
- ``src/Game/Constants.pm`` ``%actions`` (lines 55-105) for the faction
  stronghold/special actions ACTA/ACTC/ACTE/ACTG/ACTN/ACTS/ACTW, and
  ``@colors`` (line 240) for the terraform color wheel.
- Engine hooks that are behavior, not data, are quoted in each faction's
  ``notes``: ``src/scoring.pm`` lines 166-178 (Engineers bridge pass-VP),
  ``src/map.pm`` lines 511-513 and 612-614 (Giants transforms), ``src/map.pm``
  lines 281/326 and ``src/towns.pm`` lines 130-150 (Mermaids river towns),
  ``src/acting.pm`` lines 176-189 (setup order: Nomads 3 dwellings,
  Chaos Magicians 1, placed last), ``src/resources.pm``/``src/acting.pm``
  (Cultists leech timing).

Income tracks are cumulative: ``income["W"][n]`` is the total worker income
with ``n`` copies of that building on the board (index 0 = none built).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

CULTS: tuple[str, ...] = ("FIRE", "WATER", "EARTH", "AIR")

# Terraform color wheel, Constants.pm line 240.
COLOR_WHEEL: tuple[str, ...] = ("yellow", "brown", "black", "blue", "green", "gray", "red")

# Factions.pm initialize_faction, lines 53 and 79-80.
MAX_PRIESTS = 7
TOWN_SIZE = 7
BRIDGE_COUNT = 3

# Factions.pm lines 63-67: max copies of each building type.
BUILDING_MAX_COUNT: dict[str, int] = {"D": 8, "TP": 4, "TE": 3, "SH": 1, "SA": 1}

# Factions.pm lines 83-88 (%base_exchange_rates).
BASE_EXCHANGE_RATES: dict[str, dict[str, int]] = {
    "PW": {"C": 1, "W": 3, "P": 5},
    "W": {"C": 1},
    "P": {"C": 1, "W": 1},
    "C": {"VP": 3},
}

# Constants.pm %actions (lines 55-105), restricted to the once-per-round
# special actions granted by base-faction boards/strongholds.
FACTION_SPECIAL_ACTIONS: dict[str, dict[str, dict[str, int]]] = {
    # Auren SH: 2 cult steps on a single track.
    "ACTA": {"cost": {}, "gain": {"CULT": 2, "CULTS_ON_SAME_TRACK": 1}},
    # Chaos Magicians SH: take a double turn.
    "ACTC": {"cost": {}, "gain": {"GAIN_ACTION": 2}},
    # Engineers board: build a bridge for 2 workers (dont_block => 1).
    "ACTE": {"cost": {"W": 2}, "gain": {"BRIDGE": 1}},
    # Giants SH: free 2-spade transform.
    "ACTG": {"cost": {}, "gain": {"SPADE": 2}},
    # Nomads SH (Sandstorm): free transform, directly adjacent hex.
    "ACTN": {"cost": {}, "gain": {"FREE_TF": 1, "TF_NEED_HEX_ADJACENCY": 1}},
    # Swarmlings SH: free upgrade of a dwelling to a trading post.
    "ACTS": {"cost": {}, "gain": {"FREE_TP": 1}},
    # Witches SH (Witches' Ride): free dwelling, no terraform needed.
    "ACTW": {"cost": {}, "gain": {"FREE_D": 1, "TELEPORT_NO_TF": 1}},
}


@dataclass(frozen=True)
class BuildingTrack:
    """One building type: build cost, cumulative income track, on-build gains."""

    cost: dict[str, int]
    income: dict[str, tuple[int, ...]]
    max_count: int
    build_gain: tuple[dict[str, int], ...] = ()


@dataclass(frozen=True)
class ShippingTrack:
    level: int
    max_level: int
    advance_cost: dict[str, int] | None
    advance_vp: tuple[int, ...]


@dataclass(frozen=True)
class DigTrack:
    level: int
    max_level: int
    cost: tuple[dict[str, int], ...]  # spade cost, indexed by dig level
    advance_cost: dict[str, int] | None
    advance_vp: tuple[int, ...]
    dig_gain: tuple[dict[str, int], ...] = ()  # extra gain per dig, by level


@dataclass(frozen=True)
class TeleportTrack:
    """Dwarves tunneling / Fakirs carpet flight (Perl ``teleport`` hash)."""

    kind: str  # "tunnel" | "carpet"
    cost: tuple[dict[str, int], ...]  # indexed by teleport level (1 = post-SH)
    vp_gain: tuple[int, ...]
    range: int
    max_range: int
    advance_gain: tuple[dict[str, int], ...] = ()


@dataclass(frozen=True)
class FactionData:
    name: str
    display: str
    color: str
    coins: int
    workers: int
    priests: int
    power: tuple[int, int, int]  # bowls 1/2/3
    cults: dict[str, int]  # FIRE/WATER/EARTH/AIR
    shipping: ShippingTrack
    dig: DigTrack
    buildings: dict[str, BuildingTrack]
    stronghold_ability: str
    teleport: TeleportTrack | None = None
    special_gain: dict[str, dict[str, int]] = field(default_factory=dict)
    special_requires_sh: bool = False
    exchange_rate_overrides: dict[str, dict[str, int]] = field(default_factory=dict)
    leech_effect: dict[str, dict[str, int]] = field(default_factory=dict)
    special_actions: tuple[str, ...] = ()
    start_dwellings: int = 2
    notes: str = ""


def _cults(fire: int = 0, water: int = 0, earth: int = 0, air: int = 0) -> dict[str, int]:
    return {"FIRE": fire, "WATER": water, "EARTH": earth, "AIR": air}


# Standard tracks shared by most factions (verbatim in every faction .pm file
# that does not override them).
SHIP_STD = ShippingTrack(level=0, max_level=3, advance_cost={"C": 4, "P": 1}, advance_vp=(2, 3, 4))
SHIP_NONE = ShippingTrack(level=0, max_level=0, advance_cost=None, advance_vp=())
DIG_STD = DigTrack(
    level=0,
    max_level=2,
    cost=({"W": 3}, {"W": 2}, {"W": 1}),
    advance_cost={"W": 2, "C": 5, "P": 1},
    advance_vp=(6, 6),
)

D_STD = BuildingTrack(cost={"W": 1, "C": 2}, income={"W": (1, 2, 3, 4, 5, 6, 7, 8, 8)}, max_count=8)
TP_STD = BuildingTrack(
    cost={"W": 2, "C": 3}, income={"C": (0, 2, 4, 6, 8), "PW": (0, 1, 2, 4, 6)}, max_count=4
)
TE_STD = BuildingTrack(
    cost={"W": 2, "C": 5},
    income={"P": (0, 1, 2, 3)},
    max_count=3,
    build_gain=({"GAIN_FAVOR": 1},) * 3,
)
SA_STD = BuildingTrack(
    cost={"W": 4, "C": 6}, income={"P": (0, 1)}, max_count=1, build_gain=({"GAIN_FAVOR": 1},)
)


def _sh(
    build_gain: tuple[dict[str, int], ...] = (),
    income: dict[str, tuple[int, ...]] | None = None,
    cost: dict[str, int] | None = None,
) -> BuildingTrack:
    return BuildingTrack(
        cost=cost or {"W": 4, "C": 6},
        income=income or {"PW": (0, 2)},
        max_count=1,
        build_gain=build_gain,
    )


FACTIONS: dict[str, FactionData] = {
    "alchemists": FactionData(
        name="alchemists",
        display="Alchemists",
        color="black",
        coins=15,
        workers=3,
        priests=0,
        power=(5, 7, 0),
        cults=_cults(fire=1, water=1),
        shipping=SHIP_STD,
        dig=DIG_STD,
        buildings={
            "D": D_STD,
            "TP": replace(TP_STD, income={"C": (0, 2, 4, 7, 11), "PW": (0, 1, 2, 3, 4)}),
            "TE": TE_STD,
            "SH": _sh(build_gain=({"PW": 12},), income={"C": (0, 6)}),
            "SA": SA_STD,
        },
        stronghold_ability="On build: gain 12 power. Unlocks passive: 2 power per spade used.",
        special_gain={"SPADE": {"PW": 2}},
        special_requires_sh=True,
        exchange_rate_overrides={"C": {"VP": 2}, "VP": {"C": 1}},
        notes=(
            "Conversion: 'exchange_rates => { C => { VP => 2 }, VP => { C => 1 } }' — "
            "2 C per VP instead of the base 3, and may convert VP back to 1 C. "
            "Passive: 'special => { SPADE => { PW => 2 }, enable_if => { SH => 1 }, "
            "mode => gain }'."
        ),
    ),
    "auren": FactionData(
        name="auren",
        display="Auren",
        color="green",
        coins=15,
        workers=3,
        priests=0,
        power=(5, 7, 0),
        cults=_cults(water=1, air=1),
        shipping=SHIP_STD,
        dig=DIG_STD,
        buildings={
            "D": D_STD,
            "TP": TP_STD,
            "TE": TE_STD,
            "SH": _sh(build_gain=({"ACTA": 1, "GAIN_FAVOR": 1},)),
            "SA": replace(SA_STD, cost={"W": 4, "C": 8}),
        },
        stronghold_ability=(
            "On build: gain a favor tile and the ACTA special action "
            "(once per round: advance 2 steps on one cult track)."
        ),
        notes="SH 'advance_gain => [ { ACTA => 1, GAIN_FAVOR => 1 } ]'; see ACTA in %actions.",
    ),
    "chaosmagicians": FactionData(
        name="chaosmagicians",
        display="Chaos Magicians",
        color="red",
        coins=15,
        workers=4,
        priests=0,
        power=(5, 7, 0),
        cults=_cults(fire=2),
        shipping=SHIP_STD,
        dig=DIG_STD,
        buildings={
            "D": D_STD,
            "TP": TP_STD,
            "TE": replace(TE_STD, build_gain=({"GAIN_FAVOR": 2},) * 3),
            "SH": _sh(build_gain=({"ACTC": 1},), income={"W": (0, 2)}, cost={"W": 4, "C": 4}),
            "SA": replace(SA_STD, cost={"W": 4, "C": 8}, build_gain=({"GAIN_FAVOR": 2},)),
        },
        stronghold_ability=(
            "On build: gain the ACTC special action (once per round: take a double turn)."
        ),
        start_dwellings=1,
        notes=(
            "Temples and the sanctuary each grant 2 favor tiles instead of 1 "
            "('advance_gain => [ { GAIN_FAVOR => 2 } ... ]'). Setup: places a single "
            "initial dwelling, after all other factions (acting.pm setup_order)."
        ),
    ),
    "cultists": FactionData(
        name="cultists",
        display="Cultists",
        color="brown",
        coins=15,
        workers=3,
        priests=0,
        power=(5, 7, 0),
        cults=_cults(fire=1, earth=1),
        shipping=SHIP_STD,
        dig=DIG_STD,
        buildings={
            "D": D_STD,
            "TP": TP_STD,
            "TE": TE_STD,
            "SH": _sh(build_gain=({"VP": 7},), cost={"W": 4, "C": 8}),
            "SA": replace(SA_STD, cost={"W": 4, "C": 8}),
        },
        stronghold_ability="On build: gain 7 VP.",
        leech_effect={"taken": {"CULT": 1}, "not_taken": {"PW": 1}},
        notes=(
            "'leech_effect => { taken => { CULT => 1 }, not_taken => { PW => 1 } }': when at "
            "least one opponent accepts power from a Cultist build, gain 1 cult step of "
            "choice; if all decline, gain 1 power (resources.pm/acting.pm handle the timing)."
        ),
    ),
    "darklings": FactionData(
        name="darklings",
        display="Darklings",
        color="black",
        coins=15,
        workers=1,
        priests=1,
        power=(5, 7, 0),
        cults=_cults(water=1, earth=1),
        shipping=SHIP_STD,
        dig=DigTrack(
            level=0,
            max_level=0,
            cost=({"P": 1},),
            advance_cost=None,
            advance_vp=(),
            dig_gain=({"SPADE": 1, "VP": 2},),
        ),
        buildings={
            "D": D_STD,
            "TP": TP_STD,
            "TE": TE_STD,
            "SH": _sh(build_gain=({"CONVERT_W_TO_P": 3},)),
            "SA": BuildingTrack(
                cost={"W": 4, "C": 10},
                income={"P": (0, 2)},
                max_count=1,
                build_gain=({"GAIN_FAVOR": 1},),
            ),
        },
        stronghold_ability=(
            "On build: may immediately convert up to 3 workers to priests "
            "('advance_gain => [ { CONVERT_W_TO_P => 3 } ]')."
        ),
        notes=(
            "Digging costs priests, never advances, and scores: 'dig => { level => 0, "
            "max_level => 0, cost => [ { P => 1 } ], gain => [ { SPADE => 1, VP => 2 } ] }'. "
            "Sanctuary yields 2 priests income."
        ),
    ),
    "dwarves": FactionData(
        name="dwarves",
        display="Dwarves",
        color="gray",
        coins=15,
        workers=3,
        priests=0,
        power=(5, 7, 0),
        cults=_cults(earth=2),
        shipping=SHIP_NONE,
        dig=DIG_STD,
        buildings={
            "D": D_STD,
            "TP": replace(TP_STD, income={"C": (0, 3, 5, 7, 10), "PW": (0, 1, 2, 4, 6)}),
            "TE": TE_STD,
            "SH": _sh(build_gain=({"GAIN_TELEPORT": 1},)),
            "SA": SA_STD,
        },
        stronghold_ability=(
            "On build: tunneling gets cheaper (teleport level 1: 1 worker instead of 2)."
        ),
        teleport=TeleportTrack(
            kind="tunnel", cost=({"W": 2}, {"W": 1}), vp_gain=(4, 4), range=1, max_range=1
        ),
        notes=(
            "No shipping track ('ship => { level => 0, max_level => 0 }'). Tunneling: build "
            "across one skipped hex for 2 W (1 W after SH), gaining 4 VP per tunnel "
            "('teleport => { type => tunnel, cost => [ {W=>2}, {W=>1} ], gain => [ {VP=>4}, "
            "{VP=>4} ] }', tunnel_range 1)."
        ),
    ),
    "engineers": FactionData(
        name="engineers",
        display="Engineers",
        color="gray",
        coins=10,
        workers=2,
        priests=0,
        power=(3, 9, 0),
        cults=_cults(),
        shipping=SHIP_STD,
        dig=DIG_STD,
        buildings={
            "D": BuildingTrack(
                cost={"W": 1, "C": 1}, income={"W": (0, 1, 2, 2, 3, 4, 4, 5, 6)}, max_count=8
            ),
            "TP": replace(TP_STD, cost={"W": 1, "C": 2}),
            "TE": replace(
                TE_STD, cost={"W": 1, "C": 4}, income={"P": (0, 1, 1, 2), "PW": (0, 0, 5, 5)}
            ),
            "SH": _sh(cost={"W": 3, "C": 6}),
            "SA": replace(SA_STD, cost={"W": 3, "C": 6}),
        },
        stronghold_ability=(
            "Passive after build: when passing, gain 3 VP per own bridge connecting "
            "two gray-hex buildings."
        ),
        special_actions=("ACTE",),
        notes=(
            "'ACTE => 1': may build a bridge for 2 workers as a non-blocking special action. "
            "SH pass-VP is an engine hook, scoring.pm: \"if ($faction->{name} eq 'engineers' "
            "and $faction->{buildings}{SH}{level}) { ... 3 VP for each bridge whose two ends "
            'are gray hexes with buildings }". Cheap buildings but reduced income tracks '
            "(D income starts at 0; TE mixes P and PW)."
        ),
    ),
    "fakirs": FactionData(
        name="fakirs",
        display="Fakirs",
        color="yellow",
        coins=15,
        workers=3,
        priests=0,
        power=(7, 5, 0),
        cults=_cults(fire=1, air=1),
        shipping=SHIP_NONE,
        dig=DigTrack(
            level=0,
            max_level=1,
            cost=({"W": 3}, {"W": 2}),
            advance_cost={"W": 2, "C": 5, "P": 1},
            advance_vp=(6,),
        ),
        buildings={
            "D": D_STD,
            "TP": TP_STD,
            "TE": TE_STD,
            "SH": _sh(
                build_gain=({"GAIN_TELEPORT": 1},),
                income={"P": (0, 1)},
                cost={"W": 4, "C": 10},
            ),
            "SA": SA_STD,
        },
        stronghold_ability="On build: carpet flight range increases by 1 (advance teleport).",
        teleport=TeleportTrack(
            kind="carpet",
            cost=({"P": 1}, {"P": 1}),
            vp_gain=(4, 4),
            range=1,
            max_range=4,
            advance_gain=({"carpet_range": 1},),
        ),
        notes=(
            "No shipping; dig track capped at level 1. Carpet flight: build across skipped "
            "hexes for 1 priest, gaining 4 VP ('teleport => { type => carpet, cost => "
            "[ {P=>1}, {P=>1} ], gain => [ {VP=>4}, {VP=>4} ], advance_gain => "
            "[ { carpet_range => 1 } ] }', carpet_range 1, carpet_max_range 4)."
        ),
    ),
    "giants": FactionData(
        name="giants",
        display="Giants",
        color="red",
        coins=15,
        workers=3,
        priests=0,
        power=(5, 7, 0),
        cults=_cults(fire=1, air=1),
        shipping=SHIP_STD,
        dig=DIG_STD,
        buildings={
            "D": D_STD,
            "TP": TP_STD,
            "TE": TE_STD,
            "SH": _sh(build_gain=({"ACTG": 1},), income={"PW": (0, 4)}),
            "SA": SA_STD,
        },
        stronghold_ability=(
            "On build: gain the ACTG special action (once per round: transform any "
            "reachable hex to home terrain with 2 free spades)."
        ),
        notes=(
            "Passive engine hook, map.pm: \"if ($faction->{name} eq 'giants' and "
            '$color_difference != 0) { $color_difference = 2; }" — every transform costs '
            "exactly 2 spades regardless of distance on the color wheel, and always "
            "transforms directly to home terrain."
        ),
    ),
    "halflings": FactionData(
        name="halflings",
        display="Halflings",
        color="brown",
        coins=15,
        workers=3,
        priests=0,
        power=(3, 9, 0),
        cults=_cults(earth=1, air=1),
        shipping=SHIP_STD,
        dig=replace(DIG_STD, advance_cost={"W": 2, "C": 1, "P": 1}),
        buildings={
            "D": D_STD,
            "TP": TP_STD,
            "TE": TE_STD,
            "SH": _sh(build_gain=({"SPADE": 3},), cost={"W": 4, "C": 8}),
            "SA": SA_STD,
        },
        stronghold_ability=(
            "On build: immediately gain 3 spades (may transform up to 3 hexes and "
            "build 1 dwelling: 'subactions => { transform => 3, build => 1 }')."
        ),
        special_gain={"SPADE": {"VP": 1}},
        notes=(
            "Passive: 'special => { mode => gain, SPADE => { VP => 1 } }' — 1 VP per spade. "
            "Dig advances cost only 1 coin ('advance_cost => { W => 2, C => 1, P => 1 }')."
        ),
    ),
    "mermaids": FactionData(
        name="mermaids",
        display="Mermaids",
        color="blue",
        coins=15,
        workers=3,
        priests=0,
        power=(3, 9, 0),
        cults=_cults(water=2),
        shipping=ShippingTrack(
            level=1, max_level=5, advance_cost={"C": 4, "P": 1}, advance_vp=(0, 2, 3, 4, 5)
        ),
        dig=DIG_STD,
        buildings={
            "D": D_STD,
            "TP": TP_STD,
            "TE": TE_STD,
            "SH": _sh(build_gain=({"GAIN_SHIP": 1},), income={"PW": (0, 4)}),
            "SA": replace(SA_STD, cost={"W": 4, "C": 8}),
        },
        stronghold_ability="On build: immediately advance shipping one level for free.",
        notes=(
            "Starts at shipping 1, max 5 (first advance from the free SH gain scores 0 VP). "
            "Passive engine hook: may skip one river hex when founding a town (towns.pm "
            "check_mermaid_river_connection_town; map.pm \"$faction->{name} eq 'mermaids' "
            'and exists $map{$where}{skip}").'
        ),
    ),
    "nomads": FactionData(
        name="nomads",
        display="Nomads",
        color="yellow",
        coins=15,
        workers=2,
        priests=0,
        power=(5, 7, 0),
        cults=_cults(fire=1, earth=1),
        shipping=SHIP_STD,
        dig=DIG_STD,
        buildings={
            "D": D_STD,
            "TP": replace(TP_STD, income={"C": (0, 2, 4, 7, 11), "PW": (0, 1, 2, 3, 4)}),
            "TE": TE_STD,
            "SH": _sh(build_gain=({"ACTN": 1},), cost={"W": 4, "C": 8}),
            "SA": SA_STD,
        },
        stronghold_ability=(
            "On build: gain the ACTN special action (Sandstorm — once per round, freely "
            "transform a directly adjacent hex to home terrain, no spades)."
        ),
        start_dwellings=3,
        notes=(
            "Setup: places 3 initial dwellings (acting.pm setup_order pushes an extra "
            "['nomads', 'dwelling']). ACTN requires direct hex adjacency "
            "(TF_NEED_HEX_ADJACENCY => 1) and cannot build in the same action."
        ),
    ),
    "swarmlings": FactionData(
        name="swarmlings",
        display="Swarmlings",
        color="blue",
        coins=20,
        workers=8,
        priests=0,
        power=(3, 9, 0),
        cults=_cults(fire=1, water=1, earth=1, air=1),
        shipping=SHIP_STD,
        dig=DIG_STD,
        buildings={
            "D": BuildingTrack(
                cost={"W": 2, "C": 3}, income={"W": (2, 3, 4, 5, 6, 7, 8, 9, 9)}, max_count=8
            ),
            "TP": BuildingTrack(
                cost={"W": 3, "C": 4},
                income={"PW": (0, 2, 4, 6, 8), "C": (0, 2, 4, 6, 9)},
                max_count=4,
            ),
            "TE": replace(TE_STD, cost={"W": 3, "C": 6}),
            "SH": _sh(build_gain=({"ACTS": 1},), income={"PW": (0, 4)}, cost={"W": 5, "C": 8}),
            "SA": BuildingTrack(
                cost={"W": 5, "C": 8},
                income={"P": (0, 2)},
                max_count=1,
                build_gain=({"GAIN_FAVOR": 1},),
            ),
        },
        stronghold_ability=(
            "On build: gain the ACTS special action (once per round: upgrade a dwelling "
            "to a trading post for free)."
        ),
        special_gain={"TOWN": {"W": 3}},
        notes=(
            "Passive: 3 workers per town founded — 'special => { mode => gain, "
            'map(("TW$_", { W => 3 }), 1..8) }\'. All buildings cost more; all income '
            "tracks are richer; sanctuary yields 2 priests."
        ),
    ),
    "witches": FactionData(
        name="witches",
        display="Witches",
        color="green",
        coins=15,
        workers=3,
        priests=0,
        power=(5, 7, 0),
        cults=_cults(air=2),
        shipping=SHIP_STD,
        dig=DIG_STD,
        buildings={
            "D": D_STD,
            "TP": TP_STD,
            "TE": TE_STD,
            "SH": _sh(build_gain=({"ACTW": 1},)),
            "SA": SA_STD,
        },
        stronghold_ability=(
            "On build: gain the ACTW special action (Witches' Ride — once per round, build "
            "a free dwelling on any empty green hex, no adjacency or terraform needed)."
        ),
        special_gain={"TOWN": {"VP": 5}},
        notes=(
            "Passive: 5 VP per town founded — 'special => { mode => gain, "
            'map(("TW$_", { VP => 5 }), 1..8) }\'. ACTW gain: '
            "{ FREE_D => 1, TELEPORT_NO_TF => 1 }."
        ),
    ),
}
