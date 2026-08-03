"""Per-faction income computation.

Reference: jsnell/terra-mystica ``src/income.pm`` ``faction_income`` (lines
12-94). The Perl accumulates four buckets and sums them into one
``total_income``:

- building income (lines 24-36): for every building TYPE the faction's
  board tracks (D/TP/TE/SH/SA -- each initialized in
  ``Game/Factions.pm`` ``initialize_faction``, lines 62-67, regardless of
  whether any copies are built yet), look up ``income[type][level]`` where
  ``level`` is the count of that type currently on the board, and add it to
  ``total_building_income``. The tracks are cumulative
  (``income["W"][n]`` = total worker income with ``n`` copies built,
  confirmed against every ``.pm`` faction file: e.g. Darklings.pm line 26
  ``income => { W => [ 1, 2, 3, 4, 5, 6, 7, 8, 8 ] }``, Nomads.pm line 27,
  Mermaids.pm line 30 -- all identical D_STD tracks).
- bonus income (lines 43-52): the held BON tile's ``income`` dict
  (Constants.pm lines 127-141), added to ``total_bonus_income``.
- favor income (lines 43-52): every held FAV tile's ``income`` dict
  (Constants.pm lines 143-157), summed into ``total_favor_income``.
- scoring income (lines 55-63): the current round-scoring tile's cult-based
  income, scaled by cult position -- **not implemented here**. Round 1 of
  the reference game has no scoring-tile income to validate against and
  this belongs with cult/scoring machinery from a later task; this module
  only produces the four buckets the task-4 brief interface names.

Base vs. buildings split (this module's own design choice, not Perl's):
Perl's per-building-type record always exists at count 0 (see
``initialize_faction`` above), so a building type's income-track *floor*
(``income[type][0]``, the value at zero copies built) silently flows into
the same ``total_building_income`` bucket as the marginal per-building
income. Confirmed by grepping every base faction's ``.pm`` file
(``Game/Factions/*.pm``): only the dwelling (``D``) track ever has a
nonzero floor -- every faction gets 1 W from ``D`` at zero dwellings
except Engineers (``Engineers.pm`` line 27: ``income => { W => [ 0, 1, 2,
2, 3, 4, 4, 5, 6 ] }``, floor 0) and Swarmlings (``Swarmlings.pm`` line 32:
``income => { W => [ 2, 3, 4, 5, 6, 7, 8, 9, 9 ] }``, floor 2). Every
``TP``/``TE``/``SH``/``SA`` track floor is 0 for every faction, so this
split never affects them. Because the task-4 interface asks for a distinct
``"base"`` category separate from ``"buildings"``, this module reports the
floor (``income[type][0]``) under ``base`` and the marginal amount above
the floor (``income[type][count] - income[type][0]``) under ``buildings``.
The two buckets still sum to exactly the same per-type track lookup the
Perl uses for that type, so total income (and the Step-6 reference-game
data-driven check) is unaffected by which bucket a resource lands in.

Only C/W/P/PW ever appear as income resource keys (the tile/track data in
``tiles.py``/``factions_data.py`` -- transcribed from ``Constants.pm`` and
the faction ``.pm`` files -- never uses any other resource as an income
type; SPADE-granting bonus/favor tiles model that via ``special_action``,
not ``income``).
"""

from __future__ import annotations

from bgai.engine.tm.factions_data import FactionData
from bgai.engine.tm.state import FactionState
from bgai.engine.tm.tiles import BONUS_TILES, FAVOR_TILES


def _add(bucket: dict[str, int], resource: str, amount: int) -> None:
    """Accumulate ``amount`` under ``resource`` in ``bucket``, skipping zeros.

    Mirrors income.pm line 31 (``if ($delta) { ... }``): a zero contribution
    never creates a key.
    """
    if amount:
        bucket[resource] = bucket.get(resource, 0) + amount


def _building_and_base_income(
    fs: FactionState, data: FactionData
) -> tuple[dict[str, int], dict[str, int]]:
    """Split each building type's cumulative track into base floor + marginal.

    income.pm lines 26-36: for each building type, look up
    ``income[type][count]`` (cumulative track indexed by copies built). Here
    that lookup is decomposed into ``income[type][0]`` (floor, reported as
    "base") and the remainder above the floor (reported as "buildings").
    """
    base: dict[str, int] = {}
    buildings: dict[str, int] = {}
    for building_type, track in data.buildings.items():
        count = len(fs.buildings.get(building_type, frozenset()))
        for resource, cumulative in track.income.items():
            floor = cumulative[0]
            index = min(count, len(cumulative) - 1)
            value = cumulative[index]
            _add(base, resource, floor)
            _add(buildings, resource, value - floor)
    return base, buildings


def _bonus_income(fs: FactionState) -> dict[str, int]:
    """income.pm lines 43-47: the held BON tile's income dict, verbatim."""
    if fs.bonus is None:
        return {}
    bucket: dict[str, int] = {}
    for resource, amount in BONUS_TILES[fs.bonus].income.items():
        _add(bucket, resource, amount)
    return bucket


def _favor_income(fs: FactionState) -> dict[str, int]:
    """income.pm lines 43-49: sum of every held FAV tile's income dict."""
    bucket: dict[str, int] = {}
    for favor_id in fs.favors:
        for resource, amount in FAVOR_TILES[favor_id].income.items():
            _add(bucket, resource, amount)
    return bucket


def faction_income(fs: FactionState, data: FactionData) -> dict[str, dict[str, int]]:
    """Per-source income for `fs`, split into base/buildings/bonus/favors.

    Pure function of the given state and faction data -- no mutation, no
    round/scoring-tile awareness (round-scoring cult income and how these
    four buckets map onto ledger rows are later tasks; see module
    docstring). Resource keys are always a subset of C/W/P/PW; a category
    with no income for a resource simply omits that key (never a 0 entry).
    """
    base, buildings = _building_and_base_income(fs, data)
    return {
        "base": base,
        "buildings": buildings,
        "bonus": _bonus_income(fs),
        "favors": _favor_income(fs),
    }
