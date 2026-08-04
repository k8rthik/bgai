"""Final scoring: cult tracks, network size, and leftover-resource
conversion, plus the ``score_vp``/``score_resources`` ledger-row handlers.

Ported from the reference implementation (jsnell/terra-mystica, MIT),
``src/scoring.pm``.

--------------------------------------------------------------------------
Tie-split arithmetic (``score_type_rankings``, ``scoring.pm`` lines 58-75)
--------------------------------------------------------------------------

::

    sub score_type_rankings {
        my ($type, $fun, @scores) = @_;
        my @levels = sort { $a <=> $b } map { $_->{$type} // 0 } factions;
        my %count = (); $count{$_}++ for @levels;
        my %scores = ();
        for (@scores) {
            if (@levels) { $scores{pop @levels} += $_ }
        }
        for my $faction (factions) {
            my $level = $faction->{$type};
            next if !$level or !defined $scores{$level};
            my $vp = int($scores{$level} / $count{$level});
            $fun->($faction, $vp, $type) if $vp;
        }
    }

Read literally: sort every faction's level (position on a cult track, or
network size) ascending, including zeros and duplicates. Pop the *points*
list (``(8, 4, 2)`` for a cult, ``(18, 12, 6)`` for network) one entry at a
time, each pop consuming the current largest remaining level and adding
that many points into a bucket keyed by *that level value* (not by which
faction). Every faction whose own level matches a populated bucket key then
receives ``bucket[level] // count[level]`` VP, where ``count[level]`` is
the *total* number of factions at that level (not just the number of pops
that hit it) -- so a 2-way tie for first splits ``8 + 4 = 12`` two ways (6
each), a 4-way tie for first splits ``8 + 4 + 2 = 14`` four ways
(``14 // 4 == 3``, remainder dropped -- integer floor, no round-up), and a
faction whose own level never appears as a bucket key (because strictly
more distinct levels outrank it than there are ``points`` entries) scores
nothing even though it is at a nonzero level. Level 0 always scores
nothing regardless (``!$level`` short-circuits before the bucket lookup).
``_score_type_rankings`` below is a direct, order-independent transcription
(the Perl's per-faction iteration order affects only ledger row order, not
who is included in ``scores``/``count``, since both dicts are keyed by
level value; this port returns a ``{faction: vp}`` mapping instead of
calling a side-effecting ``$fun``, which is exactly as order-independent).

Corpus citation (game ``4pLeague_S10_D1L1_G1``, rows 369-388,
``moves.parquet``): FIRE score_vp rows are nomads=8, darklings=4,
mermaids=2 (engineers omitted -- FIRE=0), confirming the "position 0
scores nothing" rule and the untied 8/4/2 base case; WATER rows
nomads=4, darklings=8, mermaids=2 confirm rank is by level, not seat
order; NETWORK rows nomads=6, darklings=6, engineers=18, mermaids=6 --
engineers alone at the top network level keeps the whole 18, while
nomads/darklings/mermaids are a 3-way tie at some lower level, splitting
the "12 + 6 = 18" bucket three ways (``18 // 3 == 6`` each).

--------------------------------------------------------------------------
Network connectivity for final scoring
--------------------------------------------------------------------------

``scoring.pm`` ``score_final`` calls ``compute_network_size $_ for
factions_in_order()`` (line 82) before scoring anything, and
``compute_network_size`` (``map.pm`` lines 348-360) is::

    my %clique = find_building_cliques $faction, 1;   # allow_indirect = 1
    $faction->{network} = max values %{ +{ map { $_ => 1 } } ...};
        # (size of the largest clique)

``find_building_cliques($faction, $allow_indirect)`` (``map.pm`` lines
290-344) joins two of the faction's own building hexes whenever: (a) they
are directly adjacent (bridges folded into ``{adjacent}`` at build time),
(b) the faction is Mermaids and the pair shares a river-hex ``{skip}``
edge (checked *unconditionally*, independent of ``$allow_indirect``), or
(c) ``$allow_indirect`` and the pair is within shipping/tunnel/carpet
range of each other (``$map{$loc}{range}{$ship}{$to} <= $range``, where
``$range``/``$ship`` come from ``$faction->{teleport}`` if present --
Dwarves' ``tunnel_range``/Fakirs' ``carpet_range`` -- else
``$faction->{ship}{level}``). Since ``score_final``/``faction_vps`` always
call with ``$allow_indirect = 1``, final network scoring **does** count
Dwarves' tunneling and Fakirs' carpet flight (both factions have a
nonzero innate ``teleport.range`` from turn one -- ``factions_data.py``'s
``TeleportTrack`` -- since their ``ship.level`` is permanently 0, this is
their *only* source of indirect reach, not an addition on top of
shipping). There is no branch anywhere in ``find_building_cliques`` that
excludes them.

``connectivity.clusters(state, faction, river_skip=..., indirect=True)``
(this module's own addition, see that module's docstring) is exactly this
computation: ``river_skip`` is passed ``faction == "mermaids"`` (the
unconditional-``{skip}`` branch above) and ``indirect=True`` turns on the
range-based join, reusing ``connectivity.py``'s existing
``_shipping_reach``/``_teleport_reach``/``effective_shipping`` primitives
(the same ones ``reachable()`` uses for build-location legality) rather
than a fresh implementation -- "port EXACTLY what compute_network does"
per the task brief, achieved by sharing the one geometric primitive
instead of duplicating it. ``network_size`` below is the largest
component's hex count, or 0 if the faction has no buildings at all
(``max(..., default=0)`` -- the Perl's ``max`` over an empty clique-size
list is ``undef``, which the surrounding ``!$level`` check in
``score_type_rankings`` already treats as "scores nothing", so 0 is the
faithful Python stand-in).

--------------------------------------------------------------------------
``score_resources`` (leftover-resource-to-VP conversion at game end)
--------------------------------------------------------------------------

``scoring.pm`` ``score_final_resources_for_faction`` (lines 100-121)::

    my $b = int($faction->{P2} / 2);
    if ($b) { command "burn $b" }                    # bowl2 -> bowl3, 2:1
    for (1..$faction->{P3}) { command "convert 1pw to 1c" }   # bowl3 -> C, 1:1
    for (1..$faction->{P})  { command "convert 1p to 1c"  }   # priests -> C, 1:1
    for (1..$faction->{W})  { command "convert 1w to 1c"  }   # workers -> C, 1:1
    my $rate = $faction->{exchange_rates}{C}{VP} // 3;
    my $vp = int($faction->{C} / $rate);
    if ($vp) { command "convert ${vp}*rate C to ${vp}VP" }    # leftover C, floor

Order matters for the *burn* step only (it must happen before the bowl3
loop counts ``P3``, since burning is what feeds bowl3), the rest commute.
The corpus row for this verb (``score_resources``, ``moves.parquet``) is a
single bare marker with no fields of its own -- the whole conversion is
implicit, verified against ``deltas.parquet`` for game
``4pLeague_S10_D1L1_G1`` rows 390-393: nomads' power ``"2/3/0" -> "3/1/0"``
(burn 1: bowl2 3->1, bowl3 0->1; then convert bowl3's 1 to C: bowl3 1->0,
bowl1 2->3) with ``c_delta=+2`` (1 from the power conversion, 1 from
``w_delta=-1``'s worker conversion) and ``vp_delta=0`` (``int(2/3)==0`` --
not enough leftover coin for even 1 VP); engineers' row392 has
``P2=1 -> $b=int(1/2)=0`` (no burn at all, pw unchanged "4/1/0"->"4/1/0"),
``c_delta=+1`` from a lone worker conversion. Both match exactly.
``BASE_EXCHANGE_RATES["C"]["VP"] == 3`` (``factions_data.py``); Alchemists
override to ``{"C": {"VP": 2}}`` (that module's ``exchange_rate_overrides``
-- confirmed independently: Alchemists dominate the corpus's largest
``score_resources`` VP deltas, exactly the signature of a cheaper C->VP
rate). ``_apply_score_resources`` below is a direct, order-preserving
transcription; leftover coins below one VP's worth are left in ``coins``,
never discarded (``$c = $vp * $rate`` only converts the *spent* portion).

--------------------------------------------------------------------------
``score_vp``/``score_resources`` per-reason replay policy
--------------------------------------------------------------------------

Every ``score_vp`` row in the corpus (``moves.parquet``, ``verb ==
"score_vp"``) carries ``reason`` in ``{FIRE, WATER, EARTH, AIR,
NETWORK}`` -- no other reason code appears anywhere in the crawled corpus
(cross-checked via ``moves.filter(pl.col("verb") == "score_vp")
["reason"].value_counts()``: exactly those five keys, ~10.8k-11.1k rows
each). Round score-tile build/gain VP (e.g. a "TP >> 3vp" round tile) and
Engineers' pass-time bridge VP are *not* expressed through this verb at
all in this ledger format -- ``actions_pass.py``'s ``handle_pass`` already
folds pass-time VP (bonus/favor ``pass_vp`` tables, the Engineers hook)
directly into ``fs.vp`` as part of the ``pass`` row itself, with no
separate ``score_vp`` row following it (confirmed: the corpus's only
``score_vp`` reasons are the five final-scoring ones, and they only ever
appear in the four-cult-then-network block immediately preceding
``score_resources`` at the very end of round 6, per the row citation
above). So there is exactly one replay policy, not a table of several:

+-----------------------+---------------------------------------------------+
| ``reason``             | Policy                                             |
+-----------------------+---------------------------------------------------+
| ``FIRE``/``WATER``/    | Validate-and-apply. ``handle_score_vp`` recomputes |
| ``EARTH``/``AIR``      | that cult's ranking (``compute_cult_scoring``)     |
|                        | from the *current* ``state.cults`` and asserts     |
|                        | ``cmd.n1`` equals the engine's own answer before    |
|                        | applying it to ``fs.vp``. Cult positions are frozen |
|                        | by round 6's end (no further ``gain_cult``/``send`` |
|                        | rows follow in this block), so recomputing per row  |
|                        | is stable regardless of row order.                  |
+-----------------------+---------------------------------------------------+
| ``NETWORK``            | Validate-and-apply, same shape, via                 |
|                        | ``compute_network_scoring`` (buildings are frozen   |
|                        | too by this point -- no more build/upgrade/bridge   |
|                        | rows follow).                                       |
+-----------------------+---------------------------------------------------+
| (any other reason)     | Not observed in the corpus; ``handle_score_vp``     |
|                        | raises rather than silently trusting an unrecognized|
|                        | reason (the brief's "loud errors, not silent drift"). |
+-----------------------+---------------------------------------------------+
| ``score_resources``    | Apply-only (no fields to cross-check against -- the |
| (bare verb)            | row carries no VP/reason of its own). The engine     |
|                        | still fully *computes* the conversion itself         |
|                        | (``_apply_score_resources``) rather than trusting an |
|                        | opaque "something happened" marker -- so a wrong     |
|                        | conversion would surface as a downstream VP/resource |
|                        | mismatch against ``games_meta.final_vp`` once a      |
|                        | replay harness exists (Task 13), not silently.       |
+-----------------------+---------------------------------------------------+

No score-tile-VP or pass-VP reason is handled here because none exists in
this verb's reason vocabulary -- confirming (not just assuming) that
Tasks 8-11 already own 100% of what little of that VP the corpus's ledger
format expresses at all, and this task owns 100% of ``score_vp``/
``score_resources`` with no overlap in either direction.

--------------------------------------------------------------------------
Two-mode contract: ``final_scoring`` (simulation) vs. the handlers (replay)
--------------------------------------------------------------------------

``final_scoring(state) -> GameState`` is for **simulation** use (an agent
playing out a game with no ledger to replay against): given a
``Phase.FINISHED`` state at the end of round 6 (``round_flow.py``'s
documented Task-12 hand-off point -- "no final/area/resource-conversion
scoring has been applied yet"), it computes and applies cult scoring,
network scoring, and every faction's resource conversion in one call,
using the exact same ``compute_cult_scoring``/``compute_network_scoring``/
``_apply_score_resources`` functions the row handlers below validate
against. It must be called **at most once** per game and **never** during
ledger replay -- replay instead applies the corpus's own ``score_vp``/
``score_resources`` rows one at a time via ``apply()``, which dispatches
to ``handle_score_vp``/``handle_score_resources`` below. Calling
``final_scoring`` on a state that already replayed those rows (or calling
it twice) would double-apply every VP/resource delta: the shared
computation functions are pure reads of ``state.cults``/buildings/
resources, not idempotence-guarded, exactly like the Perl itself (running
``score_final`` twice would double-score there too -- it is simply never
invoked twice, by construction of the single-pass game loop). A replay
harness must call the handlers, not ``final_scoring``; a simulation must
call ``final_scoring``, not the handlers directly.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

from bgai.data.ledger_parser import ParsedCommand
from bgai.engine.tm.apply import EngineError, register_handler
from bgai.engine.tm.connectivity import clusters
from bgai.engine.tm.factions_data import BASE_EXCHANGE_RATES, CULTS, FACTIONS
from bgai.engine.tm.state import FactionState, GameState, Phase, with_faction

_CULT_POINTS: tuple[int, ...] = (8, 4, 2)
_NETWORK_POINTS: tuple[int, ...] = (18, 12, 6)


def _score_type_rankings(levels: Mapping[str, int], points: tuple[int, ...]) -> dict[str, int]:
    """Port of ``scoring.pm`` ``score_type_rankings`` (module docstring):
    ``{faction: vp}`` for every faction whose level places it in a scoring
    bucket, tie-split by integer floor. ``levels`` maps every participating
    faction to its raw level (cult track position, or network size).
    """
    remaining = sorted(levels.values())
    count: dict[int, int] = {}
    for level in remaining:
        count[level] = count.get(level, 0) + 1

    bucket: dict[int, int] = {}
    for pts in points:
        if not remaining:
            break
        top = remaining.pop()
        bucket[top] = bucket.get(top, 0) + pts

    result: dict[str, int] = {}
    for faction, level in levels.items():
        if level == 0 or level not in bucket:
            continue
        vp = bucket[level] // count[level]
        if vp:
            result[faction] = vp
    return result


def compute_cult_scoring(state: GameState) -> dict[str, dict[str, int]]:
    """``{cult: {faction: vp}}`` for the four cult tracks, 8/4/2 points."""
    return {
        cult: _score_type_rankings(
            {faction: state.cults[faction][cult] for faction in state.factions}, _CULT_POINTS
        )
        for cult in CULTS
    }


def network_size(state: GameState, faction: str) -> int:
    """Largest connected component of ``faction``'s buildings under final
    network-size rules (module docstring): direct adjacency/bridges, plus
    shipping/tunnel/carpet range, plus Mermaids' river-hex skip.
    """
    components = clusters(state, faction, river_skip=(faction == "mermaids"), indirect=True)
    return max((len(c) for c in components), default=0)


def compute_network_scoring(state: GameState) -> dict[str, int]:
    """``{faction: vp}`` for final network scoring, 18/12/6 points."""
    levels = {faction: network_size(state, faction) for faction in state.factions}
    return _score_type_rankings(levels, _NETWORK_POINTS)


def _exchange_rate_c_to_vp(faction: str) -> int:
    overrides = FACTIONS[faction].exchange_rate_overrides
    return overrides.get("C", {}).get("VP", BASE_EXCHANGE_RATES["C"]["VP"])


def _apply_score_resources(fs: FactionState, faction: str) -> FactionState:
    """Port of ``scoring.pm`` ``score_final_resources_for_faction`` (module
    docstring): burn floor(bowl2/2), convert all resulting bowl3/priests/
    workers to coins 1:1, then convert floor(coins/rate) coins to VP.
    """
    power = fs.power
    burn_n = power.bowl2 // 2
    if burn_n:
        power = power.burn(burn_n)

    coins = fs.coins
    if power.bowl3:
        coins += power.bowl3
        power = power.spend(power.bowl3)

    coins += fs.priests + fs.workers

    rate = _exchange_rate_c_to_vp(faction)
    vp_gained = coins // rate
    coins -= vp_gained * rate

    return replace(fs, power=power, coins=coins, priests=0, workers=0, vp=fs.vp + vp_gained)


def handle_score_vp(state: GameState, faction: str, cmd: ParsedCommand) -> GameState:
    """``+Nvp for REASON``: validation-anchor apply. Recomputes the named
    cult's (or network's) ranking from live state and asserts it matches
    ``cmd.n1`` before applying (module docstring's per-reason policy table
    -- every corpus reason is one of the five final-scoring ones).
    """
    assert cmd.n1 is not None and cmd.reason is not None
    reason = cmd.reason

    if reason == "NETWORK":
        expected = compute_network_scoring(state).get(faction, 0)
    elif reason in CULTS:
        expected = compute_cult_scoring(state)[reason].get(faction, 0)
    else:
        raise EngineError(
            f"unrecognized score_vp reason {reason!r}", state=state, faction=faction, cmd=cmd
        )

    if expected != cmd.n1:
        raise EngineError(
            f"{faction} score_vp for {reason}: engine computed {expected}, row says {cmd.n1}",
            state=state,
            faction=faction,
            cmd=cmd,
        )

    fs = state.factions[faction]
    return with_faction(state, faction, replace(fs, vp=fs.vp + cmd.n1))


def handle_score_resources(state: GameState, faction: str, cmd: ParsedCommand) -> GameState:
    """``score_resources``: apply-only (module docstring -- the bare row
    carries no fields to cross-check, but the conversion is still fully
    engine-computed, not trusted blind).
    """
    fs = state.factions[faction]
    return with_faction(state, faction, _apply_score_resources(fs, faction))


def final_scoring(state: GameState) -> GameState:
    """Simulation-mode entry point (module docstring): cult scoring,
    network scoring, then per-faction resource conversion, applied in one
    shot to a ``Phase.FINISHED`` state. Not for replay use -- see the
    module docstring's two-mode contract.
    """
    if state.phase != Phase.FINISHED:
        raise ValueError(
            f"final_scoring requires Phase.FINISHED (round 6 cleanup complete), got {state.phase}"
        )

    cult_scores = compute_cult_scoring(state)
    network_scores = compute_network_scoring(state)

    new_state = state
    for cult in CULTS:
        for faction, vp in cult_scores[cult].items():
            fs = new_state.factions[faction]
            new_state = with_faction(new_state, faction, replace(fs, vp=fs.vp + vp))

    for faction, vp in network_scores.items():
        fs = new_state.factions[faction]
        new_state = with_faction(new_state, faction, replace(fs, vp=fs.vp + vp))

    for faction in tuple(new_state.factions):
        fs = new_state.factions[faction]
        new_state = with_faction(new_state, faction, _apply_score_resources(fs, faction))

    return replace(new_state, phase=Phase.FINISHED)


register_handler("score_vp", handle_score_vp)
register_handler("score_resources", handle_score_resources)
