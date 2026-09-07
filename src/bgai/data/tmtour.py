"""tmtour.org API client and response parsing.

tmtour is the Terra Mystica tournament index. Its open API (base ``/api2``)
lists 74+ seasons of games played on terra.snellman.net; the tmtour game
``name`` is the snellman game id (``4pLeague_S{s}_D{d}L{l}_G{g}``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from bgai.data.identity import user_agent

BASE_URL = "https://tmtour.org/api2"
REQUEST_TIMEOUT_SECONDS = 30.0

_GAME_NAME_RE = re.compile(r"^4pLeague_S(\d+)_D(\d+)L(\d+)_G(\d+)$")


@dataclass(frozen=True)
class GameName:
    season: int
    division: int
    league: int
    game_number: int


@dataclass(frozen=True)
class SeatResult:
    username: str
    faction: str
    score: int | None
    rank: int | None
    seat: int | None


@dataclass(frozen=True)
class TournamentGame:
    game_id: str
    name_parts: GameName
    finished: bool
    players: tuple[SeatResult, ...]


@dataclass(frozen=True)
class QualifyingResult:
    games: tuple[TournamentGame, ...]
    unparseable: list[str]
    excluded_division: int
    excluded_unfinished: int


def parse_game_name(name: str) -> GameName | None:
    """Parse a tmtour/snellman game id into its structural parts, or None."""
    match = _GAME_NAME_RE.match(name)
    if match is None:
        return None
    season, division, league, game_number = (int(g) for g in match.groups())
    return GameName(season=season, division=division, league=league, game_number=game_number)


def _parse_seat(raw: dict) -> SeatResult:
    return SeatResult(
        username=str(raw.get("username", "")),
        faction=str(raw.get("faction", "")),
        score=raw.get("score"),
        rank=raw.get("rank"),
        seat=raw.get("seat"),
    )


def qualifying_games(raw_games: list[dict], max_division: int) -> QualifyingResult:
    """Filter raw season games to finished games in divisions <= max_division."""
    kept: list[TournamentGame] = []
    unparseable: list[str] = []
    excluded_division = 0
    excluded_unfinished = 0

    for raw in raw_games:
        name = str(raw.get("name", ""))
        parts = parse_game_name(name)
        if parts is None:
            unparseable.append(name)
            continue
        if parts.division > max_division:
            excluded_division += 1
            continue
        if not raw.get("finished", False):
            excluded_unfinished += 1
            continue
        players = tuple(_parse_seat(p) for p in raw.get("players", []))
        kept.append(
            TournamentGame(game_id=name, name_parts=parts, finished=True, players=players)
        )

    return QualifyingResult(
        games=tuple(kept),
        unparseable=unparseable,
        excluded_division=excluded_division,
        excluded_unfinished=excluded_unfinished,
    )


def _client() -> httpx.Client:
    return httpx.Client(
        headers={"User-Agent": user_agent()},
        timeout=REQUEST_TIMEOUT_SECONDS,
        follow_redirects=True,
    )


@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=2, max=30))
def _get_json(client: httpx.Client, url: str) -> dict:
    response = client.get(url)
    response.raise_for_status()
    return response.json()


def fetch_seasons(client: httpx.Client | None = None) -> list[dict]:
    """All seasons: [{id, startdate, enddate, active, divisions}, ...]."""
    with client or _client() as c:
        payload = _get_json(c, f"{BASE_URL}/seasons")
    seasons = payload.get("seasons")
    if not isinstance(seasons, list) or not seasons:
        raise ValueError(f"unexpected /seasons payload: keys={list(payload)}")
    return seasons


def fetch_season_games(season_id: int, client: httpx.Client) -> list[dict]:
    """All games of one season with results (finished and not)."""
    payload = _get_json(client, f"{BASE_URL}/seasons/{season_id}/games")
    games = payload.get("games")
    if not isinstance(games, list):
        raise ValueError(f"unexpected /seasons/{season_id}/games payload: keys={list(payload)}")
    return games


def open_client() -> httpx.Client:
    """Shared client for multi-request sessions (caller manages lifetime)."""
    return _client()
