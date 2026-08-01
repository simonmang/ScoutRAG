"""Transfers, trophies, and injury history via the same licensed API-Football client.

Unlike the Wikidata enrichment, this needs no second provider and no name/date-of-birth
matching: every record already uses the same numeric API-Football player ID as the rest
of the pipeline. ``ApiFootballClient`` already provides caching, quota tracking, and
throttling, so this module only shapes three read-only endpoints into typed records.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from scoutrag.data.api_football import ApiFootballBudgetExceeded, ApiFootballClient
from scoutrag.domain.player import (
    PlayerCareerEvents,
    PlayerInjurySpell,
    PlayerTransfer,
    PlayerTrophy,
)


def fetch_career_events(
    client: ApiFootballClient,
    player_ids: list[str],
) -> tuple[list[PlayerCareerEvents], list[str]]:
    """Fetch career events for each player until the request budget/quota runs out.

    Returns ``(completed, remaining)`` so a caller can persist what finished and simply
    pass ``remaining`` back in on a later run - already-cached players resolve instantly
    and free of charge, so a multi-day pull needs no separate checkpoint file.
    """

    completed: list[PlayerCareerEvents] = []
    for index, player_id in enumerate(player_ids):
        numeric_id = _numeric_id(player_id)
        try:
            transfers = _fetch_transfers(client, numeric_id)
            trophies = _fetch_trophies(client, numeric_id)
            injuries = _fetch_injuries(client, numeric_id)
        except ApiFootballBudgetExceeded:
            return completed, player_ids[index:]
        completed.append(
            PlayerCareerEvents(
                player_id=player_id,
                transfers=transfers,
                trophies=trophies,
                injury_spells=injuries,
                source_reference=f"api-football:/transfers,/trophies,/sidelined?player={numeric_id}",
            )
        )
    return completed, []


def _numeric_id(player_id: str) -> str:
    return player_id.rsplit(":", 1)[-1]


def _response_items(response: Any) -> list[Any]:
    return response.response if isinstance(response.response, list) else []


def _fetch_transfers(client: ApiFootballClient, numeric_id: str) -> list[PlayerTransfer]:
    response = client.get("/transfers", {"player": numeric_id})
    transfers: list[PlayerTransfer] = []
    for entry in _response_items(response):
        if not isinstance(entry, dict):
            continue
        for item in entry.get("transfers") or []:
            teams = item.get("teams") or {}
            from_team = _text(teams.get("out"), "name")
            to_team = _text(teams.get("in"), "name")
            fee_text = item.get("type")
            if from_team is None or to_team is None or not fee_text:
                continue
            transfers.append(
                PlayerTransfer(
                    transfer_date=_parse_date(item.get("date")),
                    fee_text=str(fee_text),
                    from_team=from_team,
                    to_team=to_team,
                )
            )
    return transfers


def _fetch_trophies(client: ApiFootballClient, numeric_id: str) -> list[PlayerTrophy]:
    response = client.get("/trophies", {"player": numeric_id})
    trophies: list[PlayerTrophy] = []
    for item in _response_items(response):
        if not isinstance(item, dict):
            continue
        league, country, season, place = (
            item.get("league"),
            item.get("country"),
            item.get("season"),
            item.get("place"),
        )
        if not (league and country and season and place):
            continue
        trophies.append(
            PlayerTrophy(
                competition_name=str(league),
                country=str(country),
                season=str(season),
                place=str(place),
            )
        )
    return trophies


def _fetch_injuries(client: ApiFootballClient, numeric_id: str) -> list[PlayerInjurySpell]:
    response = client.get("/sidelined", {"player": numeric_id})
    injuries: list[PlayerInjurySpell] = []
    for item in _response_items(response):
        if not isinstance(item, dict):
            continue
        injury_type = item.get("type")
        if not injury_type:
            continue
        injuries.append(
            PlayerInjurySpell(
                injury_type=str(injury_type),
                start_date=_parse_date(item.get("start")),
                end_date=_parse_date(item.get("end")),
            )
        )
    return injuries


def _text(value: Any, key: str) -> str | None:
    if not isinstance(value, dict):
        return None
    text = value.get(key)
    return str(text) if isinstance(text, str) and text.strip() else None


def _parse_date(value: Any) -> date | None:
    if not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


__all__ = ["fetch_career_events"]
