"""Deterministic normalization from nested StatsBomb JSON to flat records."""

from datetime import date
from typing import Any

from scoutrag.data.models import CompetitionSeason, MatchRecord, NormalizedEvent


def _nested_name(value: object) -> str | None:
    if not isinstance(value, dict) or value.get("name") is None:
        return None
    return str(value["name"])


def _nested_id(value: object) -> int | None:
    if not isinstance(value, dict) or value.get("id") is None:
        return None
    return int(value["id"])


def _event_payload(raw_event: dict[str, Any], event_type: str) -> dict[str, Any]:
    known_keys = {
        "Ball Receipt*": "ball_receipt",
        "Ball Recovery": "ball_recovery",
        "Goal Keeper": "goalkeeper",
    }
    key = known_keys.get(event_type, event_type.lower().replace(" ", "_").replace("*", ""))
    value = raw_event.get(key)
    return value if isinstance(value, dict) else {}


def normalize_event(
    raw_event: dict[str, Any],
    *,
    match_id: int,
    competition_id: int,
    season_id: int,
) -> NormalizedEvent:
    """Flatten relevant common fields while retaining source provenance."""
    event_id = str(raw_event["id"])
    event_type = _nested_name(raw_event.get("type"))
    if event_type is None:
        raise ValueError(f"event {event_id} has no event type")

    player = raw_event.get("player")
    team = raw_event.get("team")
    position = raw_event.get("position")
    payload = _event_payload(raw_event, event_type)
    location = raw_event.get("location")
    location_values = location if isinstance(location, list) else []
    end_location = payload.get("end_location")
    end_location_values = end_location if isinstance(end_location, list) else []
    minute = float(raw_event.get("minute", 0))
    second = float(raw_event.get("second", 0))

    return NormalizedEvent(
        event_id=event_id,
        match_id=match_id,
        competition_id=competition_id,
        season_id=season_id,
        event_index=int(raw_event["index"]),
        period=int(raw_event["period"]),
        timestamp=str(raw_event["timestamp"]),
        match_second=(minute * 60) + second,
        event_type=event_type,
        event_subtype=_nested_name(payload.get("type")),
        outcome_name=_nested_name(payload.get("outcome")),
        team_id=_nested_id(team),
        team_name=_nested_name(team),
        player_id=_nested_id(player),
        player_name=_nested_name(player),
        position_name=_nested_name(position),
        location_x=(
            float(location_values[0])
            if len(location_values) >= 1 and location_values[0] is not None
            else None
        ),
        location_y=(
            float(location_values[1])
            if len(location_values) >= 2 and location_values[1] is not None
            else None
        ),
        end_location_x=(
            float(end_location_values[0])
            if len(end_location_values) >= 1 and end_location_values[0] is not None
            else None
        ),
        end_location_y=(
            float(end_location_values[1])
            if len(end_location_values) >= 2 and end_location_values[1] is not None
            else None
        ),
        expected_goals=(
            float(payload["statsbomb_xg"]) if payload.get("statsbomb_xg") is not None else None
        ),
        pass_length=(float(payload["length"]) if payload.get("length") is not None else None),
        duration_seconds=(
            float(raw_event["duration"]) if raw_event.get("duration") is not None else None
        ),
        under_pressure=bool(raw_event.get("under_pressure", False)),
        counterpress=bool(raw_event.get("counterpress", False)),
        source_reference=f"statsbomb:matches/{match_id}/events/{event_id}",
    )


def normalize_events(
    raw_events: list[dict[str, Any]],
    *,
    match_id: int,
    competition_id: int,
    season_id: int,
) -> list[NormalizedEvent]:
    """Normalize and preserve StatsBomb's stable event order."""
    return sorted(
        (
            normalize_event(
                event,
                match_id=match_id,
                competition_id=competition_id,
                season_id=season_id,
            )
            for event in raw_events
        ),
        key=lambda event: event.event_index,
    )


def normalize_match(
    raw_match: dict[str, Any],
    competition: CompetitionSeason,
    events: list[NormalizedEvent],
) -> MatchRecord:
    """Normalize match metadata and derive its observed elapsed duration."""
    match_id = int(raw_match["match_id"])
    home_team = raw_match["home_team"]
    away_team = raw_match["away_team"]
    if not isinstance(home_team, dict) or not isinstance(away_team, dict):
        raise ValueError(f"match {match_id} has invalid team metadata")

    duration_seconds = max((event.match_second for event in events), default=0)
    if duration_seconds <= 0:
        raise ValueError(f"match {match_id} has no positive event duration")

    return MatchRecord(
        match_id=match_id,
        competition_id=competition.competition_id,
        season_id=competition.season_id,
        match_date=date.fromisoformat(str(raw_match["match_date"])),
        match_week=(
            int(raw_match["match_week"]) if raw_match.get("match_week") is not None else None
        ),
        home_team_id=int(home_team["home_team_id"]),
        home_team_name=str(home_team["home_team_name"]),
        away_team_id=int(away_team["away_team_id"]),
        away_team_name=str(away_team["away_team_name"]),
        home_score=int(raw_match["home_score"]),
        away_score=int(raw_match["away_score"]),
        duration_seconds=duration_seconds,
        source_reference=f"statsbomb:matches/{match_id}",
    )
