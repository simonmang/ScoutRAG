"""Calculate player minutes from explicit StatsBomb lineup intervals."""

from collections import defaultdict
from typing import Any

from scoutrag.data.models import CompetitionSeason, MatchRecord, PlayerMatchParticipation


def parse_elapsed_seconds(value: str) -> float:
    """Parse StatsBomb elapsed timestamps such as ``68:47`` or ``01:05:12``."""
    parts = value.split(":")
    if len(parts) == 2:
        minutes, seconds = parts
        return (float(minutes) * 60) + float(seconds)
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return (float(hours) * 3600) + (float(minutes) * 60) + float(seconds)
    raise ValueError(f"unsupported elapsed timestamp: {value}")


def position_group(position_name: str) -> str:
    """Map a detailed StatsBomb position to a scouting-oriented comparison group."""
    normalized = position_name.casefold()
    if "goalkeeper" in normalized:
        return "goalkeeper"
    if "center back" in normalized:
        return "center_back"
    if "wing back" in normalized or "full back" in normalized:
        return "fullback_wingback"
    if normalized.endswith(" back"):
        return "fullback_wingback"
    if "defensive midfield" in normalized:
        return "defensive_midfield"
    if "attacking midfield" in normalized:
        return "attacking_midfield"
    if "center midfield" in normalized:
        return "central_midfield"
    if "wing" in normalized:
        return "winger"
    if "forward" in normalized or "striker" in normalized:
        return "forward"
    return "other"


def calculate_match_participations(
    raw_lineups: list[dict[str, Any]],
    match: MatchRecord,
    competition: CompetitionSeason,
) -> list[PlayerMatchParticipation]:
    """Sum lineup intervals and retain the position with most elapsed time."""
    participations: list[PlayerMatchParticipation] = []
    for team in raw_lineups:
        team_id = int(team["team_id"])
        team_name = str(team["team_name"])
        players = team.get("lineup")
        if not isinstance(players, list):
            raise ValueError(f"match {match.match_id}, team {team_id} has no lineup array")

        for player in players:
            if not isinstance(player, dict):
                continue
            positions = player.get("positions")
            if not isinstance(positions, list) or not positions:
                continue

            seconds_by_position: defaultdict[str, float] = defaultdict(float)
            started = False
            for interval in positions:
                if not isinstance(interval, dict):
                    continue
                start_value = str(interval.get("from") or "00:00")
                end_value = interval.get("to")
                start_seconds = parse_elapsed_seconds(start_value)
                end_seconds = (
                    parse_elapsed_seconds(str(end_value))
                    if end_value is not None
                    else match.duration_seconds
                )
                elapsed = max(0.0, end_seconds - start_seconds)
                detailed_position = str(interval.get("position") or "Unknown")
                seconds_by_position[detailed_position] += elapsed
                started = started or interval.get("start_reason") == "Starting XI"

            total_seconds = sum(seconds_by_position.values())
            if total_seconds <= 0:
                continue
            primary_position = max(seconds_by_position, key=seconds_by_position.__getitem__)
            player_id = int(player["player_id"])
            participations.append(
                PlayerMatchParticipation(
                    match_id=match.match_id,
                    competition_id=competition.competition_id,
                    season_id=competition.season_id,
                    player_id=player_id,
                    player_name=str(player["player_name"]),
                    team_id=team_id,
                    team_name=team_name,
                    primary_position=primary_position,
                    position_group=position_group(primary_position),
                    minutes_played=round(total_seconds / 60, 3),
                    started=started,
                    source_reference=(
                        f"statsbomb:matches/{match.match_id}/lineups/players/{player_id}"
                    ),
                )
            )
    return participations
