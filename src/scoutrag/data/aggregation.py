"""Aggregate normalized events into raw season profiles and metric evidence."""

from collections import defaultdict
from dataclasses import dataclass

from scoutrag.data.models import (
    CompetitionSeason,
    NormalizedEvent,
    PlayerMatchParticipation,
)
from scoutrag.domain.player import PlayerMetricEvidence, PlayerSeasonProfile

RAW_METRICS = (
    "ball_recoveries",
    "carries",
    "clearances",
    "dribbles_completed",
    "interceptions",
    "passes",
    "passes_completed",
    "pressures",
    "progressive_carries",
    "progressive_passes",
    "shots",
    "expected_goals",
    "tackles",
)


@dataclass(frozen=True, slots=True)
class SeasonAggregation:
    """Phase 2 aggregate output before persistence."""

    profiles: list[PlayerSeasonProfile]
    evidence: list[PlayerMetricEvidence]


def aggregate_player_seasons(
    competition: CompetitionSeason,
    events: list[NormalizedEvent],
    participations: list[PlayerMatchParticipation],
) -> SeasonAggregation:
    """Build raw-count profiles without introducing Phase 3 per-90 features."""
    participation_groups: defaultdict[int, list[PlayerMatchParticipation]] = defaultdict(list)
    for participation in participations:
        participation_groups[participation.player_id].append(participation)

    event_counts: defaultdict[int, defaultdict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    for event in events:
        if event.player_id is None:
            continue
        metrics = event_counts[event.player_id]
        metrics["events_total"] += 1
        if event.event_type == "Pass":
            metrics["passes"] += 1
            if event.outcome_name is None:
                metrics["passes_completed"] += 1
                if _is_progressive(event):
                    metrics["progressive_passes"] += 1
        elif event.event_type == "Carry":
            metrics["carries"] += 1
            if _is_progressive(event):
                metrics["progressive_carries"] += 1
        elif event.event_type == "Pressure":
            metrics["pressures"] += 1
        elif event.event_type == "Ball Recovery":
            metrics["ball_recoveries"] += 1
        elif event.event_type == "Interception":
            metrics["interceptions"] += 1
        elif event.event_type == "Duel" and event.event_subtype == "Tackle":
            metrics["tackles"] += 1
        elif event.event_type == "Shot":
            metrics["shots"] += 1
            metrics["expected_goals"] += event.expected_goals or 0
        elif event.event_type == "Dribble" and event.outcome_name == "Complete":
            metrics["dribbles_completed"] += 1
        elif event.event_type == "Clearance":
            metrics["clearances"] += 1

    profiles: list[PlayerSeasonProfile] = []
    evidence: list[PlayerMetricEvidence] = []
    season_reference = (
        f"statsbomb:competitions/{competition.competition_id}/seasons/{competition.season_id}"
    )

    for player_id, player_participations in sorted(participation_groups.items()):
        representative = player_participations[0]
        minutes = round(
            sum(item.minutes_played for item in player_participations),
            3,
        )
        position_minutes: defaultdict[str, float] = defaultdict(float)
        for item in player_participations:
            position_minutes[item.position_group] += item.minutes_played
        dominant_group = max(position_minutes, key=position_minutes.__getitem__)

        team_minutes: defaultdict[tuple[int, str], float] = defaultdict(float)
        for item in player_participations:
            team_minutes[(item.team_id, item.team_name)] += item.minutes_played
        ordered_teams = sorted(
            team_minutes,
            key=lambda team: (-team_minutes[team], team[1]),
        )
        primary_team_name = ordered_teams[0][1]
        team_names = [team_name for _, team_name in ordered_teams]

        counts = event_counts[player_id]
        structured_features: dict[str, float] = {
            "appearances": float(len(player_participations)),
            "starts": float(sum(item.started for item in player_participations)),
            "teams_count": float(len(team_names)),
            "events_total": float(counts.get("events_total", 0)),
        }
        structured_features.update(
            {metric_name: float(counts.get(metric_name, 0)) for metric_name in RAW_METRICS}
        )

        completeness_checks = (
            minutes > 0,
            bool(player_participations),
            counts.get("events_total", 0) > 0,
        )
        provisional_quality = round(
            0.75 * (sum(completeness_checks) / len(completeness_checks)),
            3,
        )
        profile = PlayerSeasonProfile(
            player_id=str(player_id),
            player_name=representative.player_name,
            team_name=primary_team_name,
            team_names=team_names,
            competition_name=competition.competition_name,
            season_name=competition.season_name,
            position_group=dominant_group,
            minutes_played=minutes,
            structured_features=structured_features,
            percentiles={},
            profile_text=(
                f"{representative.player_name} | {' / '.join(team_names)} | "
                f"{competition.competition_name} {competition.season_name} | "
                f"{dominant_group} | {minutes:.1f} minutes | raw event counts only."
            ),
            data_quality=provisional_quality,
        )
        profiles.append(profile)

        for metric_name, raw_value in {
            "minutes_played": minutes,
            **structured_features,
        }.items():
            evidence.append(
                PlayerMetricEvidence(
                    player_id=str(player_id),
                    season_id=str(competition.season_id),
                    metric_name=metric_name,
                    raw_value=float(raw_value),
                    normalized_value=None,
                    percentile=None,
                    comparison_group=(
                        f"{competition.competition_name} {competition.season_name} {dominant_group}"
                    ),
                    sample_size=minutes,
                    source_reference=f"{season_reference}/players/{player_id}/{metric_name}",
                )
            )

    return SeasonAggregation(profiles=profiles, evidence=evidence)


def _is_progressive(event: NormalizedEvent) -> bool:
    """Return whether an action reduces straight-line goal distance by ten units."""
    if (
        event.location_x is None
        or event.location_y is None
        or event.end_location_x is None
        or event.end_location_y is None
    ):
        return False
    start_x = event.location_x
    start_y = event.location_y
    end_x = event.end_location_x
    end_y = event.end_location_y
    start_distance = ((120 - start_x) ** 2 + (40 - start_y) ** 2) ** 0.5
    end_distance = ((120 - end_x) ** 2 + (40 - end_y) ** 2) ** 0.5
    return bool(start_distance - end_distance >= 10)
