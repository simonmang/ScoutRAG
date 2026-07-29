"""Cross-record validation for season-safe Phase 2 artifacts."""

from collections import Counter

from scoutrag.data.models import (
    CompetitionSeason,
    DataValidationReport,
    MatchRecord,
    NormalizedEvent,
    PlayerMatchParticipation,
)
from scoutrag.domain.player import PlayerMetricEvidence, PlayerSeasonProfile


def validate_dataset(
    competition: CompetitionSeason,
    matches: list[MatchRecord],
    events: list[NormalizedEvent],
    participations: list[PlayerMatchParticipation],
    profiles: list[PlayerSeasonProfile],
    evidence: list[PlayerMetricEvidence],
) -> DataValidationReport:
    """Validate coverage, uniqueness, filters, and season consistency."""
    errors: list[str] = []
    warnings: list[str] = []

    if not matches:
        errors.append("No matches were imported.")
    if not events:
        errors.append("No events were normalized.")
    if not participations:
        errors.append("No positive player-minute records were calculated.")
    if not profiles:
        errors.append("No player-season profiles were generated.")

    event_id_counts = Counter(event.event_id for event in events)
    duplicate_event_ids = sorted(
        event_id for event_id, count in event_id_counts.items() if count > 1
    )
    if duplicate_event_ids:
        errors.append(f"Duplicate event IDs detected: {duplicate_event_ids[:5]}")

    match_ids = {match.match_id for match in matches}
    event_match_ids = {event.match_id for event in events}
    missing_event_match_ids = sorted(match_ids - event_match_ids)
    if missing_event_match_ids:
        errors.append(f"Matches without events: {missing_event_match_ids}")

    participation_match_ids = {item.match_id for item in participations}
    missing_lineup_match_ids = sorted(match_ids - participation_match_ids)
    if missing_lineup_match_ids:
        errors.append(f"Matches without player minutes: {missing_lineup_match_ids}")

    wrong_season_records = (
        sum(
            (
                item.competition_id != competition.competition_id
                or item.season_id != competition.season_id
            )
            for item in matches
        )
        + sum(
            (
                item.competition_id != competition.competition_id
                or item.season_id != competition.season_id
            )
            for item in events
        )
        + sum(
            (
                item.competition_id != competition.competition_id
                or item.season_id != competition.season_id
            )
            for item in participations
        )
    )
    if wrong_season_records:
        errors.append(f"{wrong_season_records} records violate competition-season consistency.")

    durations_by_match = {match.match_id: (match.duration_seconds / 60) + 2 for match in matches}
    excessive_minutes = [
        f"{item.match_id}:{item.player_id}"
        for item in participations
        if item.minutes_played > durations_by_match.get(item.match_id, 0)
    ]
    if excessive_minutes:
        errors.append(f"Player minutes exceed observed match duration: {excessive_minutes[:5]}")

    profile_ids = [profile.player_id for profile in profiles]
    duplicate_profile_ids = sorted(
        player_id for player_id, count in Counter(profile_ids).items() if count > 1
    )
    if duplicate_profile_ids:
        errors.append(f"Duplicate player-season profiles: {duplicate_profile_ids[:5]}")

    invalid_primary_teams = [
        profile.player_id
        for profile in profiles
        if profile.team_names and profile.team_name not in profile.team_names
    ]
    if invalid_primary_teams:
        errors.append(
            f"Primary team is absent from team history for players: {invalid_primary_teams[:5]}"
        )

    profile_id_set = set(profile_ids)
    unknown_evidence_players = sorted({item.player_id for item in evidence} - profile_id_set)
    if unknown_evidence_players:
        errors.append(
            f"Evidence references players without profiles: {unknown_evidence_players[:5]}"
        )

    wrong_evidence_seasons = sum(item.season_id != str(competition.season_id) for item in evidence)
    if wrong_evidence_seasons:
        errors.append(f"{wrong_evidence_seasons} evidence records use another season.")

    low_minute_profiles = sum(profile.minutes_played < 90 for profile in profiles)
    if low_minute_profiles:
        warnings.append(
            f"{low_minute_profiles} profiles contain fewer than 90 observed minutes; "
            "governance must treat them as limited evidence."
        )
    multi_team_profiles = sum(len(profile.team_names) > 1 for profile in profiles)
    if multi_team_profiles:
        warnings.append(
            f"{multi_team_profiles} player-season profiles aggregate explicitly across "
            "multiple teams; team_names preserves transfer provenance."
        )
    team_match_counts: Counter[int] = Counter()
    for match in matches:
        team_match_counts[match.home_team_id] += 1
        team_match_counts[match.away_team_id] += 1
    sorted_team_coverage = sorted(team_match_counts.values())
    if sorted_team_coverage:
        median_team_coverage = sorted_team_coverage[len(sorted_team_coverage) // 2]
        maximum_team_coverage = sorted_team_coverage[-1]
        if median_team_coverage and maximum_team_coverage >= 3 * median_team_coverage:
            warnings.append(
                "Uneven team coverage detected "
                f"(maximum {maximum_team_coverage}, median {median_team_coverage} matches); "
                "profiles reflect available source coverage, not a complete league season."
            )
    warnings.append(
        "Phase 2 features are raw counts. Per-90 values, percentiles, and final "
        "data-quality scoring are intentionally deferred to Phase 3."
    )

    return DataValidationReport(
        valid=not errors,
        competition_id=competition.competition_id,
        season_id=competition.season_id,
        match_count=len(matches),
        event_count=len(events),
        participation_count=len(participations),
        profile_count=len(profiles),
        evidence_count=len(evidence),
        errors=errors,
        warnings=warnings,
    )
