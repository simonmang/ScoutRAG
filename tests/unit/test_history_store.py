"""Multi-season storage keeps current and historical observations separate."""

from datetime import date
from pathlib import Path

from scoutrag.data.api_football_profiles import (
    ApiFootballDatasetWriter,
    ApiFootballProfileResult,
)
from scoutrag.data.history import PlayerHistoryStore
from scoutrag.data.temporal import build_season_trends
from scoutrag.domain.player import (
    MetricDefinition,
    PlayerIdentity,
    PlayerMatchPerformance,
    PlayerMetricEvidence,
    PlayerRecentForm,
    PlayerSeasonProfile,
    PlayerTeamSeasonStint,
)


def test_history_store_returns_current_first_without_averaging(tmp_path: Path) -> None:
    profiles = [_profile(2025, 900, 12, 80), _profile(2024, 1_800, 10, 70)]
    performances = [
        PlayerMatchPerformance(
            performance_id=f"match:{year}",
            player_id="api-football:1",
            profile_id=f"api-football:78:{year}:1",
            season_id=str(year),
            season_name=f"{year}/{year + 1}",
            competition_name="Bundesliga",
            fixture_id=year,
            match_date=date(year + 1, 5, 1),
            team_id=157,
            team_name="Bayern München",
            position_group="central_midfield",
            minutes_played=90,
            structured_features={"passes": float(year - 1900)},
            data_quality=0.9,
            source_reference=f"fixture:{year}",
        )
        for year in (2024, 2025)
    ]
    forms = [
        PlayerRecentForm(
            profile_id=f"api-football:78:{year}:1",
            player_id="api-football:1",
            as_of_date=date(year + 1, 5, 1),
            window_size=5,
            matches_in_window=1,
            minutes_in_window=90,
            fixture_ids=[year],
            recent_features={"passes": 100},
            data_quality=0.5,
        )
        for year in (2024, 2025)
    ]
    result = ApiFootballProfileResult(
        profiles=profiles,
        evidence=[_evidence(profile) for profile in profiles],
        definitions=[
            MetricDefinition(
                metric_name="passes_per_90",
                display_name="Passes per 90",
                description="Stored passes per 90.",
                calculation_method="passes / minutes * 90",
            )
        ],
        limitations=[],
        identities=[
            PlayerIdentity(
                player_id="api-football:1",
                player_name="Test Player",
                source_reference="api-football:/players?id=1",
            )
        ],
        stints=[
            PlayerTeamSeasonStint(
                stint_id=f"stint:{year}",
                player_id="api-football:1",
                profile_id=f"api-football:78:{year}:1",
                season_id=str(year),
                season_name=f"{year}/{year + 1}",
                competition_name="Bundesliga",
                team_id=157,
                team_name="Bayern München",
                position_group="central_midfield",
                minutes_played=minutes,
                appearances=10,
                data_quality=0.9,
                source_reference=f"season:{year}",
            )
            for year, minutes in ((2024, 1_800), (2025, 900))
        ],
        match_performances=performances,
        recent_forms=forms,
        season_trends=build_season_trends(profiles),
    )
    ApiFootballDatasetWriter().write(
        tmp_path,
        result=result,
        league_id=[78],
        season_start_year=2025,
        competition_name="History test",
    )

    context = PlayerHistoryStore(tmp_path).for_player("api-football:1", match_limit=1)

    assert [item.season_name for item in context.season_profiles] == [
        "2025/2026",
        "2024/2025",
    ]
    assert [item.structured_features["passes_per_90"] for item in context.season_profiles] == [
        12,
        10,
    ]
    assert context.season_trends[0].historical_fallback_available is True
    assert context.latest_matches[0].season_name == "2025/2026"


def _profile(year: int, minutes: float, passes: float, percentile: float) -> PlayerSeasonProfile:
    return PlayerSeasonProfile(
        player_id="api-football:1",
        profile_id=f"api-football:78:{year}:1",
        player_name="Test Player",
        team_name="Bayern München",
        team_names=["Bayern München"],
        competition_name="Bundesliga",
        season_name=f"{year}/{year + 1}",
        position_group="central_midfield",
        minutes_played=minutes,
        structured_features={"passes_per_90": passes},
        percentiles={"passes_per_90": percentile},
        profile_text=f"Test Player | {year}",
        data_quality=0.9,
    )


def _evidence(profile: PlayerSeasonProfile) -> PlayerMetricEvidence:
    return PlayerMetricEvidence(
        player_id=profile.player_id,
        profile_id=profile.profile_id,
        season_id=profile.season_name,
        metric_name="passes_per_90",
        normalized_value=profile.structured_features["passes_per_90"],
        percentile=profile.percentiles["passes_per_90"],
        comparison_group="Bundesliga central_midfield",
        sample_size=profile.minutes_played,
        source_reference=f"profile:{profile.profile_id}",
    )
