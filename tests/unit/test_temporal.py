"""Deterministic form and historical fallback tests."""

from datetime import date, timedelta

from scoutrag.data.temporal import build_recent_form, build_season_trends
from scoutrag.domain.player import (
    PlayerMatchPerformance,
    PlayerSeasonProfile,
    TrendDirection,
)


def _performance(index: int, *, passes: float) -> PlayerMatchPerformance:
    fixture_id = 1000 + index
    return PlayerMatchPerformance(
        performance_id=f"api-football:78:2025:1:fixture:{fixture_id}",
        player_id="api-football:1",
        profile_id="api-football:78:2025:1",
        season_id="api-football:78:2025",
        season_name="2025/2026",
        competition_name="Bundesliga",
        fixture_id=fixture_id,
        match_date=date(2025, 8, 1) + timedelta(days=index * 7),
        team_id=157,
        team_name="Bayern München",
        opponent_id=168,
        opponent_name="Opponent",
        home_away="home",
        position_group="midfielder",
        minutes_played=90,
        started=True,
        substitute=False,
        captain=False,
        structured_features={
            "passes": passes,
            "passes_per_90": passes,
            "passes_completed": passes * 0.9,
            "passes_completed_per_90": passes * 0.9,
            "pass_completion_rate": 90,
        },
        data_quality=1,
        source_reference=f"test:{fixture_id}",
    )


def _profile(season: str, value: float, percentile: float) -> PlayerSeasonProfile:
    start = int(season[:4])
    return PlayerSeasonProfile(
        player_id="api-football:1",
        profile_id=f"api-football:78:{start}:1",
        player_name="Test Player",
        team_name="Bayern München",
        team_names=["Bayern München"],
        competition_name="Bundesliga",
        season_name=season,
        position_group="midfielder",
        minutes_played=1800,
        structured_features={"passes_per_90": value},
        percentiles={"passes_per_90": percentile},
        profile_text=f"Test Player | Bundesliga {season}",
        data_quality=0.95,
    )


def test_recent_form_uses_latest_five_and_prior_same_season_baseline() -> None:
    performances = [_performance(index, passes=40 if index < 2 else 80) for index in range(7)]

    form = build_recent_form(performances)[0]

    assert form.matches_in_window == 5
    assert form.minutes_in_window == 450
    assert form.fixture_ids == [1002, 1003, 1004, 1005, 1006]
    assert form.recent_features["passes_per_90"] == 80
    assert form.baseline_features["passes_per_90"] == 40
    assert form.relative_changes["passes_per_90"] == 1
    assert form.data_quality == 1
    assert form.limitations == []


def test_season_trend_keeps_values_separate_and_marks_historical_fallback() -> None:
    profiles = [
        _profile("2023/2024", 55, 50),
        _profile("2024/2025", 65, 62),
        _profile("2025/2026", 80, 75),
    ]

    trend = build_season_trends(profiles)[0]

    assert trend.current_profile_id == "api-football:78:2025:1"
    assert trend.direction is TrendDirection.IMPROVING
    assert trend.latest_value == 80
    assert trend.previous_value == 65
    assert trend.historical_fallback_available is True
    assert [item.season_name for item in trend.observations] == [
        "2023/2024",
        "2024/2025",
        "2025/2026",
    ]
