"""Phase 3 feature, percentile, and evidence-quality tests."""

from datetime import date

from scoutrag.data.feature_engineering import (
    FeatureEngineeringConfig,
    engineer_player_features,
)
from scoutrag.data.models import CompetitionSeason, MatchRecord
from scoutrag.domain.player import PlayerSeasonProfile


def competition() -> CompetitionSeason:
    return CompetitionSeason(
        competition_id=9,
        season_id=281,
        country_name="Germany",
        competition_name="1. Bundesliga",
        season_name="2023/2024",
        competition_gender="male",
        source_reference="statsbomb:competitions/9/seasons/281",
    )


def match(match_id: int, opponent: str) -> MatchRecord:
    return MatchRecord(
        match_id=match_id,
        competition_id=9,
        season_id=281,
        match_date=date(2024, 1, match_id),
        match_week=match_id,
        home_team_id=1,
        home_team_name="Bayern Munich",
        away_team_id=match_id + 10,
        away_team_name=opponent,
        home_score=2,
        away_score=0,
        duration_seconds=5_400,
        source_reference=f"statsbomb:matches/{match_id}",
    )


def profile(player_id: int, passes: float, *, team: str = "Bayern Munich") -> PlayerSeasonProfile:
    raw_features = {
        "appearances": 10,
        "starts": 10,
        "teams_count": 1,
        "events_total": passes,
        "passes": passes,
        "passes_completed": passes * 0.8,
        "progressive_passes": passes * 0.1,
        "carries": 20,
        "progressive_carries": 5,
        "pressures": 40,
        "ball_recoveries": 30,
        "interceptions": 10,
        "tackles": 12,
        "shots": 3,
        "expected_goals": 0.7,
        "dribbles_completed": 4,
        "clearances": 8,
    }
    return PlayerSeasonProfile(
        player_id=str(player_id),
        player_name=f"Bayern Player {player_id}",
        team_name=team,
        team_names=[team],
        competition_name="1. Bundesliga",
        season_name="2023/2024",
        position_group="central_midfield",
        minutes_played=900,
        structured_features=raw_features,
        percentiles={},
        profile_text="Raw profile.",
        data_quality=0.5,
    )


def test_engineers_per_90_percentiles_quality_and_metric_evidence() -> None:
    result = engineer_player_features(
        competition(),
        [
            match(1, "Borussia Dortmund"),
            match(2, "RB Leipzig"),
            match(3, "VfB Stuttgart"),
        ],
        [profile(1, 300), profile(2, 500), profile(3, 700)],
        [],
    )

    middle = result.profiles[1]
    assert middle.structured_features["passes_per_90"] == 50
    assert middle.structured_features["pass_completion_rate"] == 80
    assert middle.percentiles["passes_per_90"] == 50
    assert middle.data_quality == 1
    assert "Bayern Munich" in middle.profile_text
    assert "Evidence Quality Score 1.000" in middle.profile_text
    assert len(result.definitions) == 13
    metric = next(
        item
        for item in result.evidence
        if item.player_id == "2" and item.metric_name == "passes_per_90"
    )
    assert metric.raw_value == 500
    assert metric.normalized_value == 50
    assert metric.percentile == 50
    assert metric.sample_size == 900


def test_withholds_percentiles_for_partial_source_coverage() -> None:
    result = engineer_player_features(
        competition(),
        [
            match(1, "Bayer Leverkusen"),
            match(2, "RB Leipzig"),
            match(3, "VfB Stuttgart"),
        ],
        [
            profile(1, 300),
            profile(2, 500),
            profile(3, 700),
            profile(4, 600, team="Bayer Leverkusen"),
        ],
        [],
    )

    partial = result.profiles[3]
    assert partial.structured_features["source_coverage_ratio"] == 0.3333
    assert partial.percentiles == {}
    assert partial.data_quality < 1
    assert "source coverage 0.33 below 0.80" in partial.profile_text


def test_feature_thresholds_are_validated() -> None:
    try:
        FeatureEngineeringConfig(minimum_source_coverage=0)
    except ValueError as error:
        assert "minimum_source_coverage" in str(error)
    else:
        raise AssertionError("invalid source-coverage threshold was accepted")
