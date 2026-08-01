"""LLM-free retrieval, evidence selection, governance, and runtime assembly."""

from scoutrag.domain.player import PlayerMetricEvidence, PlayerSeasonProfile
from scoutrag.governance.factory import build_governed_pipeline


def test_governed_pipeline_returns_auditable_evidence_pack() -> None:
    profile = PlayerSeasonProfile(
        player_id="5579",
        player_name="Joshua Kimmich",
        team_name="Bayern Munich",
        team_names=["Bayern Munich"],
        competition_name="1. Bundesliga",
        season_name="2023/2024",
        position_group="defensive_midfield",
        minutes_played=180,
        structured_features={
            "pressures_per_90": 9,
            "source_coverage_ratio": 0.1,
            "feature_coverage_ratio": 1,
            "comparison_group_size": 0,
        },
        percentiles={},
        profile_text="Joshua Kimmich | Bayern Munich | partial source profile",
        data_quality=0.5,
    )
    evidence = PlayerMetricEvidence(
        player_id="5579",
        season_id="281",
        metric_name="pressures_per_90",
        raw_value=18,
        normalized_value=9,
        percentile=None,
        comparison_group="Bundesliga defensive midfield n=0",
        sample_size=180,
        source_reference="statsbomb:test",
    )

    pack = build_governed_pipeline([profile], [evidence]).search(
        "Zeige das Profil von Joshua Kimmich"
    )

    assert pack.query_profile.named_players == ["Joshua Kimmich"]
    assert pack.candidates[0].profile.team_name == "Bayern Munich"
    assert pack.governance.verdict.value == "limited"
    assert pack.metric_evidence["5579"] == [evidence]
    assert pack.limitations == pack.governance.warnings
    assert pack.runtime_metrics.total_ms >= pack.runtime_metrics.governance_ms
    assert pack.retrieval_trace.candidates_before_reranking == 1


def test_same_player_in_two_leagues_keeps_evidence_separate() -> None:
    profiles = [
        PlayerSeasonProfile(
            player_id="api-football:1",
            profile_id=f"api-football:{league}:2025:1",
            player_name="Transfer Player",
            team_name=team,
            team_names=[team],
            competition_name=competition,
            season_name="2025/2026",
            position_group="midfielder",
            minutes_played=minutes,
            structured_features={"passes_per_90": value},
            percentiles={"passes_per_90": value},
            profile_text=f"Transfer Player | {team} | {competition}",
            data_quality=0.9,
        )
        for league, team, competition, minutes, value in [
            (39, "English Club", "Premier League", 900, 80),
            (135, "Italian Club", "Serie A", 450, 60),
        ]
    ]
    evidence = [
        PlayerMetricEvidence(
            player_id=profile.player_id,
            profile_id=profile.profile_id,
            season_id=profile.profile_id.rsplit(":", 1)[0],
            metric_name="passes_per_90",
            raw_value=value,
            normalized_value=value,
            percentile=value,
            comparison_group=profile.competition_name,
            sample_size=profile.minutes_played,
            source_reference=f"test:{profile.profile_id}",
        )
        for profile, value in zip(profiles, [80, 60], strict=True)
    ]

    pack = build_governed_pipeline(profiles, evidence).search(
        "Zeige das Profil von Transfer Player"
    )

    assert set(pack.metric_evidence) == {
        "api-football:39:2025:1",
        "api-football:135:2025:1",
    }
    assert pack.metric_evidence["api-football:39:2025:1"][0].normalized_value == 80
    assert pack.metric_evidence["api-football:135:2025:1"][0].normalized_value == 60
