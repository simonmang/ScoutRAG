"""Intent-aware evidence governance and score-factor tests."""

from dataclasses import replace

import pytest

from scoutrag.domain.evidence import EvidenceVerdict
from scoutrag.domain.player import PlayerMetricEvidence, PlayerSeasonProfile
from scoutrag.domain.query import QueryIntent, QueryProfile
from scoutrag.domain.retrieval import CandidateRetrievalTrace, RankedPlayerCandidate
from scoutrag.governance.rules import (
    GovernanceThresholds,
    RuleBasedRecommendationGovernor,
)

FEATURES = {
    "pressures_per_90": 12.0,
    "passes_per_90": 50.0,
    "pass_completion_rate": 88.0,
    "progressive_passes_per_90": 6.0,
    "ball_recoveries_per_90": 7.0,
    "interceptions_per_90": 2.0,
    "tackles_per_90": 3.0,
    "progressive_carries_per_90": 4.0,
    "dribbles_completed_per_90": 1.0,
    "shots_per_90": 2.0,
    "expected_goals_per_90": 0.2,
    "clearances_per_90": 2.0,
    "duels_won_per_90": 5.0,
}


def candidate(
    player_id: str,
    name: str,
    *,
    fused_score: float,
    season: str = "2023/2024",
    minutes: float = 1_800,
    source_coverage: float = 1,
    features: dict[str, float] | None = None,
) -> RankedPlayerCandidate:
    profile_features = dict(features if features is not None else FEATURES)
    profile_features.update(
        {
            "source_coverage_ratio": source_coverage,
            "feature_coverage_ratio": 1.0,
            "comparison_group_size": 4.0,
        }
    )
    return RankedPlayerCandidate(
        profile=PlayerSeasonProfile(
            player_id=player_id,
            player_name=name,
            team_name="Bayern Munich",
            team_names=["Bayern Munich"],
            competition_name="1. Bundesliga",
            season_name=season,
            position_group="defensive_midfield",
            minutes_played=minutes,
            structured_features=profile_features,
            percentiles={"pressures_per_90": 90},
            profile_text=f"{name} profile",
            data_quality=source_coverage,
        ),
        retrieval_trace=CandidateRetrievalTrace(
            player_id=player_id,
            retrieved_by=["structured", "sparse", "dense"],
            structured_score=fused_score,
            sparse_score=fused_score,
            dense_score=fused_score,
            fused_score=fused_score,
        ),
        rank=1,
    )


def pressing_query(*, seasons: list[str] | None = None) -> QueryProfile:
    return QueryProfile(
        original_query="Top 2 pressingstarke Sechser",
        normalized_query="top 2 pressingstarke sechser",
        intent=QueryIntent.AGGREGATION,
        requested_positions=["defensive_midfield"],
        requested_traits=["pressing"],
        requested_metrics=["pressures_per_90"],
        season_filters=seasons or [],
        result_count=2,
    )


def metric(
    player_id: str,
    *,
    name: str = "pressures_per_90",
    normalized: float = 12,
    percentile: float | None = 90,
    source: str = "source-a",
    season_id: str = "281",
) -> PlayerMetricEvidence:
    return PlayerMetricEvidence(
        player_id=player_id,
        season_id=season_id,
        metric_name=name,
        raw_value=120,
        normalized_value=normalized,
        percentile=percentile,
        comparison_group="Bundesliga defensive midfield n=4",
        sample_size=1_800,
        source_reference=source,
    )


def test_complete_comparable_evidence_is_sufficient() -> None:
    first = candidate("1", "Joshua Kimmich", fused_score=1)
    second = candidate("2", "Aleksandar Pavlović", fused_score=0.7)
    second = second.model_copy(update={"rank": 2})

    governance = RuleBasedRecommendationGovernor().evaluate(
        pressing_query(),
        [first, second],
        {"1": [metric("1")], "2": [metric("2")]},
    )

    assert governance.verdict is EvidenceVerdict.SUFFICIENT
    assert governance.evidence_quality_score == 1
    assert set(governance.factors) == {
        "data_coverage",
        "played_minutes",
        "feature_availability",
        "requested_trait_coverage",
        "retrieval_agreement",
        "ranking_separation",
        "comparison_group_availability",
        "season_consistency",
        "missing_value_completeness",
        "hard_filter_fulfillment",
    }


def test_exact_bayern_profile_with_partial_source_is_limited() -> None:
    partial = candidate(
        "1",
        "Joshua Kimmich",
        fused_score=1,
        minutes=120,
        source_coverage=0.1,
        features={"pressures_per_90": 4},
    )
    query = QueryProfile(
        original_query="Zeige das Profil von Joshua Kimmich",
        normalized_query="zeige das profil von joshua kimmich",
        intent=QueryIntent.EXACT_PLAYER_LOOKUP,
        named_players=["Joshua Kimmich"],
    )

    governance = RuleBasedRecommendationGovernor().evaluate(query, [partial], {})

    assert governance.verdict is EvidenceVerdict.LIMITED
    assert "Source data coverage is limited." in governance.warnings
    assert 0 < governance.evidence_quality_score < 1


def test_missing_requested_metric_and_comparison_group_are_insufficient() -> None:
    query = pressing_query().model_copy(update={"requested_metrics": ["expected_assists_per_90"]})
    result = RuleBasedRecommendationGovernor().evaluate(
        query,
        [
            candidate("1", "Joshua Kimmich", fused_score=1),
            candidate("2", "Aleksandar Pavlović", fused_score=0.7).model_copy(update={"rank": 2}),
        ],
        {},
    )

    assert result.verdict is EvidenceVerdict.INSUFFICIENT
    assert any("expected_assists_per_90" in item for item in result.missing_evidence)
    assert any("comparison group" in item for item in result.missing_evidence)


def test_no_candidates_and_out_of_scope_abstain_immediately() -> None:
    governor = RuleBasedRecommendationGovernor()
    empty = governor.evaluate(pressing_query(), [], {})
    out_of_scope = governor.evaluate(
        QueryProfile(
            original_query="Wer gewinnt die WM?",
            normalized_query="wer gewinnt die wm?",
            intent=QueryIntent.OUT_OF_SCOPE,
        ),
        [],
        {},
    )

    assert empty.verdict is EvidenceVerdict.INSUFFICIENT
    assert empty.evidence_quality_score == 0
    assert out_of_scope.verdict is EvidenceVerdict.OUT_OF_SCOPE
    assert "predictions" in out_of_scope.missing_evidence[0]


def test_multiple_seasons_and_conflicting_values_are_reported() -> None:
    players = [
        candidate("1", "Joshua Kimmich", fused_score=1),
        candidate("2", "Aleksandar Pavlović", fused_score=0.7).model_copy(update={"rank": 2}),
    ]
    season_conflict = RuleBasedRecommendationGovernor().evaluate(
        pressing_query(seasons=["2022/2023", "2023/2024"]),
        players,
        {"1": [metric("1")], "2": [metric("2")]},
    )
    value_conflict = RuleBasedRecommendationGovernor().evaluate(
        pressing_query(),
        players,
        {
            "1": [metric("1", normalized=10), metric("1", normalized=20, source="source-b")],
            "2": [metric("2")],
        },
    )

    assert season_conflict.verdict is EvidenceVerdict.CONFLICTING
    assert "multiple season filters" in season_conflict.reasons[0]
    assert value_conflict.verdict is EvidenceVerdict.CONFLICTING
    assert "Conflicting values" in value_conflict.reasons[0]


def test_metric_evidence_from_multiple_seasons_is_conflicting() -> None:
    players = [
        candidate("1", "Joshua Kimmich", fused_score=1),
        candidate("2", "Aleksandar Pavlović", fused_score=0.7).model_copy(update={"rank": 2}),
    ]

    result = RuleBasedRecommendationGovernor().evaluate(
        pressing_query(),
        players,
        {
            "1": [metric("1"), metric("1", season_id="282")],
            "2": [metric("2")],
        },
    )

    assert result.verdict is EvidenceVerdict.CONFLICTING
    assert "multiple season IDs" in result.reasons[0]


def test_governance_thresholds_reject_invalid_configuration() -> None:
    valid = GovernanceThresholds()
    assert replace(valid, sufficient_score=0.8).sufficient_score == 0.8
    with pytest.raises(ValueError, match="score thresholds"):
        GovernanceThresholds(insufficient_score=0.8, sufficient_score=0.7)
    with pytest.raises(ValueError, match="full_sample_minutes"):
        GovernanceThresholds(full_sample_minutes=0)
