"""Evidence governance and pack invariants."""

import pytest
from pydantic import ValidationError

from scoutrag.application.noop import NoOpPlayerReranker
from scoutrag.domain.evidence import (
    EvidenceVerdict,
    RecommendationEvidencePack,
    RecommendationGovernance,
    RuntimeMetrics,
)
from scoutrag.domain.player import PlayerMetricEvidence
from scoutrag.domain.query import QueryIntent, QueryProfile
from scoutrag.domain.retrieval import PlayerCandidate, RetrievalTrace


def discovery_query() -> QueryProfile:
    return QueryProfile(
        original_query="Pressingstarker Sechser",
        normalized_query="pressingstarker sechser",
        intent=QueryIntent.PLAYER_DISCOVERY,
        requested_positions=["defensive_midfield"],
        requested_traits=["pressing"],
        requested_metrics=["pressures_per_90"],
        expected_evidence_types=["player_metric"],
    )


def test_non_sufficient_verdict_requires_explanation() -> None:
    with pytest.raises(ValidationError):
        RecommendationGovernance(
            verdict=EvidenceVerdict.INSUFFICIENT,
            evidence_quality_score=0.2,
        )


def test_noop_reranker_preserves_fused_order(
    player_candidate: PlayerCandidate,
) -> None:
    ranked = NoOpPlayerReranker().rerank(discovery_query(), [player_candidate])

    assert ranked[0].rank == 1
    assert ranked[0].reranker_score is None
    assert ranked[0].profile.player_id == player_candidate.profile.player_id


def test_evidence_pack_is_llm_independent_and_serializable(
    player_candidate: PlayerCandidate,
) -> None:
    query = discovery_query()
    ranked = NoOpPlayerReranker().rerank(query, [player_candidate])
    player_id = player_candidate.profile.player_id
    governance = RecommendationGovernance(
        verdict=EvidenceVerdict.SUFFICIENT,
        evidence_quality_score=0.87,
        reasons=["Requested metric and comparison group are available."],
    )
    evidence = PlayerMetricEvidence(
        player_id=player_id,
        season_id="season-2025",
        metric_name="pressures_per_90",
        raw_value=14.3,
        normalized_value=1.4,
        percentile=91,
        comparison_group="Bundesliga central midfielders",
        sample_size=1_420,
        source_reference="statsbomb:competition/9/season/281",
    )

    pack = RecommendationEvidencePack(
        query_profile=query,
        governance=governance,
        candidates=ranked,
        retrieval_trace=RetrievalTrace(
            query_id="query-1",
            query_intent=query.intent.value,
            strategies_used=["sparse", "structured"],
            candidates_per_strategy={"sparse": 31, "structured": 18},
            candidates_before_reranking=40,
            candidates_after_reranking=1,
            filters_applied={"position_group": "defensive_midfield"},
            stage_timings_ms={"retrieval": 12.4, "reranking": 0.1},
        ),
        metric_evidence={player_id: [evidence]},
        runtime_metrics=RuntimeMetrics(total_ms=15.2, candidate_retrieval_ms=12.4),
    )

    payload = pack.model_dump(mode="json")
    assert payload["governance"]["verdict"] == "sufficient"
    assert payload["metric_evidence"][player_id][0]["raw_value"] == 14.3


def test_evidence_for_an_unreturned_player_is_rejected(
    player_candidate: PlayerCandidate,
) -> None:
    query = discovery_query()
    ranked = NoOpPlayerReranker().rerank(query, [player_candidate])
    foreign_evidence = PlayerMetricEvidence(
        player_id="foreign-player",
        season_id="season-1",
        metric_name="pressures_per_90",
        comparison_group="midfielders",
        source_reference="test-source",
    )

    with pytest.raises(ValidationError):
        RecommendationEvidencePack(
            query_profile=query,
            governance=RecommendationGovernance(
                verdict=EvidenceVerdict.LIMITED,
                evidence_quality_score=0.5,
                warnings=["Only one retrieval strategy agreed."],
            ),
            candidates=ranked,
            retrieval_trace=RetrievalTrace(
                query_id="query-2",
                query_intent=query.intent.value,
                candidates_before_reranking=1,
                candidates_after_reranking=1,
            ),
            metric_evidence={"foreign-player": [foreign_evidence]},
            runtime_metrics=RuntimeMetrics(total_ms=1),
        )
