"""Deterministic answers obey governance and cite stored evidence only."""

from scoutrag.answering.templates import TemplateAnswerGenerator
from scoutrag.domain.evidence import (
    EvidenceVerdict,
    RecommendationEvidencePack,
    RecommendationGovernance,
    RuntimeMetrics,
)
from scoutrag.domain.player import PlayerMetricEvidence, PlayerSeasonProfile
from scoutrag.domain.query import QueryIntent, QueryProfile
from scoutrag.domain.retrieval import (
    CandidateRetrievalTrace,
    RankedPlayerCandidate,
    RetrievalTrace,
)


def test_sufficient_answer_uses_stored_metric_value_and_percentile() -> None:
    pack = evidence_pack(EvidenceVerdict.SUFFICIENT)

    answer = TemplateAnswerGenerator().generate(pack)

    assert answer.verdict is EvidenceVerdict.SUFFICIENT
    assert answer.cited_player_ids == ["5579"]
    assert "pressures per 90 14.3 P91" in answer.text
    assert "Evidence Quality Score: 0.900" in answer.text


def test_insufficient_answer_never_cites_available_candidate() -> None:
    pack = evidence_pack(EvidenceVerdict.INSUFFICIENT)

    answer = TemplateAnswerGenerator().generate(pack)

    assert answer.cited_player_ids == []
    assert "Keine belastbare Spielerempfehlung" in answer.text
    assert "comparison group" in answer.text


def evidence_pack(verdict: EvidenceVerdict) -> RecommendationEvidencePack:
    query = QueryProfile(
        original_query="Pressingstarker Sechser",
        normalized_query="pressingstarker sechser",
        intent=QueryIntent.PLAYER_DISCOVERY,
        requested_positions=["defensive_midfield"],
        requested_metrics=["pressures_per_90"],
        result_count=1,
    )
    profile = PlayerSeasonProfile(
        player_id="5579",
        player_name="Joshua Kimmich",
        team_name="Bayern Munich",
        team_names=["Bayern Munich"],
        competition_name="1. Bundesliga",
        season_name="2023/2024",
        position_group="defensive_midfield",
        minutes_played=1_200,
        structured_features={"pressures_per_90": 14.3},
        percentiles={"pressures_per_90": 91},
        profile_text="Joshua Kimmich profile",
        data_quality=0.9,
    )
    candidate = RankedPlayerCandidate(
        profile=profile,
        retrieval_trace=CandidateRetrievalTrace(
            player_id="5579",
            retrieved_by=["structured", "dense"],
            fused_score=0.9,
        ),
        rank=1,
    )
    metric = PlayerMetricEvidence(
        player_id="5579",
        season_id="281",
        metric_name="pressures_per_90",
        raw_value=190,
        normalized_value=14.3,
        percentile=91,
        comparison_group="Bundesliga midfielders",
        sample_size=1_200,
        source_reference="test:5579",
    )
    governance = RecommendationGovernance(
        verdict=verdict,
        evidence_quality_score=0.9 if verdict is EvidenceVerdict.SUFFICIENT else 0.3,
        reasons=["test verdict"],
        missing_evidence=(
            ["No valid comparison group."] if verdict is EvidenceVerdict.INSUFFICIENT else []
        ),
    )
    return RecommendationEvidencePack(
        query_profile=query,
        governance=governance,
        candidates=[candidate],
        retrieval_trace=RetrievalTrace(
            query_id="query-1",
            query_intent=query.intent.value,
            candidates_before_reranking=1,
            candidates_after_reranking=1,
        ),
        metric_evidence={"5579": [metric]},
        runtime_metrics=RuntimeMetrics(total_ms=1),
    )
