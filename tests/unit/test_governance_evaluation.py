"""False-recommendation, abstention, coverage, and selective metrics."""

from scoutrag.domain.evidence import (
    EvidenceVerdict,
    RecommendationEvidencePack,
    RecommendationGovernance,
    RuntimeMetrics,
)
from scoutrag.domain.query import QueryIntent, QueryProfile
from scoutrag.domain.retrieval import RetrievalTrace
from scoutrag.evaluation.governance import (
    GovernanceEvaluator,
    GovernanceGoldenCase,
    GovernanceGoldenDataset,
)


def pack(verdict: EvidenceVerdict) -> RecommendationEvidencePack:
    query = QueryProfile(
        original_query="test query",
        normalized_query="test query",
        intent=QueryIntent.PLAYER_DISCOVERY,
    )
    return RecommendationEvidencePack(
        query_profile=query,
        governance=RecommendationGovernance(
            verdict=verdict,
            evidence_quality_score=0.5,
            reasons=["test decision"],
        ),
        candidates=[],
        retrieval_trace=RetrievalTrace(
            query_id="query",
            query_intent=query.intent.value,
            candidates_before_reranking=0,
            candidates_after_reranking=0,
        ),
        runtime_metrics=RuntimeMetrics(total_ms=1),
    )


def case(
    case_id: str,
    expected: EvidenceVerdict,
) -> GovernanceGoldenCase:
    return GovernanceGoldenCase(
        case_id=case_id,
        query=case_id,
        expected_verdict=expected,
        should_abstain=expected
        in {
            EvidenceVerdict.INSUFFICIENT,
            EvidenceVerdict.CONFLICTING,
            EvidenceVerdict.OUT_OF_SCOPE,
        },
        rationale="metric fixture",
    )


def test_governance_safety_metrics_keep_false_recommendations_visible() -> None:
    dataset = GovernanceGoldenDataset(
        schema_version="test-v1",
        cases=[
            case("unsafe-false-positive", EvidenceVerdict.INSUFFICIENT),
            case("unsafe-correct", EvidenceVerdict.OUT_OF_SCOPE),
            case("limited-correct", EvidenceVerdict.LIMITED),
            case("sufficient-missed", EvidenceVerdict.SUFFICIENT),
        ],
    )
    actual = {
        "unsafe-false-positive": EvidenceVerdict.SUFFICIENT,
        "unsafe-correct": EvidenceVerdict.OUT_OF_SCOPE,
        "limited-correct": EvidenceVerdict.LIMITED,
        "sufficient-missed": EvidenceVerdict.LIMITED,
    }

    report = GovernanceEvaluator().evaluate(lambda query: pack(actual[query]), dataset)

    assert report.metrics.false_recommendation_rate == 0.5
    assert report.metrics.abstention_recall == 0.5
    assert report.metrics.abstention_precision == 1
    assert report.metrics.coverage == 0.75
    assert report.metrics.selective_accuracy == 0.333333
    assert report.metrics.limited_case_recall == 1
    assert report.metrics.verdict_accuracy == 0.5
