"""Phase 6 compares fused and reranked order over one shared broad pool."""

from collections.abc import Sequence

from scoutrag.domain.player import PlayerSeasonProfile
from scoutrag.domain.query import QueryProfile
from scoutrag.domain.retrieval import CandidateRetrievalTrace, PlayerCandidate
from scoutrag.evaluation.models import GoldenDataset, GoldenJudgment, GoldenQuery
from scoutrag.evaluation.reranking import RerankingEvaluator
from scoutrag.reranking.cross_encoder import CrossEncoderPlayerReranker
from scoutrag.retrieval.fusion import FusionWeights, WeightedRetrievalFusion
from scoutrag.retrieval.pipeline import HybridRetrievalPipeline
from scoutrag.retrieval.query_analysis import RuleBasedQueryAnalyzer


def profile(player_id: str, name: str, pressing: float) -> PlayerSeasonProfile:
    return PlayerSeasonProfile(
        player_id=player_id,
        player_name=name,
        team_name="Bayern Munich",
        team_names=["Bayern Munich"],
        competition_name="1. Bundesliga",
        season_name="2023/2024",
        position_group="defensive_midfield",
        minutes_played=1_000,
        structured_features={"pressures_per_90": pressing},
        percentiles={"pressures_per_90": pressing},
        profile_text=f"{name} | Bayern Munich | pressure percentile {pressing}.",
        data_quality=0.9,
    )


class StaticSparseRetriever:
    strategy_name = "sparse"

    def __init__(self, profiles: list[PlayerSeasonProfile]) -> None:
        self.profiles = profiles

    def retrieve(self, query_profile: QueryProfile, *, limit: int) -> list[PlayerCandidate]:
        del query_profile
        return [
            PlayerCandidate(
                profile=item,
                retrieval_trace=CandidateRetrievalTrace(
                    player_id=item.player_id,
                    retrieved_by=["sparse"],
                    sparse_score=score,
                    fused_score=score,
                ),
            )
            for item, score in zip(self.profiles, (1.0, 0.5), strict=True)
        ][:limit]


class FootballPairModel:
    model_name = "fake-football-reranker"

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        return [1.0 if "Aleksandar" in profile_text else 0.0 for _, profile_text in pairs]


def test_reranking_report_calculates_mrr_ndcg_hit_rate_and_latency() -> None:
    profiles = [
        profile("1", "Joshua Kimmich", 60),
        profile("2", "Aleksandar Pavlović", 95),
    ]
    pipeline = HybridRetrievalPipeline(
        RuleBasedQueryAnalyzer(profiles),
        (StaticSparseRetriever(profiles),),
        WeightedRetrievalFusion(FusionWeights(dense=0, sparse=1, structured=0, exact=0)),
        candidate_pool_size=2,
    )
    dataset = GoldenDataset(
        schema_version="phase6-test-v1",
        name="Phase 6 test",
        competition_id=9,
        season_id=281,
        source_reference="test",
        labeling_method="deterministic",
        queries=[
            GoldenQuery(
                query_id="pressing-six",
                query="pressingstarker Sechser",
                language="de",
                category="trait",
                judgments=[
                    GoldenJudgment(
                        player_id="2",
                        relevance=3,
                        rationale="Highest pressing percentile.",
                    )
                ],
            )
        ],
    )

    report = RerankingEvaluator(k_values=(1, 2)).compare(
        pipeline,
        CrossEncoderPlayerReranker(FootballPairModel()),
        dataset,
        model_name=FootballPairModel.model_name,
        backend="test",
    )

    assert report.backend == "test"
    assert report.baseline.queries[0].broad_candidate_ids == ["1", "2"]
    assert report.reranked.queries[0].broad_candidate_ids == ["1", "2"]
    assert report.baseline.aggregate.mean_reciprocal_rank == 0.5
    assert report.reranked.aggregate.mean_reciprocal_rank == 1
    assert report.delta.mean_reciprocal_rank == 0.5
    assert report.delta.ndcg_at_k[1] == 1
    assert report.delta.hit_rate_at_k[1] == 1
    assert report.reranked.aggregate.at_k[2].hit_rate == 1
    assert report.latency.mean_ms >= 0
