"""Isolated before/after evaluation of a PlayerReranker."""

from time import perf_counter

from scoutrag.evaluation.metrics import evaluate_ranking, mean_metrics
from scoutrag.evaluation.models import (
    EvaluationReport,
    GoldenDataset,
    LatencyStats,
    QueryEvaluation,
    RerankingComparisonReport,
    RerankingDelta,
)
from scoutrag.ports.reranking import PlayerReranker
from scoutrag.retrieval.pipeline import HybridRetrievalPipeline


class RerankingEvaluator:
    """Compare fused and reranked order over identical broad candidate pools."""

    def __init__(self, k_values: tuple[int, ...] = (1, 5, 10)) -> None:
        if not k_values or any(k < 1 for k in k_values):
            raise ValueError("k_values must contain positive integers")
        self.k_values = tuple(sorted(set(k_values)))

    def compare(
        self,
        retrieval_pipeline: HybridRetrievalPipeline,
        reranker: PlayerReranker,
        dataset: GoldenDataset,
        *,
        model_name: str,
        backend: str,
    ) -> RerankingComparisonReport:
        baseline_queries: list[QueryEvaluation] = []
        reranked_queries: list[QueryEvaluation] = []

        for golden_query in dataset.queries:
            result = retrieval_pipeline.search(golden_query.query)
            relevance = {
                judgment.player_id: judgment.relevance for judgment in golden_query.judgments
            }
            broad_ids = [candidate.profile.player_id for candidate in result.broad_candidates]
            result_limit = result.query_profile.result_count
            baseline_ids = broad_ids[:result_limit]

            reranking_started = perf_counter()
            reranked_candidates = reranker.rerank(
                result.query_profile,
                result.broad_candidates,
            )
            reranking_ms = round((perf_counter() - reranking_started) * 1_000, 3)
            reranked_ids = [
                candidate.profile.player_id for candidate in reranked_candidates[:result_limit]
            ]

            baseline_queries.append(
                self._query_evaluation(
                    golden_query.query_id,
                    golden_query.query,
                    relevance,
                    broad_ids,
                    baseline_ids,
                    result.retrieval_trace.stage_timings_ms.get("reranking", 0),
                )
            )
            reranked_queries.append(
                self._query_evaluation(
                    golden_query.query_id,
                    golden_query.query,
                    relevance,
                    broad_ids,
                    reranked_ids,
                    reranking_ms,
                )
            )

        baseline = self._report(
            "hybrid_fused_order",
            dataset.schema_version,
            baseline_queries,
        )
        reranked_report = self._report(
            "hybrid_plus_cross_encoder",
            dataset.schema_version,
            reranked_queries,
        )
        return RerankingComparisonReport(
            dataset_version=dataset.schema_version,
            model_name=model_name,
            backend=backend,
            baseline=baseline,
            reranked=reranked_report,
            delta=RerankingDelta(
                mean_reciprocal_rank=_rounded(
                    reranked_report.aggregate.mean_reciprocal_rank
                    - baseline.aggregate.mean_reciprocal_rank
                ),
                ndcg_at_k={
                    k: _rounded(
                        reranked_report.aggregate.at_k[k].ndcg - baseline.aggregate.at_k[k].ndcg
                    )
                    for k in self.k_values
                },
                hit_rate_at_k={
                    k: _rounded(
                        reranked_report.aggregate.at_k[k].hit_rate
                        - baseline.aggregate.at_k[k].hit_rate
                    )
                    for k in self.k_values
                },
            ),
            latency=_latency_stats([query.reranking_ms for query in reranked_queries]),
        )

    def _query_evaluation(
        self,
        query_id: str,
        query: str,
        relevance: dict[str, int],
        broad_ids: list[str],
        ranked_ids: list[str],
        reranking_ms: float,
    ) -> QueryEvaluation:
        return QueryEvaluation(
            query_id=query_id,
            query=query,
            relevant_player_ids=list(relevance),
            broad_candidate_ids=broad_ids,
            ranked_player_ids=ranked_ids,
            reranking_ms=reranking_ms,
            metrics=evaluate_ranking(
                ranked_ids,
                broad_ids,
                relevance,
                k_values=self.k_values,
            ),
        )

    def _report(
        self,
        variant_name: str,
        dataset_version: str,
        queries: list[QueryEvaluation],
    ) -> EvaluationReport:
        return EvaluationReport(
            variant_name=variant_name,
            dataset_version=dataset_version,
            query_count=len(queries),
            k_values=list(self.k_values),
            aggregate=mean_metrics(
                [query.metrics for query in queries],
                k_values=self.k_values,
            ),
            queries=queries,
        )


def _latency_stats(values: list[float]) -> LatencyStats:
    if not values:
        raise ValueError("cannot summarize empty latency values")
    ordered = sorted(values)
    return LatencyStats(
        mean_ms=round(sum(ordered) / len(ordered), 3),
        p50_ms=round(_percentile(ordered, 0.50), 3),
        p95_ms=round(_percentile(ordered, 0.95), 3),
        minimum_ms=round(ordered[0], 3),
        maximum_ms=round(ordered[-1], 3),
    )


def _percentile(ordered: list[float], quantile: float) -> float:
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * quantile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _rounded(value: float) -> float:
    return round(value, 6)
