"""Timed orchestration of query analysis, broad recall, fusion, and reranking."""

from collections.abc import Callable
from time import perf_counter
from typing import TypeVar
from uuid import uuid4

from scoutrag.application.noop import NoOpPlayerReranker
from scoutrag.domain.query import QueryProfile
from scoutrag.domain.retrieval import HybridRetrievalResult, RetrievalTrace
from scoutrag.ports.query_analysis import QueryAnalyzer
from scoutrag.ports.reranking import PlayerReranker
from scoutrag.ports.retrieval import PlayerRetriever, RetrievalFusion

T = TypeVar("T")


class HybridRetrievalPipeline:
    """Run Phase 4 while keeping every strategy and timing auditable."""

    def __init__(
        self,
        query_analyzer: QueryAnalyzer,
        retrievers: tuple[PlayerRetriever, ...],
        retrieval_fusion: RetrievalFusion,
        *,
        player_reranker: PlayerReranker | None = None,
        candidate_pool_size: int = 40,
    ) -> None:
        if candidate_pool_size < 1:
            raise ValueError("candidate_pool_size must be positive")
        strategy_names = [retriever.strategy_name for retriever in retrievers]
        if len(strategy_names) != len(set(strategy_names)):
            raise ValueError("retrieval strategy names must be unique")
        self.query_analyzer = query_analyzer
        self.retrievers = retrievers
        self.retrieval_fusion = retrieval_fusion
        self.player_reranker = player_reranker or NoOpPlayerReranker()
        self.candidate_pool_size = candidate_pool_size

    def search(
        self,
        query: str,
        *,
        result_count: int | None = None,
    ) -> HybridRetrievalResult:
        started = perf_counter()
        query_profile, query_analysis_ms = _timed(lambda: self.query_analyzer.analyze(query))
        if result_count is not None:
            if not 1 <= result_count <= 100:
                raise ValueError("result_count must be between 1 and 100")
            query_profile = query_profile.model_copy(update={"result_count": result_count})
        retrieval_limit = max(self.candidate_pool_size, query_profile.result_count)

        candidates_by_strategy = {}
        stage_timings = {"query_analysis": query_analysis_ms}
        for retriever in self.retrievers:
            retrieval_started = perf_counter()
            candidates = retriever.retrieve(
                query_profile,
                limit=retrieval_limit,
            )
            candidates_by_strategy[retriever.strategy_name] = candidates
            stage_timings[f"{retriever.strategy_name}_retrieval"] = round(
                (perf_counter() - retrieval_started) * 1_000,
                3,
            )

        fused, fusion_ms = _timed(
            lambda: self.retrieval_fusion.fuse(
                query_profile,
                candidates_by_strategy,
                limit=retrieval_limit,
            )
        )
        ranked, reranking_ms = _timed(lambda: self.player_reranker.rerank(query_profile, fused))
        ranked = ranked[: query_profile.result_count]
        stage_timings["fusion"] = fusion_ms
        stage_timings["reranking"] = reranking_ms
        stage_timings["total"] = round((perf_counter() - started) * 1_000, 3)

        return HybridRetrievalResult(
            query_profile=query_profile,
            broad_candidates=fused,
            candidates=ranked,
            retrieval_trace=RetrievalTrace(
                query_id=str(uuid4()),
                query_intent=query_profile.intent.value,
                strategies_used=[
                    strategy
                    for strategy, candidates in candidates_by_strategy.items()
                    if candidates
                ],
                candidates_per_strategy={
                    strategy: len(candidates)
                    for strategy, candidates in candidates_by_strategy.items()
                },
                candidates_before_reranking=len(fused),
                candidates_after_reranking=len(ranked),
                filters_applied=_filters(query_profile),
                stage_timings_ms=stage_timings,
            ),
        )


def _timed(operation: Callable[[], T]) -> tuple[T, float]:
    started = perf_counter()
    result = operation()
    return result, round((perf_counter() - started) * 1_000, 3)


def _filters(query: QueryProfile) -> dict[str, object]:
    return {
        "positions": query.requested_positions,
        "teams": query.team_filters,
        "competitions": query.competition_filters,
        "seasons": query.season_filters,
        "minimum_minutes": query.minimum_minutes,
        "requested_metrics": query.requested_metrics,
    }
