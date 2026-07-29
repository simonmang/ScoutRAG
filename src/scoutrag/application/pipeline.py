"""Dependency-injection contract for the staged ScoutRAG pipeline."""

from dataclasses import dataclass

from scoutrag.ports.answering import AnswerGenerator
from scoutrag.ports.governance import RecommendationGovernor
from scoutrag.ports.query_analysis import QueryAnalyzer
from scoutrag.ports.reranking import PlayerReranker
from scoutrag.ports.retrieval import PlayerRetriever, RetrievalFusion


@dataclass(frozen=True, slots=True)
class PipelineComponents:
    """Explicit component graph; no role is silently replaced by another."""

    query_analyzer: QueryAnalyzer
    candidate_retrievers: tuple[PlayerRetriever, ...]
    retrieval_fusion: RetrievalFusion
    player_reranker: PlayerReranker
    recommendation_governor: RecommendationGovernor
    answer_generator: AnswerGenerator | None = None
