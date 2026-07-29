"""Composition helper for the default governed hybrid pipeline."""

from scoutrag.application.noop import NoOpPlayerReranker
from scoutrag.domain.player import PlayerMetricEvidence, PlayerSeasonProfile
from scoutrag.governance.evidence import PlayerMetricEvidenceIndex
from scoutrag.governance.pipeline import GovernedRetrievalPipeline
from scoutrag.governance.rules import RuleBasedRecommendationGovernor
from scoutrag.ports.reranking import PlayerReranker
from scoutrag.ports.retrieval import PlayerRetriever
from scoutrag.retrieval.dense import DensePlayerRetriever
from scoutrag.retrieval.exact import ExactPlayerRetriever
from scoutrag.retrieval.fusion import FusionWeights, WeightedRetrievalFusion
from scoutrag.retrieval.pipeline import HybridRetrievalPipeline
from scoutrag.retrieval.query_analysis import RuleBasedQueryAnalyzer
from scoutrag.retrieval.sparse import BM25PlayerRetriever
from scoutrag.retrieval.structured import StructuredFeaturePlayerRetriever


def build_governed_pipeline(
    profiles: list[PlayerSeasonProfile],
    evidence: list[PlayerMetricEvidence],
    *,
    dense_retriever: DensePlayerRetriever | None = None,
    player_reranker: PlayerReranker | None = None,
    candidate_pool_size: int = 40,
) -> GovernedRetrievalPipeline:
    """Compose default recall while keeping ranking and governance injectable."""
    retrievers: list[PlayerRetriever] = [
        ExactPlayerRetriever(profiles),
        StructuredFeaturePlayerRetriever(profiles),
        BM25PlayerRetriever(profiles),
    ]
    weights = FusionWeights()
    if dense_retriever is not None:
        retrievers.append(dense_retriever)
    else:
        available = weights.sparse + weights.structured + weights.exact
        weights = FusionWeights(
            dense=0,
            sparse=weights.sparse / available,
            structured=weights.structured / available,
            exact=weights.exact / available,
        )
    retrieval = HybridRetrievalPipeline(
        RuleBasedQueryAnalyzer(profiles),
        tuple(retrievers),
        WeightedRetrievalFusion(weights),
        player_reranker=player_reranker or NoOpPlayerReranker(),
        candidate_pool_size=candidate_pool_size,
    )
    return GovernedRetrievalPipeline(
        retrieval,
        PlayerMetricEvidenceIndex(evidence),
        RuleBasedRecommendationGovernor(),
    )
