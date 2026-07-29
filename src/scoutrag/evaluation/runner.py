"""Execute golden-query evaluation and retrieval ablation variants."""

from dataclasses import dataclass

from scoutrag.domain.player import PlayerSeasonProfile
from scoutrag.evaluation.metrics import evaluate_ranking, mean_metrics
from scoutrag.evaluation.models import (
    AblationReport,
    EvaluationReport,
    GoldenDataset,
    QueryEvaluation,
)
from scoutrag.retrieval.dense import DensePlayerRetriever
from scoutrag.retrieval.exact import ExactPlayerRetriever
from scoutrag.retrieval.fusion import FusionWeights, WeightedRetrievalFusion
from scoutrag.retrieval.pipeline import HybridRetrievalPipeline
from scoutrag.retrieval.query_analysis import RuleBasedQueryAnalyzer
from scoutrag.retrieval.sparse import BM25PlayerRetriever
from scoutrag.retrieval.structured import StructuredFeaturePlayerRetriever


class RetrievalEvaluator:
    """Evaluate one configured pipeline against every golden query."""

    def __init__(self, k_values: tuple[int, ...] = (1, 5, 10)) -> None:
        if not k_values or any(k < 1 for k in k_values):
            raise ValueError("k_values must contain positive integers")
        self.k_values = tuple(sorted(set(k_values)))

    def evaluate(
        self,
        pipeline: HybridRetrievalPipeline,
        dataset: GoldenDataset,
        *,
        variant_name: str,
    ) -> EvaluationReport:
        query_evaluations: list[QueryEvaluation] = []
        for golden_query in dataset.queries:
            result = pipeline.search(golden_query.query)
            relevance = {
                judgment.player_id: judgment.relevance for judgment in golden_query.judgments
            }
            broad_ids = [candidate.profile.player_id for candidate in result.broad_candidates]
            ranked_ids = [candidate.profile.player_id for candidate in result.candidates]
            metrics = evaluate_ranking(
                ranked_ids,
                broad_ids,
                relevance,
                k_values=self.k_values,
            )
            query_evaluations.append(
                QueryEvaluation(
                    query_id=golden_query.query_id,
                    query=golden_query.query,
                    relevant_player_ids=list(relevance),
                    broad_candidate_ids=broad_ids,
                    ranked_player_ids=ranked_ids,
                    metrics=metrics,
                )
            )

        aggregate = mean_metrics(
            [evaluation.metrics for evaluation in query_evaluations],
            k_values=self.k_values,
        )
        return EvaluationReport(
            variant_name=variant_name,
            dataset_version=dataset.schema_version,
            query_count=len(query_evaluations),
            k_values=list(self.k_values),
            aggregate=aggregate,
            queries=query_evaluations,
        )


@dataclass(frozen=True, slots=True)
class AblationVariant:
    """One explicit combination of independent recall strategies."""

    name: str
    weights: FusionWeights
    use_exact: bool = False
    use_structured: bool = False
    use_sparse: bool = False
    use_dense: bool = False


DEFAULT_ABLATIONS = (
    AblationVariant(
        name="A_bm25",
        weights=FusionWeights(dense=0, sparse=1, structured=0, exact=0),
        use_sparse=True,
    ),
    AblationVariant(
        name="B_pretrained_bi_encoder",
        weights=FusionWeights(dense=1, sparse=0, structured=0, exact=0),
        use_dense=True,
    ),
    AblationVariant(
        name="C_bm25_plus_bi_encoder",
        weights=FusionWeights(dense=0.5, sparse=0.5, structured=0, exact=0),
        use_sparse=True,
        use_dense=True,
    ),
    AblationVariant(
        name="D_bm25_bi_encoder_structured",
        weights=FusionWeights(dense=0.35, sparse=0.30, structured=0.35, exact=0),
        use_structured=True,
        use_sparse=True,
        use_dense=True,
    ),
    AblationVariant(
        name="H_full_phase4_hybrid",
        weights=FusionWeights(),
        use_exact=True,
        use_structured=True,
        use_sparse=True,
        use_dense=True,
    ),
)


class AblationRunner:
    """Evaluate A-D retrieval baselines and the complete Phase 4 hybrid."""

    def __init__(
        self,
        profiles: list[PlayerSeasonProfile],
        dense_retriever: DensePlayerRetriever,
        *,
        candidate_pool_size: int = 40,
        evaluator: RetrievalEvaluator | None = None,
    ) -> None:
        self.profiles = profiles
        self.query_analyzer = RuleBasedQueryAnalyzer(profiles)
        self.exact = ExactPlayerRetriever(profiles)
        self.structured = StructuredFeaturePlayerRetriever(profiles)
        self.sparse = BM25PlayerRetriever(profiles)
        self.dense = dense_retriever
        self.candidate_pool_size = candidate_pool_size
        self.evaluator = evaluator or RetrievalEvaluator()

    def run(
        self,
        dataset: GoldenDataset,
        variants: tuple[AblationVariant, ...] = DEFAULT_ABLATIONS,
    ) -> AblationReport:
        reports = []
        for variant in variants:
            retrievers = tuple(
                retriever
                for enabled, retriever in (
                    (variant.use_exact, self.exact),
                    (variant.use_structured, self.structured),
                    (variant.use_sparse, self.sparse),
                    (variant.use_dense, self.dense),
                )
                if enabled
            )
            pipeline = HybridRetrievalPipeline(
                self.query_analyzer,
                retrievers,
                WeightedRetrievalFusion(variant.weights),
                candidate_pool_size=self.candidate_pool_size,
            )
            reports.append(
                self.evaluator.evaluate(
                    pipeline,
                    dataset,
                    variant_name=variant.name,
                )
            )
        return AblationReport(
            dataset_version=dataset.schema_version,
            reports=reports,
        )
