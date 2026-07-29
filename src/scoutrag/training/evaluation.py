"""Compare pretrained and football-fine-tuned bi-encoders."""

from collections import defaultdict

from scoutrag.domain.player import PlayerSeasonProfile
from scoutrag.evaluation.models import GoldenDataset
from scoutrag.evaluation.runner import RetrievalEvaluator
from scoutrag.retrieval.dense import (
    DensePlayerRetriever,
    TextEmbeddingModel,
    cosine_similarity,
)
from scoutrag.retrieval.fusion import FusionWeights, WeightedRetrievalFusion
from scoutrag.retrieval.pipeline import HybridRetrievalPipeline
from scoutrag.retrieval.query_analysis import RuleBasedQueryAnalyzer
from scoutrag.training.models import (
    BiEncoderComparisonReport,
    BiEncoderMetricDelta,
    BiEncoderModelEvaluation,
    MinedTrainingDataset,
    PairwiseRetrievalMetrics,
)


class BiEncoderEvaluator:
    """Measure golden retrieval and held-out hard-negative ranking."""

    def __init__(
        self,
        profiles: list[PlayerSeasonProfile],
        golden_dataset: GoldenDataset,
        mined_dataset: MinedTrainingDataset,
        *,
        candidate_pool_size: int = 40,
    ) -> None:
        self.profiles = profiles
        self.golden_dataset = golden_dataset
        self.mined_dataset = mined_dataset
        self.candidate_pool_size = candidate_pool_size

    def evaluate(
        self,
        embedding_model: TextEmbeddingModel,
        *,
        variant_name: str,
    ) -> BiEncoderModelEvaluation:
        dense = DensePlayerRetriever(self.profiles, embedding_model)
        pipeline = HybridRetrievalPipeline(
            RuleBasedQueryAnalyzer(self.profiles),
            (dense,),
            WeightedRetrievalFusion(FusionWeights(dense=1, sparse=0, structured=0, exact=0)),
            candidate_pool_size=self.candidate_pool_size,
        )
        golden = RetrievalEvaluator().evaluate(
            pipeline,
            self.golden_dataset,
            variant_name=variant_name,
        )
        return BiEncoderModelEvaluation(
            model_name=embedding_model.model_name,
            golden_retrieval=golden,
            pairwise=self.evaluate_pairwise(embedding_model),
        )

    def compare(
        self,
        baseline: TextEmbeddingModel,
        fine_tuned: TextEmbeddingModel,
    ) -> BiEncoderComparisonReport:
        baseline_report = self.evaluate(baseline, variant_name="pretrained_bi_encoder")
        fine_tuned_report = self.evaluate(fine_tuned, variant_name="football_fine_tuned_bi_encoder")
        baseline_metrics = baseline_report.golden_retrieval.aggregate
        fine_tuned_metrics = fine_tuned_report.golden_retrieval.aggregate
        return BiEncoderComparisonReport(
            dataset_version=self.mined_dataset.source_dataset_version,
            baseline=baseline_report,
            fine_tuned=fine_tuned_report,
            delta=BiEncoderMetricDelta(
                candidate_recall=_rounded(
                    fine_tuned_metrics.candidate_recall - baseline_metrics.candidate_recall
                ),
                mean_reciprocal_rank=_rounded(
                    fine_tuned_metrics.mean_reciprocal_rank - baseline_metrics.mean_reciprocal_rank
                ),
                ndcg_at_5=_rounded(fine_tuned_metrics.at_k[5].ndcg - baseline_metrics.at_k[5].ndcg),
                hard_negative_accuracy=_rounded(
                    fine_tuned_report.pairwise.hard_negative_accuracy
                    - baseline_report.pairwise.hard_negative_accuracy
                ),
                bilingual_pair_stability=_rounded(
                    fine_tuned_report.pairwise.bilingual_pair_stability
                    - baseline_report.pairwise.bilingual_pair_stability
                ),
            ),
        )

    def evaluate_pairwise(self, model: TextEmbeddingModel) -> PairwiseRetrievalMetrics:
        """Evaluate validation triplets without applying hard query filters."""
        examples = [
            example for example in self.mined_dataset.examples if example.split == "validation"
        ]
        queries = model.encode_queries([example.query_text for example in examples])
        documents = model.encode_documents(
            [
                text
                for example in examples
                for text in (
                    example.positive_text,
                    example.hard_negative_text,
                    example.easy_negative_text,
                )
            ]
        )
        if len(queries) != len(examples) or len(documents) != len(examples) * 3:
            raise ValueError("embedding backend returned an invalid pairwise matrix")

        hard_wins = 0
        easy_wins = 0
        reciprocal_ranks: list[float] = []
        margins: list[float] = []
        wins_by_language: dict[str, list[bool]] = defaultdict(list)
        wins_by_concept: dict[str, list[bool]] = defaultdict(list)
        for index, (example, query_vector) in enumerate(zip(examples, queries, strict=True)):
            offset = index * 3
            scores = [
                cosine_similarity(query_vector, document)
                for document in documents[offset : offset + 3]
            ]
            positive, hard, easy = scores
            hard_win = positive > hard
            easy_win = positive > easy
            hard_wins += hard_win
            easy_wins += easy_win
            rank = 1 + sum(score > positive for score in (hard, easy))
            reciprocal_ranks.append(1 / rank)
            margins.append(positive - hard)
            wins_by_language[example.language].append(hard_win)
            wins_by_concept[example.concept_id].append(hard_win)

        bilingual = [wins for wins in wins_by_concept.values() if len(wins) >= 2]
        return PairwiseRetrievalMetrics(
            example_count=len(examples),
            hard_negative_accuracy=_rounded(hard_wins / len(examples)),
            easy_negative_accuracy=_rounded(easy_wins / len(examples)),
            mean_reciprocal_rank=_rounded(sum(reciprocal_ranks) / len(examples)),
            mean_positive_hard_margin=_rounded(sum(margins) / len(examples)),
            language_accuracy={
                language: _rounded(sum(wins) / len(wins))
                for language, wins in sorted(wins_by_language.items())
            },
            bilingual_pair_stability=_rounded(
                sum(all(wins) for wins in bilingual) / len(bilingual) if bilingual else 0
            ),
        )


def _rounded(value: float) -> float:
    return round(value, 6)
