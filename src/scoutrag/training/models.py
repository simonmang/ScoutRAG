"""Strict contracts for football retrieval training and comparison reports."""

from typing import Literal

from pydantic import Field, model_validator

from scoutrag.domain.base import ScoutRAGModel
from scoutrag.evaluation.models import EvaluationReport

TrainingSplit = Literal["train", "validation"]
NegativeConstraint = Literal["lower_target_metric", "wrong_team", "wrong_player"]


class RetrievalTrainingQuery(ScoutRAGModel):
    """One labeled query whose negatives are resolved against typed profiles."""

    query_id: str = Field(min_length=1)
    concept_id: str = Field(min_length=1)
    split: TrainingSplit
    query: str = Field(min_length=2)
    language: Literal["de", "en"]
    positive_player_id: str = Field(min_length=1)
    negative_constraint: NegativeConstraint
    target_metric: str | None = None
    required_team: str | None = None
    minimum_percentile_gap: float = Field(default=25, ge=0, le=100)
    rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def constraint_has_required_fields(self) -> "RetrievalTrainingQuery":
        if self.negative_constraint == "lower_target_metric" and not self.target_metric:
            raise ValueError("lower_target_metric requires target_metric")
        if self.negative_constraint == "wrong_team" and not self.required_team:
            raise ValueError("wrong_team requires required_team")
        return self


class BiEncoderTrainingDataset(ScoutRAGModel):
    """Versioned, source-bound training and validation query specifications."""

    schema_version: str = Field(min_length=1)
    name: str = Field(min_length=1)
    competition_id: int = Field(gt=0)
    season_id: int = Field(gt=0)
    source_reference: str = Field(min_length=1)
    labeling_method: str = Field(min_length=1)
    limitations: list[str] = Field(default_factory=list)
    queries: list[RetrievalTrainingQuery] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_query_identity_and_splits(self) -> "BiEncoderTrainingDataset":
        query_ids = [query.query_id for query in self.queries]
        if len(query_ids) != len(set(query_ids)):
            raise ValueError("training query IDs must be unique")
        if {query.split for query in self.queries} != {"train", "validation"}:
            raise ValueError("dataset must contain train and validation examples")
        return self


class MinedTrainingExample(ScoutRAGModel):
    """Resolved positive, hard-negative, and easy-negative training tuple."""

    query_id: str
    concept_id: str
    split: TrainingSplit
    language: Literal["de", "en"]
    original_query: str
    query_text: str
    positive_player_id: str
    positive_player_name: str
    positive_text: str
    hard_negative_player_id: str
    hard_negative_player_name: str
    hard_negative_text: str
    easy_negative_player_id: str
    easy_negative_player_name: str
    easy_negative_text: str
    positive_score: float
    hard_negative_score: float
    easy_negative_score: float
    negative_constraint: NegativeConstraint
    rationale: str

    @model_validator(mode="after")
    def distinct_players(self) -> "MinedTrainingExample":
        player_ids = {
            self.positive_player_id,
            self.hard_negative_player_id,
            self.easy_negative_player_id,
        }
        if len(player_ids) != 3:
            raise ValueError("positive, hard negative, and easy negative must be distinct")
        return self


class MinedTrainingDataset(ScoutRAGModel):
    """Auditable output of deterministic constrained hard-negative mining."""

    schema_version: str = "phase9-mined-v1"
    source_dataset_version: str
    embedding_model: str
    examples: list[MinedTrainingExample] = Field(min_length=1)

    @model_validator(mode="after")
    def retain_both_splits(self) -> "MinedTrainingDataset":
        if {example.split for example in self.examples} != {"train", "validation"}:
            raise ValueError("mined dataset must retain train and validation splits")
        return self


class PairwiseRetrievalMetrics(ScoutRAGModel):
    """Positive-vs-negative behavior independent of the full retrieval pipeline."""

    example_count: int = Field(ge=1)
    hard_negative_accuracy: float = Field(ge=0, le=1)
    easy_negative_accuracy: float = Field(ge=0, le=1)
    mean_reciprocal_rank: float = Field(ge=0, le=1)
    mean_positive_hard_margin: float
    language_accuracy: dict[str, float]
    bilingual_pair_stability: float = Field(ge=0, le=1)


class BiEncoderModelEvaluation(ScoutRAGModel):
    """Golden retrieval and held-out pairwise results for one encoder."""

    model_name: str
    golden_retrieval: EvaluationReport
    pairwise: PairwiseRetrievalMetrics


class BiEncoderMetricDelta(ScoutRAGModel):
    """Signed fine-tuned-minus-baseline metric changes."""

    candidate_recall: float
    mean_reciprocal_rank: float
    ndcg_at_5: float
    hard_negative_accuracy: float
    bilingual_pair_stability: float


class BiEncoderComparisonReport(ScoutRAGModel):
    """Auditable before/after comparison for Phase 9."""

    dataset_version: str
    baseline: BiEncoderModelEvaluation
    fine_tuned: BiEncoderModelEvaluation
    delta: BiEncoderMetricDelta
