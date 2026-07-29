"""Strict contracts for versioned relevance judgments and evaluation reports."""

from pydantic import Field, model_validator

from scoutrag.domain.base import ScoutRAGModel


class GoldenJudgment(ScoutRAGModel):
    """Graded relevance judgment for one player and query."""

    player_id: str = Field(min_length=1)
    relevance: int = Field(ge=1, le=3)
    rationale: str = Field(min_length=1)


class GoldenQuery(ScoutRAGModel):
    """One human-inspectable query with explicit expected players."""

    query_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    language: str = Field(pattern=r"^(de|en)$")
    category: str = Field(min_length=1)
    judgments: list[GoldenJudgment] = Field(min_length=1)
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_player_judgments(self) -> "GoldenQuery":
        player_ids = [judgment.player_id for judgment in self.judgments]
        if len(player_ids) != len(set(player_ids)):
            raise ValueError("a golden query cannot judge one player more than once")
        return self


class GoldenDataset(ScoutRAGModel):
    """Versioned golden set tied to one explicit source partition."""

    schema_version: str = Field(min_length=1)
    name: str = Field(min_length=1)
    competition_id: int = Field(gt=0)
    season_id: int = Field(gt=0)
    source_reference: str = Field(min_length=1)
    labeling_method: str = Field(min_length=1)
    limitations: list[str] = Field(default_factory=list)
    queries: list[GoldenQuery] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_query_ids(self) -> "GoldenDataset":
        query_ids = [query.query_id for query in self.queries]
        if len(query_ids) != len(set(query_ids)):
            raise ValueError("golden query IDs must be unique")
        return self


class KMetrics(ScoutRAGModel):
    """Metrics calculated at one cutoff."""

    precision: float = Field(ge=0, le=1)
    recall: float = Field(ge=0, le=1)
    ndcg: float = Field(ge=0, le=1)
    hit_rate: float = Field(ge=0, le=1)


class RetrievalMetrics(ScoutRAGModel):
    """Candidate and final-ranking metrics for one query or macro average."""

    candidate_recall: float = Field(ge=0, le=1)
    mean_reciprocal_rank: float = Field(ge=0, le=1)
    at_k: dict[int, KMetrics]


class QueryEvaluation(ScoutRAGModel):
    """Auditable per-query results before macro aggregation."""

    query_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    relevant_player_ids: list[str] = Field(default_factory=list)
    broad_candidate_ids: list[str] = Field(default_factory=list)
    ranked_player_ids: list[str] = Field(default_factory=list)
    reranking_ms: float = Field(default=0, ge=0)
    metrics: RetrievalMetrics


class EvaluationReport(ScoutRAGModel):
    """One retrieval configuration evaluated against a golden dataset."""

    variant_name: str = Field(min_length=1)
    dataset_version: str = Field(min_length=1)
    query_count: int = Field(ge=1)
    k_values: list[int] = Field(min_length=1)
    aggregate: RetrievalMetrics
    queries: list[QueryEvaluation] = Field(min_length=1)


class AblationReport(ScoutRAGModel):
    """Comparable reports for multiple retrieval configurations."""

    dataset_version: str = Field(min_length=1)
    reports: list[EvaluationReport] = Field(min_length=1)


class LatencyStats(ScoutRAGModel):
    """Summary of warm reranking latency across golden queries."""

    mean_ms: float = Field(ge=0)
    p50_ms: float = Field(ge=0)
    p95_ms: float = Field(ge=0)
    minimum_ms: float = Field(ge=0)
    maximum_ms: float = Field(ge=0)


class RerankingDelta(ScoutRAGModel):
    """Signed changes from fused order to cross-encoder order."""

    mean_reciprocal_rank: float
    ndcg_at_k: dict[int, float]
    hit_rate_at_k: dict[int, float]


class RerankingComparisonReport(ScoutRAGModel):
    """Before/after evaluation over the exact same broad candidate pools."""

    dataset_version: str = Field(min_length=1)
    model_name: str = Field(min_length=1)
    backend: str = Field(min_length=1)
    baseline: EvaluationReport
    reranked: EvaluationReport
    delta: RerankingDelta
    latency: LatencyStats
