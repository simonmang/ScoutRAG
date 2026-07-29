"""Governance outcomes and the primary ScoutRAG result contract."""

from enum import StrEnum

from pydantic import Field, field_validator, model_validator

from scoutrag.domain.base import ScoutRAGModel
from scoutrag.domain.player import PlayerMetricEvidence
from scoutrag.domain.query import QueryProfile
from scoutrag.domain.retrieval import RankedPlayerCandidate, RetrievalTrace


class EvidenceVerdict(StrEnum):
    """Rule-based verdict about whether recommendations are supportable."""

    SUFFICIENT = "sufficient"
    LIMITED = "limited"
    INSUFFICIENT = "insufficient"
    CONFLICTING = "conflicting"
    OUT_OF_SCOPE = "out_of_scope"


class GenerationMode(StrEnum):
    """How the final natural-language projection was produced."""

    TEMPLATE = "template"
    GROUNDED_MODEL = "grounded_model"
    SAFE_FALLBACK = "safe_fallback"


class GroundingReport(ScoutRAGModel):
    """Validation result for generated claims; the score is not a probability."""

    validation_passed: bool = True
    grounding_score: float = Field(default=1, ge=0, le=1)
    claim_count: int = Field(default=0, ge=0)
    supported_claim_count: int = Field(default=0, ge=0)
    cited_fact_ids: list[str] = Field(default_factory=list)
    violations: list[str] = Field(default_factory=list)
    generator: str = Field(default="template", min_length=1)
    fallback_used: bool = False

    @model_validator(mode="after")
    def validate_supported_claim_count(self) -> "GroundingReport":
        if self.supported_claim_count > self.claim_count:
            raise ValueError("supported_claim_count cannot exceed claim_count")
        if self.validation_passed and self.violations:
            raise ValueError("a passed grounding report cannot contain violations")
        return self


class RecommendationGovernance(ScoutRAGModel):
    """Transparent assessment of evidence quality, never a probability."""

    verdict: EvidenceVerdict
    evidence_quality_score: float = Field(ge=0, le=1)
    reasons: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    factors: dict[str, float] = Field(default_factory=dict)

    @field_validator("factors")
    @classmethod
    def validate_factors(cls, values: dict[str, float]) -> dict[str, float]:
        invalid = {name: value for name, value in values.items() if not 0 <= value <= 1}
        if invalid:
            raise ValueError(f"governance factors must be between 0 and 1: {invalid}")
        return values

    @model_validator(mode="after")
    def require_explanation_for_non_sufficient_verdict(
        self,
    ) -> "RecommendationGovernance":
        if self.verdict is not EvidenceVerdict.SUFFICIENT and not (
            self.reasons or self.missing_evidence or self.warnings
        ):
            raise ValueError("non-sufficient governance verdicts require an explanation")
        return self


class RuntimeMetrics(ScoutRAGModel):
    """Latency data for observing stages without mixing it into relevance."""

    total_ms: float = Field(ge=0)
    query_analysis_ms: float = Field(default=0, ge=0)
    candidate_retrieval_ms: float = Field(default=0, ge=0)
    fusion_ms: float = Field(default=0, ge=0)
    reranking_ms: float = Field(default=0, ge=0)
    governance_ms: float = Field(default=0, ge=0)
    evidence_assembly_ms: float = Field(default=0, ge=0)


class RecommendationEvidencePack(ScoutRAGModel):
    """LLM-independent, auditable output of the retrieval pipeline."""

    query_profile: QueryProfile
    governance: RecommendationGovernance
    candidates: list[RankedPlayerCandidate] = Field(default_factory=list)
    retrieval_trace: RetrievalTrace
    metric_evidence: dict[str, list[PlayerMetricEvidence]] = Field(default_factory=dict)
    limitations: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    runtime_metrics: RuntimeMetrics

    @model_validator(mode="after")
    def validate_ranked_candidates(self) -> "RecommendationEvidencePack":
        ranks = [candidate.rank for candidate in self.candidates]
        if len(ranks) != len(set(ranks)):
            raise ValueError("candidate ranks must be unique")
        if ranks != list(range(1, len(ranks) + 1)):
            raise ValueError("candidates must be sorted with contiguous ranks starting at one")
        candidate_ids = {candidate.profile.player_id for candidate in self.candidates}
        unknown_evidence_ids = set(self.metric_evidence) - candidate_ids
        if unknown_evidence_ids:
            raise ValueError(
                "metric evidence may only reference returned candidate keys: "
                f"{sorted(unknown_evidence_ids)}"
            )
        mismatched_evidence = {
            key: item.player_id
            for key, items in self.metric_evidence.items()
            for item in items
            if item.player_id != key
        }
        if mismatched_evidence:
            raise ValueError(
                f"metric evidence player_id must match its dictionary key: {mismatched_evidence}"
            )
        return self


class GeneratedAnswer(ScoutRAGModel):
    """Governed natural-language projection of an evidence pack."""

    query_id: str = Field(min_length=1)
    verdict: EvidenceVerdict
    text: str = Field(min_length=1)
    cited_player_ids: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    generation_mode: GenerationMode = GenerationMode.TEMPLATE
    grounding: GroundingReport = Field(default_factory=GroundingReport)
