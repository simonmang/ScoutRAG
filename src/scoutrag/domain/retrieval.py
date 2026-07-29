"""Candidate and trace models shared by retrieval and reranking."""

from typing import Any

from pydantic import Field, field_validator, model_validator

from scoutrag.domain.base import ScoutRAGModel
from scoutrag.domain.player import PlayerSeasonProfile
from scoutrag.domain.query import QueryProfile


class CandidateRetrievalTrace(ScoutRAGModel):
    """Per-candidate provenance across independent retrieval strategies."""

    player_id: str = Field(min_length=1)
    retrieved_by: list[str] = Field(min_length=1)
    dense_score: float | None = None
    sparse_score: float | None = None
    structured_score: float | None = None
    exact_score: float | None = None
    fused_score: float


class PlayerCandidate(ScoutRAGModel):
    """A season-specific player candidate before precise reranking."""

    profile: PlayerSeasonProfile
    retrieval_trace: CandidateRetrievalTrace

    @model_validator(mode="after")
    def ensure_trace_matches_profile(self) -> "PlayerCandidate":
        if self.profile.player_id != self.retrieval_trace.player_id:
            raise ValueError("candidate profile and retrieval trace player_id must match")
        return self


class RankedPlayerCandidate(PlayerCandidate):
    """A candidate after reranking; scores are relevance signals, not confidence."""

    rank: int = Field(ge=1)
    reranker_score: float | None = None
    ranking_reasons: list[str] = Field(default_factory=list)


class RetrievalTrace(ScoutRAGModel):
    """Audit and performance trace for one complete retrieval request."""

    query_id: str = Field(min_length=1)
    query_intent: str = Field(min_length=1)
    strategies_used: list[str] = Field(default_factory=list)
    candidates_per_strategy: dict[str, int] = Field(default_factory=dict)
    candidates_before_reranking: int = Field(ge=0)
    candidates_after_reranking: int = Field(ge=0)
    filters_applied: dict[str, Any] = Field(default_factory=dict)
    stage_timings_ms: dict[str, float] = Field(default_factory=dict)

    @field_validator("candidates_per_strategy")
    @classmethod
    def validate_candidate_counts(cls, values: dict[str, int]) -> dict[str, int]:
        if any(value < 0 for value in values.values()):
            raise ValueError("candidate counts cannot be negative")
        return values

    @field_validator("stage_timings_ms")
    @classmethod
    def validate_timings(cls, values: dict[str, float]) -> dict[str, float]:
        if any(value < 0 for value in values.values()):
            raise ValueError("stage timings cannot be negative")
        return values


class HybridRetrievalResult(ScoutRAGModel):
    """LLM-free Phase 4 result before evidence governance is implemented."""

    query_profile: QueryProfile
    broad_candidates: list[PlayerCandidate] = Field(default_factory=list)
    candidates: list[RankedPlayerCandidate] = Field(default_factory=list)
    retrieval_trace: RetrievalTrace

    @model_validator(mode="after")
    def validate_candidate_ranks(self) -> "HybridRetrievalResult":
        ranks = [candidate.rank for candidate in self.candidates]
        if ranks != list(range(1, len(ranks) + 1)):
            raise ValueError("candidates must have contiguous ranks starting at one")
        return self
