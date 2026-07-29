"""Transport-specific response models."""

from typing import Literal

from pydantic import BaseModel, Field

from scoutrag.domain.evidence import (
    EvidenceVerdict,
    RecommendationEvidencePack,
)


class HealthResponse(BaseModel):
    """Stable health response for probes and smoke tests."""

    status: Literal["ok"] = "ok"
    service: str
    version: str
    environment: str


class RetrievalRequest(BaseModel):
    """Validated natural-language retrieval request."""

    query: str = Field(min_length=2, max_length=500)
    result_count: int | None = Field(default=None, ge=1, le=100)


class AnswerRequest(BaseModel):
    """Answer generation receives a completed, validated Evidence Pack."""

    evidence_pack: RecommendationEvidencePack


class CompactCandidate(BaseModel):
    """Dashboard-friendly projection of one ranked candidate."""

    player_id: str
    player_name: str
    team_name: str
    competition_name: str
    season_name: str
    position_group: str
    minutes_played: float
    data_quality: float
    rank: int
    relevance_score: float
    retrieved_by: list[str]


class CompactSearchResponse(BaseModel):
    """Compact facade derived from the same governed retrieval result."""

    query_id: str
    query: str
    verdict: EvidenceVerdict
    evidence_quality_score: float
    candidates: list[CompactCandidate]
    warnings: list[str]
    missing_evidence: list[str]
    total_ms: float
