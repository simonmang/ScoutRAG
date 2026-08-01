"""Governed retrieval, compact search, and safe answer endpoints."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status

from scoutrag.api.dependencies import get_governed_pipeline, get_player_history_store
from scoutrag.api.schemas import (
    AnswerRequest,
    CompactCandidate,
    CompactSearchResponse,
    RetrievalRequest,
)
from scoutrag.config import Settings
from scoutrag.data.history import PlayerHistoryStore
from scoutrag.domain.evidence import (
    GeneratedAnswer,
    RecommendationEvidencePack,
)
from scoutrag.domain.player import PlayerTemporalContext
from scoutrag.governance.pipeline import GovernedRetrievalPipeline
from scoutrag.ports.answering import AnswerGenerator

router = APIRouter(tags=["scouting"])
PipelineDependency = Annotated[
    GovernedRetrievalPipeline,
    Depends(get_governed_pipeline),
]
HistoryDependency = Annotated[PlayerHistoryStore, Depends(get_player_history_store)]


@router.post("/retrieve", response_model=RecommendationEvidencePack)
def retrieve(
    payload: RetrievalRequest,
    pipeline: PipelineDependency,
    request: Request,
) -> RecommendationEvidencePack:
    """Return the complete LLM-independent Evidence Pack."""
    _validate_result_count(payload, request.app.state.settings)
    return pipeline.search(payload.query, result_count=payload.result_count)


@router.post("/search", response_model=CompactSearchResponse)
def search(
    payload: RetrievalRequest,
    pipeline: PipelineDependency,
    request: Request,
) -> CompactSearchResponse:
    """Return a compact facade derived from the same governed pipeline."""
    pack = retrieve(payload, pipeline, request)
    return CompactSearchResponse(
        query_id=pack.retrieval_trace.query_id,
        query=pack.query_profile.original_query,
        verdict=pack.governance.verdict,
        evidence_quality_score=pack.governance.evidence_quality_score,
        candidates=[
            CompactCandidate(
                player_id=candidate.profile.player_id,
                player_name=candidate.profile.player_name,
                team_name=candidate.profile.team_name,
                competition_name=candidate.profile.competition_name,
                season_name=candidate.profile.season_name,
                position_group=candidate.profile.position_group,
                minutes_played=candidate.profile.minutes_played,
                data_quality=candidate.profile.data_quality,
                rank=candidate.rank,
                relevance_score=(
                    candidate.reranker_score
                    if candidate.reranker_score is not None
                    else candidate.retrieval_trace.fused_score
                ),
                retrieved_by=candidate.retrieval_trace.retrieved_by,
            )
            for candidate in pack.candidates
        ],
        warnings=pack.governance.warnings,
        missing_evidence=pack.governance.missing_evidence,
        total_ms=pack.runtime_metrics.total_ms,
    )


@router.post("/answer", response_model=GeneratedAnswer)
def answer(payload: AnswerRequest, request: Request) -> GeneratedAnswer:
    """Render only the supplied validated Evidence Pack; no retrieval is hidden here."""
    generator: AnswerGenerator = request.app.state.answer_generator
    return generator.generate(payload.evidence_pack)


@router.get("/players/{player_id}/history", response_model=PlayerTemporalContext)
def player_history(
    player_id: str,
    history_store: HistoryDependency,
    match_limit: int = 10,
) -> PlayerTemporalContext:
    """Return separate season, club, form, trend, and recent-match evidence."""

    if not 0 <= match_limit <= 50:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="match_limit must be between zero and 50",
        )
    context = history_store.for_player(player_id, match_limit=match_limit)
    if context.identity is None and not context.season_profiles:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No player history found for player_id={player_id}",
        )
    return context


def _validate_result_count(payload: RetrievalRequest, settings: Settings) -> None:
    if payload.result_count is not None and payload.result_count > settings.max_result_count:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"result_count cannot exceed {settings.max_result_count}",
        )
