"""Health endpoint; Phase 1 intentionally exposes no retrieval endpoint."""

from fastapi import APIRouter, Request

from scoutrag import __version__
from scoutrag.api.schemas import HealthResponse
from scoutrag.config import Settings

router = APIRouter(tags=["operations"])


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    """Report application liveness and basic build metadata."""
    settings: Settings = request.app.state.settings
    return HealthResponse(
        service=settings.app_name,
        version=__version__,
        environment=settings.environment,
    )
