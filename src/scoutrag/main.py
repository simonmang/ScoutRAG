"""ScoutRAG FastAPI application factory."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from scoutrag import __version__
from scoutrag.api.routes.health import router as health_router
from scoutrag.config import Settings, get_settings
from scoutrag.logging import configure_logging


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create an isolated app instance suitable for tests and deployment."""
    resolved_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        configure_logging(resolved_settings.log_level)
        yield

    application = FastAPI(
        title=resolved_settings.app_name,
        version=__version__,
        description="Evidence-based multi-stage retrieval for football scouting.",
        lifespan=lifespan,
    )
    application.state.settings = resolved_settings
    application.include_router(health_router)
    return application


app = create_app()
