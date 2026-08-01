"""ScoutRAG FastAPI application factory."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from scoutrag import __version__
from scoutrag.answering.generator import GroundedAnswerGenerator
from scoutrag.answering.openai_backend import OpenAIResponsesBackend
from scoutrag.answering.templates import TemplateAnswerGenerator
from scoutrag.api.dependencies import (
    GovernedPipelineProvider,
    default_pipeline_loader,
)
from scoutrag.api.routes.dashboard import DASHBOARD_ROOT
from scoutrag.api.routes.dashboard import router as dashboard_router
from scoutrag.api.routes.health import router as health_router
from scoutrag.api.routes.retrieval import router as retrieval_router
from scoutrag.config import Settings, get_settings
from scoutrag.data.history import PlayerHistoryStore
from scoutrag.governance.pipeline import GovernedRetrievalPipeline
from scoutrag.logging import configure_logging
from scoutrag.ports.answering import AnswerGenerator


def create_app(
    settings: Settings | None = None,
    *,
    pipeline: GovernedRetrievalPipeline | None = None,
    answer_generator: AnswerGenerator | None = None,
    history_store: PlayerHistoryStore | None = None,
) -> FastAPI:
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
    resolved_history_store = history_store
    if resolved_history_store is None and resolved_settings.history_path.exists():
        resolved_history_store = PlayerHistoryStore(resolved_settings.history_path)
    application.state.settings = resolved_settings
    application.state.history_store = resolved_history_store
    application.state.pipeline_provider = GovernedPipelineProvider(
        default_pipeline_loader(resolved_settings, resolved_history_store),
        pipeline=pipeline,
    )
    application.state.answer_generator = answer_generator or _build_answer_generator(
        resolved_settings
    )
    application.include_router(health_router)
    application.include_router(retrieval_router, prefix=resolved_settings.api_prefix)
    application.include_router(dashboard_router)
    application.mount(
        "/assets",
        StaticFiles(directory=Path(DASHBOARD_ROOT)),
        name="dashboard-assets",
    )
    return application


def _build_answer_generator(settings: Settings) -> AnswerGenerator:
    if settings.answer_mode == "openai":
        return GroundedAnswerGenerator(
            OpenAIResponsesBackend(
                model=settings.openai_model,
                max_output_tokens=settings.openai_max_output_tokens,
            )
        )
    return TemplateAnswerGenerator()


app = create_app()
