"""Lazy, process-wide application services for FastAPI dependencies."""

from collections.abc import Callable
from threading import Lock

from fastapi import HTTPException, Request, status

from scoutrag.config import Settings
from scoutrag.governance.evidence import load_metric_evidence
from scoutrag.governance.factory import build_governed_pipeline
from scoutrag.governance.pipeline import GovernedRetrievalPipeline
from scoutrag.retrieval.common import load_profiles
from scoutrag.retrieval.dense import DensePlayerRetriever, SentenceTransformerEmbeddingModel


class PipelineUnavailableError(RuntimeError):
    """Raised when local data or model artifacts cannot serve retrieval."""


class GovernedPipelineProvider:
    """Load one governed pipeline lazily and reuse it across requests."""

    def __init__(
        self,
        loader: Callable[[], GovernedRetrievalPipeline],
        *,
        pipeline: GovernedRetrievalPipeline | None = None,
    ) -> None:
        self._loader = loader
        self._pipeline = pipeline
        self._lock = Lock()

    def get(self) -> GovernedRetrievalPipeline:
        if self._pipeline is None:
            with self._lock:
                if self._pipeline is None:
                    try:
                        self._pipeline = self._loader()
                    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
                        raise PipelineUnavailableError(str(error)) from error
        return self._pipeline


def default_pipeline_loader(settings: Settings) -> Callable[[], GovernedRetrievalPipeline]:
    """Create a deferred loader from environment-backed artifact paths."""

    def load() -> GovernedRetrievalPipeline:
        missing = [
            path
            for path in (settings.profiles_path, settings.metric_evidence_path)
            if not path.exists()
        ]
        if missing:
            names = ", ".join(str(path) for path in missing)
            raise FileNotFoundError(
                f"ScoutRAG data artifacts are missing: {names}. Run 'scoutrag-data build' first."
            )
        profiles = load_profiles(settings.profiles_path)
        evidence = load_metric_evidence(settings.metric_evidence_path)
        dense = None
        if settings.enable_dense_retrieval:
            dense = DensePlayerRetriever(
                profiles,
                SentenceTransformerEmbeddingModel(
                    settings.dense_model_name,
                    local_files_only=settings.local_files_only,
                ),
                index_path=settings.dense_index_path,
            )
        return build_governed_pipeline(
            profiles,
            evidence,
            dense_retriever=dense,
            candidate_pool_size=settings.candidate_pool_size,
        )

    return load


def get_governed_pipeline(request: Request) -> GovernedRetrievalPipeline:
    """Resolve the shared pipeline or return an actionable 503 response."""
    provider: GovernedPipelineProvider = request.app.state.pipeline_provider
    try:
        return provider.get()
    except PipelineUnavailableError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(error),
        ) from error
