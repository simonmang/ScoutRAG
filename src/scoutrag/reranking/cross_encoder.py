"""Cross-encoder reranking behind the model-independent PlayerReranker port."""

import importlib
from collections.abc import Sequence
from typing import Any, Literal, Protocol

from scoutrag.domain.query import QueryProfile
from scoutrag.domain.retrieval import PlayerCandidate, RankedPlayerCandidate
from scoutrag.retrieval.common import profile_search_text, query_search_text

DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


class PairScoringModel(Protocol):
    """Small injectable boundary for pairwise query/profile relevance models."""

    @property
    def model_name(self) -> str:
        """Stable model identifier for reports and audit logs."""
        ...

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        """Return exactly one uncalibrated relevance score per text pair."""
        ...


class SentenceTransformerCrossEncoderModel:
    """Lazy Sentence Transformers CrossEncoder adapter with optional ONNX backend."""

    def __init__(
        self,
        model_name: str = DEFAULT_CROSS_ENCODER_MODEL,
        *,
        batch_size: int = 16,
        local_files_only: bool = False,
        backend: Literal["torch", "onnx"] = "torch",
        onnx_file_name: str = "onnx/model.onnx",
        device: str | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self._model_name = model_name
        self.batch_size = batch_size
        self.local_files_only = local_files_only
        self.backend = backend
        self.onnx_file_name = onnx_file_name
        self.device = device
        self._model: Any | None = None

    @property
    def model_name(self) -> str:
        return self._model_name

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []
        scores = self._load().predict(
            list(pairs),
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        values = scores.tolist() if hasattr(scores, "tolist") else scores
        return [float(value) for value in values]

    def warmup(self) -> None:
        """Load the model and run one pair outside measured request latency."""
        self.score_pairs([("football player", "football player profile")])

    def _load(self) -> Any:
        if self._model is None:
            try:
                module = importlib.import_module("sentence_transformers")
            except ModuleNotFoundError as error:
                raise RuntimeError(
                    "Cross-encoder reranking requires the optional 'retrieval' dependencies: "
                    'pip install -e ".[retrieval]"'
                ) from error
            try:
                model_kwargs = (
                    {"file_name": self.onnx_file_name} if self.backend == "onnx" else None
                )
                self._model = module.CrossEncoder(
                    self.model_name,
                    local_files_only=self.local_files_only,
                    backend=self.backend,
                    device=self.device,
                    model_kwargs=model_kwargs,
                )
            except ImportError as error:
                if self.backend == "onnx":
                    raise RuntimeError(
                        "ONNX reranking requires the optional 'onnx' dependencies: "
                        'pip install -e ".[onnx]"'
                    ) from error
                raise
        return self._model


class CrossEncoderPlayerReranker:
    """Jointly score each query/profile pair and return a stable relevance order."""

    def __init__(self, scoring_model: PairScoringModel) -> None:
        self.scoring_model = scoring_model

    def rerank(
        self,
        query_profile: QueryProfile,
        candidates: list[PlayerCandidate],
    ) -> list[RankedPlayerCandidate]:
        if not candidates:
            return []
        query_text = query_search_text(query_profile)
        pairs = [(query_text, profile_search_text(candidate.profile)) for candidate in candidates]
        scores = self.scoring_model.score_pairs(pairs)
        if len(scores) != len(candidates):
            raise ValueError("cross-encoder backend must return one score per candidate")

        scored = list(zip(candidates, scores, strict=True))
        scored.sort(
            key=lambda item: (
                -item[1],
                -item[0].retrieval_trace.fused_score,
                item[0].profile.player_name,
                item[0].profile.season_name,
            )
        )
        return [
            RankedPlayerCandidate(
                profile=candidate.profile,
                retrieval_trace=candidate.retrieval_trace,
                rank=rank,
                reranker_score=round(score, 6),
                ranking_reasons=[
                    (
                        f"Pairwise relevance from {self.scoring_model.model_name}; "
                        "the raw score is not a calibrated probability."
                    )
                ],
            )
            for rank, (candidate, score) in enumerate(scored, start=1)
        ]
