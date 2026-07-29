"""Bi-encoder retrieval with an injectable embedding backend."""

import importlib
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

from scoutrag.domain.player import PlayerSeasonProfile
from scoutrag.domain.query import QueryIntent, QueryProfile
from scoutrag.domain.retrieval import CandidateRetrievalTrace, PlayerCandidate
from scoutrag.retrieval.common import (
    matches_hard_filters,
    profile_search_text,
    query_search_text,
)

DEFAULT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class TextEmbeddingModel(Protocol):
    """Minimal model boundary used for real and deterministic test embeddings."""

    @property
    def model_name(self) -> str:
        """Stable model identifier used in logs and index metadata."""
        ...

    def encode_queries(self, texts: Sequence[str]) -> list[list[float]]:
        """Encode query text."""
        ...

    def encode_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Encode player-profile text."""
        ...


class SentenceTransformerEmbeddingModel:
    """Lazy adapter around a multilingual pretrained Sentence Transformer."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        *,
        local_files_only: bool = False,
    ) -> None:
        self._model_name = model_name
        self.local_files_only = local_files_only
        self._model: Any | None = None

    @property
    def model_name(self) -> str:
        return self._model_name

    def encode_queries(self, texts: Sequence[str]) -> list[list[float]]:
        model = self._load()
        embeddings = model.encode_query(
            list(texts),
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return _to_lists(embeddings)

    def encode_documents(self, texts: Sequence[str]) -> list[list[float]]:
        model = self._load()
        embeddings = model.encode_document(
            list(texts),
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return _to_lists(embeddings)

    def _load(self) -> Any:
        if self._model is None:
            try:
                module = importlib.import_module("sentence_transformers")
            except ModuleNotFoundError as error:
                raise RuntimeError(
                    "Dense retrieval requires the optional 'retrieval' dependencies: "
                    'pip install -e ".[retrieval]"'
                ) from error
            model_path = self.model_name
            if self.local_files_only and not Path(model_path).exists():
                hub = importlib.import_module("huggingface_hub")
                model_path = hub.snapshot_download(
                    repo_id=self.model_name,
                    local_files_only=True,
                )
            self._model = module.SentenceTransformer(
                model_path,
                local_files_only=self.local_files_only,
            )
        return self._model


class DensePlayerRetriever:
    """Encode query and profiles separately, then retrieve by cosine similarity."""

    strategy_name = "dense"

    def __init__(
        self,
        profiles: list[PlayerSeasonProfile],
        embedding_model: TextEmbeddingModel,
        *,
        index_path: Path | None = None,
        rebuild_index: bool = False,
    ) -> None:
        self.profiles = tuple(profiles)
        self.embedding_model = embedding_model
        self.index_path = index_path
        self.document_embeddings = self._load_or_build_index(rebuild_index=rebuild_index)
        if len(self.document_embeddings) != len(self.profiles):
            raise ValueError("embedding backend returned the wrong number of document vectors")

    def retrieve(self, query_profile: QueryProfile, *, limit: int) -> list[PlayerCandidate]:
        if query_profile.intent is QueryIntent.OUT_OF_SCOPE or not self.profiles:
            return []
        encoded = self.embedding_model.encode_queries([query_search_text(query_profile)])
        if len(encoded) != 1:
            raise ValueError("embedding backend must return exactly one query vector")
        query_embedding = encoded[0]
        scored: list[tuple[float, PlayerSeasonProfile]] = []
        for profile, document_embedding in zip(
            self.profiles,
            self.document_embeddings,
            strict=True,
        ):
            if not matches_hard_filters(profile, query_profile):
                continue
            scored.append((cosine_similarity(query_embedding, document_embedding), profile))
        scored.sort(key=lambda item: (-item[0], item[1].player_name, item[1].season_name))
        return [
            PlayerCandidate(
                profile=profile,
                retrieval_trace=CandidateRetrievalTrace(
                    player_id=profile.player_id,
                    retrieved_by=[self.strategy_name],
                    dense_score=round(score, 6),
                    fused_score=round(score, 6),
                ),
            )
            for score, profile in scored[:limit]
        ]

    def _load_or_build_index(self, *, rebuild_index: bool) -> list[list[float]]:
        expected_keys = [
            [profile.player_id, profile.competition_name, profile.season_name]
            for profile in self.profiles
        ]
        if self.index_path is not None and self.index_path.exists() and not rebuild_index:
            payload = json.loads(self.index_path.read_text(encoding="utf-8"))
            if payload.get("schema_version") != "dense-index-v1":
                raise ValueError("unsupported dense index schema")
            if payload.get("model_name") != self.embedding_model.model_name:
                raise ValueError("dense index model does not match the configured embedding model")
            if payload.get("profile_keys") != expected_keys:
                raise ValueError("dense index profiles do not match the loaded profile dataset")
            embeddings = payload.get("embeddings")
            if not isinstance(embeddings, list):
                raise ValueError("dense index contains no embedding matrix")
            return [[float(value) for value in vector] for vector in embeddings]

        documents = [profile_search_text(profile) for profile in self.profiles]
        embeddings = self.embedding_model.encode_documents(documents) if documents else []
        if self.index_path is not None:
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            self.index_path.write_text(
                json.dumps(
                    {
                        "schema_version": "dense-index-v1",
                        "model_name": self.embedding_model.model_name,
                        "profile_keys": expected_keys,
                        "embedding_dimension": len(embeddings[0]) if embeddings else 0,
                        "embeddings": embeddings,
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                encoding="utf-8",
            )
        return embeddings


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Calculate cosine similarity without assigning confidence semantics."""
    if len(left) != len(right):
        raise ValueError("query and document embedding dimensions differ")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=True)) / (left_norm * right_norm)


def _to_lists(embeddings: Any) -> list[list[float]]:
    values = embeddings.tolist() if hasattr(embeddings, "tolist") else embeddings
    return [[float(value) for value in vector] for vector in values]
