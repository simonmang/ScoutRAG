"""Ports that explicitly separate broad recall from result fusion."""

from collections.abc import Mapping, Sequence
from typing import Protocol

from scoutrag.domain.query import QueryProfile
from scoutrag.domain.retrieval import PlayerCandidate


class PlayerRetriever(Protocol):
    """Retrieve a broad candidate list with one independent strategy."""

    @property
    def strategy_name(self) -> str:
        """Stable name stored in the retrieval trace."""
        ...

    def retrieve(
        self,
        query_profile: QueryProfile,
        *,
        limit: int,
    ) -> list[PlayerCandidate]:
        """Return season-specific candidates, prioritizing recall."""
        ...


class CandidateRetriever(PlayerRetriever, Protocol):
    """Semantic alias used by the pipeline for broad-recall retrievers."""


class RetrievalFusion(Protocol):
    """Normalize and fuse scores without claiming evidence quality."""

    def fuse(
        self,
        query_profile: QueryProfile,
        candidates_by_strategy: Mapping[str, Sequence[PlayerCandidate]],
        *,
        limit: int,
    ) -> list[PlayerCandidate]:
        """Return a deduplicated, fused candidate pool."""
        ...
