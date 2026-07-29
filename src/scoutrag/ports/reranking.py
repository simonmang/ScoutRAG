"""Port for ranking candidates after broad recall."""

from typing import Protocol

from scoutrag.domain.query import QueryProfile
from scoutrag.domain.retrieval import PlayerCandidate, RankedPlayerCandidate


class PlayerReranker(Protocol):
    """Assign final relevance ordering to an existing candidate pool."""

    def rerank(
        self,
        query_profile: QueryProfile,
        candidates: list[PlayerCandidate],
    ) -> list[RankedPlayerCandidate]:
        """Rerank candidates without evaluating evidence sufficiency."""
        ...
