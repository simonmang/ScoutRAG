"""Minimal adapters used before model-backed components exist."""

from scoutrag.domain.query import QueryProfile
from scoutrag.domain.retrieval import PlayerCandidate, RankedPlayerCandidate


class NoOpPlayerReranker:
    """Preserve fused order while providing the final ranking contract."""

    def rerank(
        self,
        query_profile: QueryProfile,
        candidates: list[PlayerCandidate],
    ) -> list[RankedPlayerCandidate]:
        del query_profile
        return [
            RankedPlayerCandidate(
                profile=candidate.profile,
                retrieval_trace=candidate.retrieval_trace,
                rank=rank,
                reranker_score=None,
                ranking_reasons=["Fused retrieval order; cross-encoder is not enabled."],
            )
            for rank, candidate in enumerate(candidates, start=1)
        ]
