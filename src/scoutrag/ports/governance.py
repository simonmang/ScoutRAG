"""Port that separates evidence governance from retrieval scores."""

from typing import Protocol

from scoutrag.domain.evidence import RecommendationGovernance
from scoutrag.domain.player import PlayerMetricEvidence
from scoutrag.domain.query import QueryProfile
from scoutrag.domain.retrieval import RankedPlayerCandidate


class RecommendationGovernor(Protocol):
    """Assess whether a ranked result is sufficiently supported."""

    def evaluate(
        self,
        query_profile: QueryProfile,
        ranked_candidates: list[RankedPlayerCandidate],
        evidence: dict[str, list[PlayerMetricEvidence]],
    ) -> RecommendationGovernance:
        """Return a transparent verdict and evidence quality assessment."""
        ...
