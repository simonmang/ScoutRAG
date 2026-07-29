"""Per-strategy score normalization and weighted retrieval fusion."""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from scoutrag.domain.query import QueryProfile
from scoutrag.domain.retrieval import CandidateRetrievalTrace, PlayerCandidate
from scoutrag.retrieval.common import profile_key

STRATEGY_SCORE_FIELDS = {
    "exact": "exact_score",
    "structured": "structured_score",
    "sparse": "sparse_score",
    "dense": "dense_score",
}


@dataclass(frozen=True, slots=True)
class FusionWeights:
    """Configurable weights whose sum defines one normalized fused score."""

    dense: float = 0.30
    sparse: float = 0.25
    structured: float = 0.30
    exact: float = 0.15

    def __post_init__(self) -> None:
        values = (self.dense, self.sparse, self.structured, self.exact)
        if any(value < 0 for value in values):
            raise ValueError("fusion weights cannot be negative")
        if not math.isclose(sum(values), 1.0, abs_tol=1e-9):
            raise ValueError("fusion weights must sum to 1.0")

    def as_dict(self) -> dict[str, float]:
        return {
            "dense": self.dense,
            "sparse": self.sparse,
            "structured": self.structured,
            "exact": self.exact,
        }


class WeightedRetrievalFusion:
    """Normalize independent score scales and reward cross-strategy agreement."""

    def __init__(self, weights: FusionWeights | None = None) -> None:
        self.weights = weights or FusionWeights()

    def fuse(
        self,
        query_profile: QueryProfile,
        candidates_by_strategy: Mapping[str, Sequence[PlayerCandidate]],
        *,
        limit: int,
    ) -> list[PlayerCandidate]:
        del query_profile
        normalized = {
            strategy: self._normalized_strategy_scores(strategy, candidates)
            for strategy, candidates in candidates_by_strategy.items()
            if strategy in STRATEGY_SCORE_FIELDS
        }
        profiles = {
            profile_key(candidate.profile): candidate.profile
            for candidates in candidates_by_strategy.values()
            for candidate in candidates
        }
        strategy_order = tuple(self.weights.as_dict())
        fused: list[PlayerCandidate] = []
        for key, profile in profiles.items():
            scores = {
                strategy: strategy_scores[key]
                for strategy, strategy_scores in normalized.items()
                if key in strategy_scores
            }
            retrieved_by = [strategy for strategy in strategy_order if strategy in scores]
            fused_score = sum(
                self.weights.as_dict()[strategy] * score for strategy, score in scores.items()
            )
            fused.append(
                PlayerCandidate(
                    profile=profile,
                    retrieval_trace=CandidateRetrievalTrace(
                        player_id=profile.player_id,
                        retrieved_by=retrieved_by,
                        dense_score=scores.get("dense"),
                        sparse_score=scores.get("sparse"),
                        structured_score=scores.get("structured"),
                        exact_score=scores.get("exact"),
                        fused_score=round(fused_score, 6),
                    ),
                )
            )
        fused.sort(
            key=lambda candidate: (
                -candidate.retrieval_trace.fused_score,
                candidate.profile.player_name,
                candidate.profile.season_name,
            )
        )
        return fused[:limit]

    @staticmethod
    def _normalized_strategy_scores(
        strategy: str,
        candidates: Sequence[PlayerCandidate],
    ) -> dict[tuple[str, str, str], float]:
        field = STRATEGY_SCORE_FIELDS[strategy]
        raw_scores = [
            float(score)
            for candidate in candidates
            if (score := getattr(candidate.retrieval_trace, field)) is not None
        ]
        if not raw_scores:
            return {}
        minimum = min(raw_scores)
        maximum = max(raw_scores)
        return {
            profile_key(candidate.profile): _min_max(
                float(score),
                minimum,
                maximum,
            )
            for candidate in candidates
            if (score := getattr(candidate.retrieval_trace, field)) is not None
        }


def _min_max(value: float, minimum: float, maximum: float) -> float:
    if maximum == minimum:
        return 1.0
    return round((value - minimum) / (maximum - minimum), 6)
