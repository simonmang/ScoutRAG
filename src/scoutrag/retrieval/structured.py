"""Structured retrieval over filters, per-90 features, and percentiles."""

from dataclasses import dataclass

from scoutrag.domain.player import PlayerSeasonProfile
from scoutrag.domain.query import QueryIntent, QueryProfile
from scoutrag.domain.retrieval import CandidateRetrievalTrace, PlayerCandidate
from scoutrag.retrieval.common import matches_hard_filters


@dataclass(frozen=True, slots=True)
class StructuredRetrievalConfig:
    """Configurable broad-recall threshold for structured relevance."""

    minimum_score: float = 0.0

    def __post_init__(self) -> None:
        if not 0 <= self.minimum_score <= 1:
            raise ValueError("minimum_score must be between 0 and 1")


class StructuredFeaturePlayerRetriever:
    """Rank filtered profiles directly from typed statistical features."""

    strategy_name = "structured"

    def __init__(
        self,
        profiles: list[PlayerSeasonProfile],
        config: StructuredRetrievalConfig | None = None,
    ) -> None:
        self.profiles = tuple(profiles)
        self.config = config or StructuredRetrievalConfig()

    def retrieve(self, query_profile: QueryProfile, *, limit: int) -> list[PlayerCandidate]:
        if query_profile.intent is QueryIntent.OUT_OF_SCOPE:
            return []
        metrics = query_profile.requested_metrics
        filtered = [
            profile for profile in self.profiles if matches_hard_filters(profile, query_profile)
        ]
        if not metrics and not _has_structured_constraint(query_profile):
            return []

        ranges = _metric_ranges(filtered, metrics)
        scored: list[tuple[float, PlayerSeasonProfile]] = []
        for profile in filtered:
            score = _structured_score(profile, metrics, ranges)
            if score >= self.config.minimum_score:
                scored.append((score, profile))
        scored.sort(key=lambda item: (-item[0], item[1].player_name, item[1].season_name))
        return [
            PlayerCandidate(
                profile=profile,
                retrieval_trace=CandidateRetrievalTrace(
                    player_id=profile.player_id,
                    retrieved_by=[self.strategy_name],
                    structured_score=round(score, 6),
                    fused_score=round(score, 6),
                ),
            )
            for score, profile in scored[:limit]
        ]


def _has_structured_constraint(query: QueryProfile) -> bool:
    return bool(
        query.requested_positions
        or query.minimum_minutes is not None
        or query.competition_filters
        or query.season_filters
    )


def _metric_ranges(
    profiles: list[PlayerSeasonProfile],
    metrics: list[str],
) -> dict[str, tuple[float, float]]:
    ranges: dict[str, tuple[float, float]] = {}
    for metric in metrics:
        values = [
            profile.structured_features[metric]
            for profile in profiles
            if metric in profile.structured_features
        ]
        if values:
            ranges[metric] = min(values), max(values)
    return ranges


def _structured_score(
    profile: PlayerSeasonProfile,
    metrics: list[str],
    ranges: dict[str, tuple[float, float]],
) -> float:
    if not metrics:
        return 1.0
    scores: list[float] = []
    for metric in metrics:
        if metric in profile.percentiles:
            scores.append(profile.percentiles[metric] / 100)
            continue
        if metric not in profile.structured_features or metric not in ranges:
            continue
        minimum, maximum = ranges[metric]
        value = profile.structured_features[metric]
        scores.append((value - minimum) / (maximum - minimum) if maximum > minimum else 0.5)
    return sum(scores) / len(scores) if scores else 0.0
