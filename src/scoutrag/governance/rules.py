"""Intent-aware, rule-based evidence quality governance."""

from dataclasses import dataclass
from statistics import fmean

from scoutrag.domain.evidence import EvidenceVerdict, RecommendationGovernance
from scoutrag.domain.player import PlayerMetricEvidence
from scoutrag.domain.query import QueryIntent, QueryProfile
from scoutrag.domain.retrieval import RankedPlayerCandidate
from scoutrag.retrieval.common import matches_hard_filters


@dataclass(frozen=True, slots=True)
class GovernanceThresholds:
    """Configurable boundaries for evidence sufficiency, never probabilities."""

    full_sample_minutes: float = 900
    expected_feature_count: int = 13
    retrieval_strategy_target: int = 2
    sufficient_score: float = 0.75
    insufficient_score: float = 0.45
    limited_factor_threshold: float = 0.60
    ranking_gap_target: float = 0.15
    conflict_relative_tolerance: float = 0.05
    minimum_discovery_candidates: int = 2

    def __post_init__(self) -> None:
        if self.full_sample_minutes <= 0:
            raise ValueError("full_sample_minutes must be positive")
        if self.expected_feature_count < 1:
            raise ValueError("expected_feature_count must be positive")
        if self.retrieval_strategy_target < 1:
            raise ValueError("retrieval_strategy_target must be positive")
        if not 0 <= self.insufficient_score < self.sufficient_score <= 1:
            raise ValueError("score thresholds must satisfy 0 <= insufficient < sufficient <= 1")
        if not 0 <= self.limited_factor_threshold <= 1:
            raise ValueError("limited_factor_threshold must be between 0 and 1")
        if self.ranking_gap_target <= 0:
            raise ValueError("ranking_gap_target must be positive")
        if self.conflict_relative_tolerance < 0:
            raise ValueError("conflict_relative_tolerance must be non-negative")
        if self.minimum_discovery_candidates < 1:
            raise ValueError("minimum_discovery_candidates must be positive")


class RuleBasedRecommendationGovernor:
    """Assess evidence sufficiency independently of recall and ranking."""

    def __init__(self, thresholds: GovernanceThresholds | None = None) -> None:
        self.thresholds = thresholds or GovernanceThresholds()

    def evaluate(
        self,
        query_profile: QueryProfile,
        ranked_candidates: list[RankedPlayerCandidate],
        evidence: dict[str, list[PlayerMetricEvidence]],
    ) -> RecommendationGovernance:
        if query_profile.intent is QueryIntent.OUT_OF_SCOPE:
            return RecommendationGovernance(
                verdict=EvidenceVerdict.OUT_OF_SCOPE,
                evidence_quality_score=0,
                reasons=["The query requests a function outside football player retrieval."],
                missing_evidence=["ScoutRAG does not provide match or tournament predictions."],
                factors={},
            )

        if not ranked_candidates:
            return RecommendationGovernance(
                verdict=EvidenceVerdict.INSUFFICIENT,
                evidence_quality_score=0,
                reasons=["No player satisfies the analyzed hard filters."],
                missing_evidence=["At least one matching player-season profile is required."],
                factors={},
            )

        factors = self._factors(query_profile, ranked_candidates, evidence)
        score = self._weighted_score(query_profile, factors)
        conflicts = self._conflicts(query_profile, ranked_candidates, evidence)
        if conflicts:
            return RecommendationGovernance(
                verdict=EvidenceVerdict.CONFLICTING,
                evidence_quality_score=score,
                reasons=conflicts,
                warnings=["Results may be displayed, but the conflict must be resolved first."],
                factors=factors,
            )

        missing = self._blocking_missing_evidence(query_profile, ranked_candidates, evidence)
        if factors["hard_filter_fulfillment"] < 1:
            missing.append("At least one returned profile violates a hard user filter.")
        if self._requires_ranked_choice(query_profile) and (
            len(ranked_candidates) < self.thresholds.minimum_discovery_candidates
        ):
            missing.append(
                "Too few matching player profiles are available for a comparative recommendation."
            )
        if missing:
            return RecommendationGovernance(
                verdict=EvidenceVerdict.INSUFFICIENT,
                evidence_quality_score=score,
                reasons=["The available evidence cannot support a reliable ranking."],
                missing_evidence=_unique(missing),
                factors=factors,
            )

        warnings = self._limitations(query_profile, ranked_candidates, factors)
        if score < self.thresholds.insufficient_score:
            return RecommendationGovernance(
                verdict=EvidenceVerdict.INSUFFICIENT,
                evidence_quality_score=score,
                reasons=["The combined evidence factors remain below the configured minimum."],
                missing_evidence=warnings or ["More complete player evidence is required."],
                factors=factors,
            )
        if warnings or score < self.thresholds.sufficient_score:
            return RecommendationGovernance(
                verdict=EvidenceVerdict.LIMITED,
                evidence_quality_score=score,
                reasons=["A result can be shown, but material evidence limitations remain."],
                warnings=warnings,
                factors=factors,
            )
        return RecommendationGovernance(
            verdict=EvidenceVerdict.SUFFICIENT,
            evidence_quality_score=score,
            reasons=["Requested evidence and comparison context meet the configured thresholds."],
            factors=factors,
        )

    def _factors(
        self,
        query: QueryProfile,
        candidates: list[RankedPlayerCandidate],
        evidence: dict[str, list[PlayerMetricEvidence]],
    ) -> dict[str, float]:
        requested = set(query.requested_metrics)
        relevant_evidence = [
            item
            for candidate in candidates
            for item in evidence.get(candidate.profile.player_id, [])
            if not requested or item.metric_name in requested
        ]
        requested_slots = len(candidates) * len(requested)
        available_requested = {
            (item.player_id, item.metric_name)
            for item in relevant_evidence
            if item.normalized_value is not None or item.raw_value is not None
        }
        comparable_requested = {
            (item.player_id, item.metric_name)
            for item in relevant_evidence
            if item.percentile is not None and item.comparison_group
        }

        target_minutes = max(
            float(query.minimum_minutes or 0),
            self.thresholds.full_sample_minutes,
        )
        feature_counts = [
            len(
                {
                    name
                    for name in candidate.profile.structured_features
                    if name
                    not in {
                        "source_coverage_ratio",
                        "feature_coverage_ratio",
                        "comparison_group_size",
                    }
                }
            )
            for candidate in candidates
        ]
        if requested:
            missing_fields = sum(
                value is None
                for item in relevant_evidence
                for value in (item.raw_value, item.normalized_value, item.percentile)
            )
            total_fields = max(len(relevant_evidence) * 3, requested_slots * 3)
            missing_value_score = 1 - (missing_fields / total_fields) if total_fields else 0
        else:
            missing_value_score = 1

        source_scores = [
            candidate.profile.structured_features.get(
                "source_coverage_ratio",
                candidate.profile.data_quality,
            )
            for candidate in candidates
        ]
        return {
            "data_coverage": _bounded(fmean(source_scores)),
            "played_minutes": _bounded(
                fmean(
                    min(candidate.profile.minutes_played / target_minutes, 1)
                    for candidate in candidates
                )
            ),
            "feature_availability": _bounded(
                fmean(
                    min(count / self.thresholds.expected_feature_count, 1)
                    for count in feature_counts
                )
            ),
            "requested_trait_coverage": (
                _bounded(len(available_requested) / requested_slots) if requested_slots else 1
            ),
            "retrieval_agreement": _bounded(
                fmean(
                    min(
                        len(candidate.retrieval_trace.retrieved_by)
                        / self.thresholds.retrieval_strategy_target,
                        1,
                    )
                    for candidate in candidates
                )
            ),
            "ranking_separation": self._ranking_separation(query, candidates),
            "comparison_group_availability": (
                _bounded(len(comparable_requested) / requested_slots) if requested_slots else 1
            ),
            "season_consistency": self._season_consistency(
                query,
                candidates,
                evidence,
            ),
            "missing_value_completeness": _bounded(missing_value_score),
            "hard_filter_fulfillment": _bounded(
                fmean(
                    float(matches_hard_filters(candidate.profile, query))
                    for candidate in candidates
                )
            ),
        }

    def _weighted_score(self, query: QueryProfile, factors: dict[str, float]) -> float:
        weights = {
            "data_coverage": 0.14,
            "played_minutes": 0.10,
            "feature_availability": 0.08,
            "requested_trait_coverage": 0.17,
            "retrieval_agreement": 0.09,
            "ranking_separation": 0.07,
            "comparison_group_availability": 0.14,
            "season_consistency": 0.08,
            "missing_value_completeness": 0.07,
            "hard_filter_fulfillment": 0.06,
        }
        if query.intent is QueryIntent.EXACT_PLAYER_LOOKUP and not query.requested_metrics:
            weights = {
                "data_coverage": 0.15,
                "played_minutes": 0.05,
                "feature_availability": 0.10,
                "requested_trait_coverage": 0.05,
                "retrieval_agreement": 0.15,
                "ranking_separation": 0.05,
                "comparison_group_availability": 0.05,
                "season_consistency": 0.15,
                "missing_value_completeness": 0.05,
                "hard_filter_fulfillment": 0.20,
            }
        return round(sum(factors[name] * weight for name, weight in weights.items()), 3)

    def _blocking_missing_evidence(
        self,
        query: QueryProfile,
        candidates: list[RankedPlayerCandidate],
        evidence: dict[str, list[PlayerMetricEvidence]],
    ) -> list[str]:
        if not query.requested_metrics:
            return []
        requested = set(query.requested_metrics)
        available_metrics = {
            item.metric_name
            for candidate in candidates
            for item in evidence.get(candidate.profile.player_id, [])
            if item.raw_value is not None or item.normalized_value is not None
        }
        missing_metrics = sorted(requested - available_metrics)
        messages = [f"Requested metric is unavailable: {metric}." for metric in missing_metrics]
        comparable = [
            item
            for candidate in candidates
            for item in evidence.get(candidate.profile.player_id, [])
            if item.metric_name in requested and item.percentile is not None
        ]
        if not comparable:
            messages.append("No requested metric has a valid position-specific comparison group.")
        return messages

    def _limitations(
        self,
        query: QueryProfile,
        candidates: list[RankedPlayerCandidate],
        factors: dict[str, float],
    ) -> list[str]:
        warnings: list[str] = []
        threshold = self.thresholds.limited_factor_threshold
        labels = {
            "data_coverage": "Source data coverage is limited.",
            "played_minutes": "At least one leading candidate has a limited minute sample.",
            "feature_availability": "Player feature coverage is incomplete.",
            "requested_trait_coverage": "Not every candidate covers every requested metric.",
            "retrieval_agreement": "Independent retrieval strategies show weak agreement.",
            "ranking_separation": "The leading candidates have little ranking separation.",
            "comparison_group_availability": "Some requested metrics lack comparable percentiles.",
            "season_consistency": "The result spans inconsistent season context.",
            "missing_value_completeness": (
                "A material share of requested evidence values is missing."
            ),
        }
        for name, message in labels.items():
            if factors[name] < threshold:
                warnings.append(message)
        if self._requires_ranked_choice(query) and len(candidates) < query.result_count:
            warnings.append(
                f"Only {len(candidates)} of {query.result_count} requested results are available."
            )
        return warnings

    def _conflicts(
        self,
        query: QueryProfile,
        candidates: list[RankedPlayerCandidate],
        evidence: dict[str, list[PlayerMetricEvidence]],
    ) -> list[str]:
        conflicts: list[str] = []
        if len(query.season_filters) > 1:
            conflicts.append(
                "The query contains multiple season filters that cannot form one season ranking."
            )
        seasons_by_player: dict[str, set[str]] = {}
        for candidate in candidates:
            seasons_by_player.setdefault(candidate.profile.player_id, set()).add(
                candidate.profile.season_name
            )
        if any(len(seasons) > 1 for seasons in seasons_by_player.values()):
            conflicts.append(
                "The same player appears in multiple seasons in one candidate ranking."
            )

        values: dict[tuple[str, str, str], list[float]] = {}
        for player_id, items in evidence.items():
            evidence_seasons = {item.season_id for item in items}
            if len(evidence_seasons) > 1:
                conflicts.append(
                    f"Metric evidence for player={player_id} spans multiple season IDs."
                )
            for item in items:
                value = (
                    item.normalized_value if item.normalized_value is not None else item.raw_value
                )
                if value is not None:
                    values.setdefault(
                        (player_id, item.season_id, item.metric_name),
                        [],
                    ).append(value)
        for key, observations in values.items():
            if len(observations) < 2:
                continue
            scale = max(max(abs(value) for value in observations), 1)
            if (max(observations) - min(observations)) / scale > (
                self.thresholds.conflict_relative_tolerance
            ):
                conflicts.append(
                    "Conflicting values exist for "
                    f"player={key[0]}, season={key[1]}, metric={key[2]}."
                )
        return conflicts

    def _ranking_separation(
        self,
        query: QueryProfile,
        candidates: list[RankedPlayerCandidate],
    ) -> float:
        if len(candidates) < 2 or query.intent is QueryIntent.EXACT_PLAYER_LOOKUP:
            return 1
        scores = [
            candidate.reranker_score
            if candidate.reranker_score is not None
            else candidate.retrieval_trace.fused_score
            for candidate in candidates
        ]
        score_range = max(scores) - min(scores)
        if score_range <= 0:
            return 0
        normalized_gap = max(scores[0] - scores[1], 0) / score_range
        return _bounded(normalized_gap / self.thresholds.ranking_gap_target)

    @staticmethod
    def _season_consistency(
        query: QueryProfile,
        candidates: list[RankedPlayerCandidate],
        evidence: dict[str, list[PlayerMetricEvidence]],
    ) -> float:
        seasons = {candidate.profile.season_name for candidate in candidates}
        if len(seasons) != 1:
            return 0
        if any(
            len({item.season_id for item in evidence.get(candidate.profile.player_id, [])}) > 1
            for candidate in candidates
        ):
            return 0
        if query.season_filters:
            return float(next(iter(seasons)) in query.season_filters)
        return 1

    @staticmethod
    def _requires_ranked_choice(query: QueryProfile) -> bool:
        return query.intent in {
            QueryIntent.PLAYER_DISCOVERY,
            QueryIntent.SIMILAR_PLAYER,
            QueryIntent.AGGREGATION,
        }


def _bounded(value: float) -> float:
    return round(min(max(float(value), 0), 1), 3)


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))
