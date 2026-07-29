"""Position-aware Phase 3 feature engineering without model inference."""

from collections import Counter, defaultdict
from dataclasses import dataclass

from scoutrag.data.metric_definitions import FEATURE_METRICS, FeatureMetricSpec, metric_definitions
from scoutrag.data.models import CompetitionSeason, MatchRecord
from scoutrag.domain.player import MetricDefinition, PlayerMetricEvidence, PlayerSeasonProfile


@dataclass(frozen=True, slots=True)
class FeatureEngineeringConfig:
    """Thresholds controlling comparability and evidence quality."""

    minimum_minutes: float = 450
    full_sample_minutes: float = 900
    minimum_comparison_group_size: int = 3
    minimum_source_coverage: float = 0.8

    def __post_init__(self) -> None:
        if self.minimum_minutes < 0:
            raise ValueError("minimum_minutes must be non-negative")
        if self.full_sample_minutes <= 0:
            raise ValueError("full_sample_minutes must be positive")
        if self.minimum_comparison_group_size < 2:
            raise ValueError("minimum_comparison_group_size must be at least 2")
        if not 0 < self.minimum_source_coverage <= 1:
            raise ValueError("minimum_source_coverage must be in (0, 1]")


@dataclass(frozen=True, slots=True)
class FeatureEngineeringResult:
    """Final profiles, evidence, and documentation generated for Phase 3."""

    profiles: list[PlayerSeasonProfile]
    evidence: list[PlayerMetricEvidence]
    definitions: list[MetricDefinition]


def engineer_player_features(
    competition: CompetitionSeason,
    matches: list[MatchRecord],
    raw_profiles: list[PlayerSeasonProfile],
    raw_evidence: list[PlayerMetricEvidence],
    *,
    config: FeatureEngineeringConfig | None = None,
) -> FeatureEngineeringResult:
    """Create normalized features and fair position-group percentiles."""
    settings = config or FeatureEngineeringConfig()
    team_match_counts = _team_match_counts(matches)
    maximum_team_matches = max(team_match_counts.values(), default=0)

    prepared: list[PlayerSeasonProfile] = []
    for profile in raw_profiles:
        features = dict(profile.structured_features)
        for spec in FEATURE_METRICS:
            features[spec.metric_name] = _calculate_feature(spec, features, profile.minutes_played)

        source_coverage = (
            team_match_counts.get(profile.team_name, 0) / maximum_team_matches
            if maximum_team_matches
            else 0
        )
        feature_coverage = sum(spec.metric_name in features for spec in FEATURE_METRICS) / len(
            FEATURE_METRICS
        )
        features["source_coverage_ratio"] = round(source_coverage, 4)
        features["feature_coverage_ratio"] = round(feature_coverage, 4)
        prepared.append(profile.model_copy(update={"structured_features": features}))

    eligible_groups: defaultdict[str, list[PlayerSeasonProfile]] = defaultdict(list)
    for profile in prepared:
        if _eligible(profile, settings):
            eligible_groups[profile.position_group].append(profile)

    finalized: list[PlayerSeasonProfile] = []
    derived_evidence: list[PlayerMetricEvidence] = []
    season_reference = (
        f"statsbomb:competitions/{competition.competition_id}/seasons/{competition.season_id}"
    )
    for profile in prepared:
        group = eligible_groups[profile.position_group]
        comparison_size = len(group)
        features = dict(profile.structured_features)
        features["comparison_group_size"] = float(comparison_size)
        percentiles: dict[str, float] = {}
        if (
            _eligible(profile, settings)
            and comparison_size >= settings.minimum_comparison_group_size
        ):
            for spec in FEATURE_METRICS:
                values = [candidate.structured_features[spec.metric_name] for candidate in group]
                percentiles[spec.metric_name] = _percentile(
                    profile.structured_features[spec.metric_name],
                    values,
                )

        data_quality = _data_quality(
            profile,
            feature_coverage=features["feature_coverage_ratio"],
            source_coverage=features["source_coverage_ratio"],
            comparison_size=comparison_size,
            settings=settings,
        )
        enriched = profile.model_copy(
            update={
                "structured_features": features,
                "percentiles": percentiles,
                "data_quality": data_quality,
                "profile_text": _profile_text(
                    profile,
                    features,
                    percentiles,
                    data_quality,
                    settings,
                ),
            }
        )
        finalized.append(enriched)

        comparison_group = (
            f"{competition.competition_name} {competition.season_name} "
            f"{profile.position_group} eligible source-covered profiles (n={comparison_size}, "
            f"min_minutes={settings.minimum_minutes:g}, "
            f"min_source_coverage={settings.minimum_source_coverage:g})"
        )
        for spec in FEATURE_METRICS:
            sample_size = (
                features.get("passes", 0) if spec.calculation == "rate" else profile.minutes_played
            )
            derived_evidence.append(
                PlayerMetricEvidence(
                    player_id=profile.player_id,
                    season_id=str(competition.season_id),
                    metric_name=spec.metric_name,
                    raw_value=features.get(spec.raw_metric),
                    normalized_value=features[spec.metric_name],
                    percentile=percentiles.get(spec.metric_name),
                    comparison_group=comparison_group,
                    sample_size=sample_size,
                    source_reference=(
                        f"{season_reference}/players/{profile.player_id}/features/"
                        f"{spec.metric_name}"
                    ),
                )
            )

    return FeatureEngineeringResult(
        profiles=finalized,
        evidence=[*raw_evidence, *derived_evidence],
        definitions=metric_definitions(),
    )


def _team_match_counts(matches: list[MatchRecord]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for match in matches:
        counts[match.home_team_name] += 1
        counts[match.away_team_name] += 1
    return counts


def _calculate_feature(
    spec: FeatureMetricSpec,
    raw_features: dict[str, float],
    minutes: float,
) -> float:
    numerator = raw_features.get(spec.raw_metric, 0)
    if spec.calculation == "rate":
        attempts = raw_features.get("passes", 0)
        return round((numerator / attempts) * 100, 4) if attempts else 0
    return round((numerator / minutes) * 90, 4) if minutes else 0


def _eligible(profile: PlayerSeasonProfile, settings: FeatureEngineeringConfig) -> bool:
    return (
        profile.minutes_played >= settings.minimum_minutes
        and profile.structured_features.get("source_coverage_ratio", 0)
        >= settings.minimum_source_coverage
    )


def _percentile(value: float, values: list[float]) -> float:
    """Tie-aware percentile rank on a stable inclusive 0..100 scale."""
    if len(values) == 1:
        return 50
    lower = sum(item < value for item in values)
    equal_others = sum(item == value for item in values) - 1
    return round(((lower + (0.5 * equal_others)) / (len(values) - 1)) * 100, 1)


def _data_quality(
    profile: PlayerSeasonProfile,
    *,
    feature_coverage: float,
    source_coverage: float,
    comparison_size: int,
    settings: FeatureEngineeringConfig,
) -> float:
    minutes_score = min(profile.minutes_played / settings.full_sample_minutes, 1)
    comparison_score = min(comparison_size / settings.minimum_comparison_group_size, 1)
    score = (
        (0.30 * source_coverage)
        + (0.30 * minutes_score)
        + (0.25 * feature_coverage)
        + (0.15 * comparison_score)
    )
    return round(min(max(score, 0), 1), 3)


def _profile_text(
    profile: PlayerSeasonProfile,
    features: dict[str, float],
    percentiles: dict[str, float],
    data_quality: float,
    settings: FeatureEngineeringConfig,
) -> str:
    base = (
        f"{profile.player_name} | {' / '.join(profile.team_names)} | "
        f"{profile.competition_name} {profile.season_name} | {profile.position_group} | "
        f"{profile.minutes_played:.1f} minutes | Evidence Quality Score {data_quality:.3f}."
    )
    if percentiles:
        labels = {spec.metric_name: spec.display_name for spec in FEATURE_METRICS}
        strongest = sorted(percentiles.items(), key=lambda item: (-item[1], item[0]))[:3]
        summary = ", ".join(f"{labels[name]} P{value:.0f}" for name, value in strongest)
        return f"{base} Highest position-group percentiles: {summary}."

    reasons: list[str] = []
    if profile.minutes_played < settings.minimum_minutes:
        reasons.append(f"fewer than {settings.minimum_minutes:g} minutes")
    if features["source_coverage_ratio"] < settings.minimum_source_coverage:
        reasons.append(
            f"source coverage {features['source_coverage_ratio']:.2f} below "
            f"{settings.minimum_source_coverage:.2f}"
        )
    if features["comparison_group_size"] < settings.minimum_comparison_group_size:
        reasons.append(
            f"comparison group n={int(features['comparison_group_size'])} below "
            f"{settings.minimum_comparison_group_size}"
        )
    return f"{base} No position percentile: {'; '.join(reasons) or 'not comparable'}."
