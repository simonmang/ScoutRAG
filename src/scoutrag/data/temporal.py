"""Deterministic recent-form and multi-season trend derivation."""

from __future__ import annotations

from collections import defaultdict
from itertools import pairwise

from scoutrag.domain.player import (
    PlayerMatchPerformance,
    PlayerRecentForm,
    PlayerSeasonProfile,
    PlayerSeasonTrend,
    SeasonMetricObservation,
    TrendDirection,
)

_RATIO_COMPONENTS = {
    "duel_win_rate": ("duels_won", "duels"),
    "dribble_success_rate": ("dribbles_completed", "dribbles_attempted"),
    "shots_on_target_rate": ("shots_on_target", "shots"),
    "pass_completion_rate": ("passes_completed", "passes"),
}


def build_recent_form(
    performances: list[PlayerMatchPerformance],
    *,
    window_size: int = 5,
    full_window_minutes: float = 450,
) -> list[PlayerRecentForm]:
    """Compare the latest matches with earlier matches in the same profile."""

    if window_size < 1:
        raise ValueError("window_size must be positive")
    if full_window_minutes <= 0:
        raise ValueError("full_window_minutes must be positive")
    grouped: defaultdict[str, list[PlayerMatchPerformance]] = defaultdict(list)
    for performance in performances:
        grouped[performance.profile_id].append(performance)

    snapshots: list[PlayerRecentForm] = []
    for profile_id, items in grouped.items():
        ordered = sorted(
            items,
            key=lambda item: (
                item.match_date is not None,
                item.match_date,
                item.fixture_id,
            ),
        )
        recent = ordered[-window_size:]
        earlier = ordered[:-window_size]
        recent_features = _aggregate_match_features(recent)
        baseline_features = _aggregate_match_features(earlier)
        changes = {
            name: round((value - baseline) / abs(baseline), 4)
            for name, value in recent_features.items()
            if (baseline := baseline_features.get(name)) is not None and abs(baseline) > 1e-9
        }
        window_minutes = sum(item.minutes_played for item in recent)
        limitations: list[str] = []
        if len(recent) < window_size:
            limitations.append(
                f"Only {len(recent)} of {window_size} requested recent matches are available."
            )
        if window_minutes < full_window_minutes:
            limitations.append(
                f"Recent form covers only {window_minutes:g} of "
                f"{full_window_minutes:g} full-window minutes."
            )
        if not earlier:
            limitations.append("No earlier same-season matches are available for a form baseline.")
        weighted_quality = (
            sum(item.data_quality * item.minutes_played for item in recent) / window_minutes
        )
        quality = round(
            weighted_quality
            * min(window_minutes / full_window_minutes, 1)
            * (1 if earlier else 0.7),
            3,
        )
        snapshots.append(
            PlayerRecentForm(
                profile_id=profile_id,
                player_id=recent[-1].player_id,
                as_of_date=recent[-1].match_date,
                window_size=window_size,
                matches_in_window=len(recent),
                minutes_in_window=round(window_minutes, 3),
                fixture_ids=[item.fixture_id for item in recent],
                recent_features=recent_features,
                baseline_features=baseline_features,
                relative_changes=changes,
                data_quality=quality,
                limitations=limitations,
            )
        )
    return sorted(snapshots, key=lambda item: (item.profile_id, item.player_id))


def build_season_trends(
    profiles: list[PlayerSeasonProfile],
    *,
    maximum_seasons: int = 3,
    stable_percentile_band: float = 5,
) -> list[PlayerSeasonTrend]:
    """Build current-first trends while retaining each historical observation."""

    if maximum_seasons < 2:
        raise ValueError("maximum_seasons must be at least two")
    if stable_percentile_band < 0:
        raise ValueError("stable_percentile_band must be non-negative")
    grouped: defaultdict[str, list[PlayerSeasonProfile]] = defaultdict(list)
    for profile in profiles:
        grouped[profile.player_id].append(profile)

    trends: list[PlayerSeasonTrend] = []
    for player_id, player_profiles in grouped.items():
        latest_start = max(_season_start(profile.season_name) for profile in player_profiles)
        current_profiles = [
            profile
            for profile in player_profiles
            if _season_start(profile.season_name) == latest_start
        ]
        historical = [
            profile
            for profile in player_profiles
            if _season_start(profile.season_name) < latest_start
        ]
        for current in current_profiles:
            for metric_name, latest_value in current.structured_features.items():
                if metric_name not in current.percentiles:
                    continue
                observations = _metric_observations(
                    current,
                    historical,
                    metric_name=metric_name,
                    maximum_seasons=maximum_seasons,
                )
                previous = observations[-2] if len(observations) > 1 else None
                relative_change = (
                    round(
                        (latest_value - previous.normalized_value) / abs(previous.normalized_value),
                        4,
                    )
                    if previous is not None and abs(previous.normalized_value) > 1e-9
                    else None
                )
                direction = _trend_direction(
                    observations,
                    stable_percentile_band=stable_percentile_band,
                )
                trends.append(
                    PlayerSeasonTrend(
                        trend_id=f"{current.profile_id}:trend:{metric_name}",
                        player_id=player_id,
                        current_profile_id=current.profile_id or current.player_id,
                        metric_name=metric_name,
                        direction=direction,
                        latest_value=latest_value,
                        previous_value=(
                            previous.normalized_value if previous is not None else None
                        ),
                        relative_change=relative_change,
                        historical_fallback_available=len(observations) > 1,
                        observations=observations,
                        limitations=[
                            "Trend direction is descriptive and is not a performance forecast.",
                            "Percentiles remain relative to each season's league and position.",
                        ],
                    )
                )
    return sorted(trends, key=lambda item: (item.current_profile_id, item.metric_name))


def _aggregate_match_features(
    performances: list[PlayerMatchPerformance],
) -> dict[str, float]:
    if not performances:
        return {}
    minutes = sum(item.minutes_played for item in performances)
    raw_names = {
        name
        for item in performances
        for name in item.structured_features
        if not name.endswith("_per_90") and name not in _RATIO_COMPONENTS and name != "rating"
    }
    features: dict[str, float] = {}
    for name in raw_names:
        if not all(name in item.structured_features for item in performances):
            continue
        total = sum(item.structured_features[name] for item in performances)
        features[name] = round(total, 4)
        per_90_name = f"{name}_per_90"
        if any(per_90_name in item.structured_features for item in performances):
            features[per_90_name] = round((total / minutes) * 90, 4)
    ratings = [
        (item.structured_features["rating"], item.minutes_played)
        for item in performances
        if "rating" in item.structured_features
    ]
    if ratings:
        rating_minutes = sum(item_minutes for _, item_minutes in ratings)
        features["average_rating"] = round(
            sum(value * item_minutes for value, item_minutes in ratings) / rating_minutes,
            4,
        )
    for ratio_name, (numerator_name, denominator_name) in _RATIO_COMPONENTS.items():
        numerator = features.get(numerator_name)
        denominator = features.get(denominator_name)
        if numerator is not None and denominator is not None and denominator > 0:
            features[ratio_name] = round((numerator / denominator) * 100, 4)
    return features


def _metric_observations(
    current: PlayerSeasonProfile,
    historical: list[PlayerSeasonProfile],
    *,
    metric_name: str,
    maximum_seasons: int,
) -> list[SeasonMetricObservation]:
    by_season: defaultdict[int, list[PlayerSeasonProfile]] = defaultdict(list)
    for profile in historical:
        if metric_name in profile.structured_features:
            by_season[_season_start(profile.season_name)].append(profile)
    selected_historical = [
        max(items, key=lambda item: (item.minutes_played, item.data_quality))
        for _, items in sorted(by_season.items(), reverse=True)[: maximum_seasons - 1]
    ]
    selected = [*reversed(selected_historical), current]
    return [
        SeasonMetricObservation(
            profile_id=profile.profile_id or profile.player_id,
            competition_name=profile.competition_name,
            season_name=profile.season_name,
            team_names=profile.team_names,
            minutes_played=profile.minutes_played,
            normalized_value=profile.structured_features[metric_name],
            percentile=profile.percentiles.get(metric_name),
            data_quality=profile.data_quality,
        )
        for profile in selected
    ]


def _trend_direction(
    observations: list[SeasonMetricObservation],
    *,
    stable_percentile_band: float,
) -> TrendDirection:
    comparable = [
        observation.percentile for observation in observations if observation.percentile is not None
    ]
    if len(comparable) < 2:
        return TrendDirection.INSUFFICIENT
    deltas = [current - previous for previous, current in pairwise(comparable)]
    material = [delta for delta in deltas if abs(delta) > stable_percentile_band]
    if not material:
        return TrendDirection.STABLE
    if all(delta > 0 for delta in material):
        return TrendDirection.IMPROVING
    if all(delta < 0 for delta in material):
        return TrendDirection.DECLINING
    return TrendDirection.MIXED


def _season_start(season_name: str) -> int:
    try:
        return int(season_name[:4])
    except ValueError as exc:
        raise ValueError(f"season_name must begin with a four-digit year: {season_name}") from exc
