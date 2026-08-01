"""Convert API-Football ``/players`` payloads into canonical ScoutRAG artifacts.

The provider exposes season aggregates rather than event data.  Consequently this
module only derives rates from counts that are actually present in the response.
In particular, it does not manufacture event-only concepts such as pressures or
progressive passes.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from scoutrag.domain.base import ScoutRAGModel
from scoutrag.domain.player import (
    MetricDefinition,
    PlayerIdentity,
    PlayerMatchPerformance,
    PlayerMetricEvidence,
    PlayerRecentForm,
    PlayerSeasonProfile,
    PlayerSeasonTrend,
    PlayerTeamSeasonStint,
)


@dataclass(frozen=True, slots=True)
class ApiFootballMetricSpec:
    """Mapping from one documented provider count to a per-90 feature."""

    raw_metric: str
    metric_name: str
    display_name: str
    source_path: tuple[str, ...]
    description: str
    higher_is_better: bool = True
    summary_eligible: bool = False

    def definition(self) -> MetricDefinition:
        source_field = ".".join(("players", "statistics", *self.source_path))
        direction = (
            "higher values rank higher" if self.higher_is_better else "lower values rank higher"
        )
        return MetricDefinition(
            metric_name=self.metric_name,
            display_name=self.display_name,
            description=self.description,
            calculation_method=f"{self.raw_metric} / minutes_played * 90",
            required_event_types=[source_field, "players.statistics.games.minutes"],
            limitations=[
                "The value is a provider season aggregate, not an event-level reconstruction.",
                "Per-90 volume does not adjust for team tactics, possession, or opponent strength.",
                "A missing provider value is withheld rather than interpreted as zero.",
                f"Percentile desirability direction: {direction}.",
            ],
        )


@dataclass(frozen=True, slots=True)
class ApiFootballCountSpec:
    """A season count or non-ranking context value copied from provider blocks."""

    metric_name: str
    display_name: str
    source_path: tuple[str, ...]
    description: str
    aggregation: str = "sum"

    def definition(self) -> MetricDefinition:
        source_field = ".".join(("players", "statistics", *self.source_path))
        calculation = {
            "sum": f"sum({source_field}) across unique team blocks",
            "any": f"true when any complete team block reports {source_field}=true",
            "consistent": f"retain {source_field} only when all reported values agree",
        }[self.aggregation]
        return MetricDefinition(
            metric_name=self.metric_name,
            display_name=self.display_name,
            description=self.description,
            calculation_method=calculation,
            required_event_types=[source_field],
            limitations=[
                "This is provider season context and is not used for percentile ranking.",
                "A missing provider value is withheld rather than interpreted as zero.",
            ],
        )


@dataclass(frozen=True, slots=True)
class ApiFootballAggregateSpec:
    """A provider average combined across transfer blocks using exposure weights."""

    metric_name: str
    display_name: str
    source_path: tuple[str, ...]
    weight_path: tuple[str, ...]
    description: str
    summary_eligible: bool = False

    def definition(self) -> MetricDefinition:
        source_field = ".".join(("players", "statistics", *self.source_path))
        weight_field = ".".join(("players", "statistics", *self.weight_path))
        return MetricDefinition(
            metric_name=self.metric_name,
            display_name=self.display_name,
            description=self.description,
            calculation_method=(
                f"appearance-weighted mean of {source_field} across unique team blocks"
            ),
            required_event_types=[source_field, weight_field],
            limitations=[
                "The provider field is an aggregate and is not reconstructed from matches.",
                "Every positive-exposure team block must report the value; "
                "otherwise it is withheld.",
                "Percentile desirability direction: higher values rank higher.",
            ],
        )


@dataclass(frozen=True, slots=True)
class ApiFootballRatioSpec:
    """A bounded rate derived from two complete season counts."""

    metric_name: str
    display_name: str
    numerator_metric: str
    denominator_metric: str
    description: str

    def definition(self) -> MetricDefinition:
        return MetricDefinition(
            metric_name=self.metric_name,
            display_name=self.display_name,
            description=self.description,
            calculation_method=f"{self.numerator_metric} / {self.denominator_metric} * 100",
            required_event_types=[self.numerator_metric, self.denominator_metric],
            limitations=[
                "The rate is withheld when either count is missing or the denominator is zero.",
                "The provider counts are season aggregates and may contain provider corrections.",
                "Percentile desirability direction: higher values rank higher.",
            ],
        )


API_FOOTBALL_METRICS: tuple[ApiFootballMetricSpec, ...] = (
    ApiFootballMetricSpec(
        "shots",
        "shots_per_90",
        "Shots per 90",
        ("shots", "total"),
        "Shots recorded by API-Football per 90 reported minutes.",
    ),
    ApiFootballMetricSpec(
        "shots_on_target",
        "shots_on_target_per_90",
        "Shots on target per 90",
        ("shots", "on"),
        "Shots on target recorded by API-Football per 90 reported minutes.",
    ),
    ApiFootballMetricSpec(
        "goals",
        "goals_per_90",
        "Goals per 90",
        ("goals", "total"),
        "Goals recorded by API-Football per 90 reported minutes.",
        summary_eligible=True,
    ),
    ApiFootballMetricSpec(
        "assists",
        "assists_per_90",
        "Assists per 90",
        ("goals", "assists"),
        "Assists recorded by API-Football per 90 reported minutes.",
        summary_eligible=True,
    ),
    ApiFootballMetricSpec(
        "goals_conceded",
        "goals_conceded_per_90",
        "Goals conceded per 90",
        ("goals", "conceded"),
        "Goals conceded recorded for the player per 90 reported minutes.",
        higher_is_better=False,
    ),
    ApiFootballMetricSpec(
        "saves",
        "saves_per_90",
        "Saves per 90",
        ("goals", "saves"),
        "Saves recorded by API-Football per 90 reported minutes.",
        summary_eligible=True,
    ),
    ApiFootballMetricSpec(
        "passes",
        "passes_per_90",
        "Passes per 90",
        ("passes", "total"),
        "Passes recorded by API-Football per 90 reported minutes.",
    ),
    ApiFootballMetricSpec(
        "key_passes",
        "key_passes_per_90",
        "Key passes per 90",
        ("passes", "key"),
        "Key passes recorded by API-Football per 90 reported minutes.",
        summary_eligible=True,
    ),
    ApiFootballMetricSpec(
        "tackles",
        "tackles_per_90",
        "Tackles per 90",
        ("tackles", "total"),
        "Tackles recorded by API-Football per 90 reported minutes.",
    ),
    ApiFootballMetricSpec(
        "blocks",
        "blocks_per_90",
        "Blocks per 90",
        ("tackles", "blocks"),
        "Blocks recorded by API-Football per 90 reported minutes.",
    ),
    ApiFootballMetricSpec(
        "interceptions",
        "interceptions_per_90",
        "Interceptions per 90",
        ("tackles", "interceptions"),
        "Interceptions recorded by API-Football per 90 reported minutes.",
        summary_eligible=True,
    ),
    ApiFootballMetricSpec(
        "duels",
        "duels_per_90",
        "Duels per 90",
        ("duels", "total"),
        "Duels recorded by API-Football per 90 reported minutes.",
    ),
    ApiFootballMetricSpec(
        "duels_won",
        "duels_won_per_90",
        "Duels won per 90",
        ("duels", "won"),
        "Duels won recorded by API-Football per 90 reported minutes.",
    ),
    ApiFootballMetricSpec(
        "dribbles_attempted",
        "dribbles_attempted_per_90",
        "Dribbles attempted per 90",
        ("dribbles", "attempts"),
        "Dribbles attempted recorded by API-Football per 90 reported minutes.",
    ),
    ApiFootballMetricSpec(
        "dribbles_completed",
        "dribbles_completed_per_90",
        "Dribbles completed per 90",
        ("dribbles", "success"),
        "Successful dribbles recorded by API-Football per 90 reported minutes.",
        summary_eligible=True,
    ),
    ApiFootballMetricSpec(
        "dribbles_past",
        "dribbles_past_per_90",
        "Times dribbled past per 90",
        ("dribbles", "past"),
        "Times the player was dribbled past per 90 reported minutes.",
        higher_is_better=False,
    ),
    ApiFootballMetricSpec(
        "fouls_drawn",
        "fouls_drawn_per_90",
        "Fouls drawn per 90",
        ("fouls", "drawn"),
        "Fouls drawn recorded by API-Football per 90 reported minutes.",
    ),
    ApiFootballMetricSpec(
        "fouls_committed",
        "fouls_committed_per_90",
        "Fouls committed per 90",
        ("fouls", "committed"),
        "Fouls committed recorded by API-Football per 90 reported minutes.",
        higher_is_better=False,
    ),
    ApiFootballMetricSpec(
        "yellow_cards",
        "yellow_cards_per_90",
        "Yellow cards per 90",
        ("cards", "yellow"),
        "Yellow cards recorded by API-Football per 90 reported minutes.",
        higher_is_better=False,
    ),
    ApiFootballMetricSpec(
        "yellow_red_cards",
        "yellow_red_cards_per_90",
        "Second-yellow red cards per 90",
        ("cards", "yellowred"),
        "Second-yellow dismissals recorded by API-Football per 90 reported minutes.",
        higher_is_better=False,
    ),
    ApiFootballMetricSpec(
        "red_cards",
        "red_cards_per_90",
        "Red cards per 90",
        ("cards", "red"),
        "Red cards recorded by API-Football per 90 reported minutes.",
        higher_is_better=False,
    ),
    ApiFootballMetricSpec(
        "penalties_won",
        "penalties_won_per_90",
        "Penalties won per 90",
        ("penalty", "won"),
        "Penalties won recorded by API-Football per 90 reported minutes.",
        summary_eligible=True,
    ),
    ApiFootballMetricSpec(
        "penalties_committed",
        "penalties_committed_per_90",
        "Penalties committed per 90",
        ("penalty", "commited"),
        "Penalties committed recorded by API-Football per 90 reported minutes.",
        higher_is_better=False,
    ),
    ApiFootballMetricSpec(
        "penalties_scored",
        "penalties_scored_per_90",
        "Penalties scored per 90",
        ("penalty", "scored"),
        "Penalties scored recorded by API-Football per 90 reported minutes.",
        summary_eligible=True,
    ),
    ApiFootballMetricSpec(
        "penalties_missed",
        "penalties_missed_per_90",
        "Penalties missed per 90",
        ("penalty", "missed"),
        "Penalties missed recorded by API-Football per 90 reported minutes.",
        higher_is_better=False,
    ),
    ApiFootballMetricSpec(
        "penalties_saved",
        "penalties_saved_per_90",
        "Penalties saved per 90",
        ("penalty", "saved"),
        "Penalties saved recorded by API-Football per 90 reported minutes.",
        summary_eligible=True,
    ),
)

API_FOOTBALL_COUNTS: tuple[ApiFootballCountSpec, ...] = (
    ApiFootballCountSpec(
        "appearances",
        "Appearances",
        ("games", "appearences"),
        "Appearances reported for the selected league and season.",
    ),
    ApiFootballCountSpec(
        "starts",
        "Starts",
        ("games", "lineups"),
        "Starting-lineup appearances reported for the selected league and season.",
    ),
    ApiFootballCountSpec(
        "substitutions_in",
        "Substitutions in",
        ("substitutes", "in"),
        "Substitute appearances reported for the selected league and season.",
    ),
    ApiFootballCountSpec(
        "substitutions_out",
        "Substitutions out",
        ("substitutes", "out"),
        "Times substituted off reported for the selected league and season.",
    ),
    ApiFootballCountSpec(
        "bench_appearances",
        "Bench appearances",
        ("substitutes", "bench"),
        "Bench selections reported for the selected league and season.",
    ),
    ApiFootballCountSpec(
        "captain_flag",
        "Captain flag",
        ("games", "captain"),
        "Whether API-Football marks the player as captain in any included team block.",
        aggregation="any",
    ),
    ApiFootballCountSpec(
        "shirt_number",
        "Shirt number",
        ("games", "number"),
        "Shirt number when the provider reports one unambiguous value.",
        aggregation="consistent",
    ),
)

API_FOOTBALL_AGGREGATES: tuple[ApiFootballAggregateSpec, ...] = (
    ApiFootballAggregateSpec(
        "average_rating",
        "Average provider rating",
        ("games", "rating"),
        ("games", "appearences"),
        "API-Football's season rating combined across team blocks.",
        summary_eligible=True,
    ),
    ApiFootballAggregateSpec(
        "pass_accuracy",
        "Provider pass accuracy",
        ("passes", "accuracy"),
        ("games", "appearences"),
        "API-Football's pass-accuracy aggregate combined across team blocks.",
    ),
)

API_FOOTBALL_RATIOS: tuple[ApiFootballRatioSpec, ...] = (
    ApiFootballRatioSpec(
        "shots_on_target_rate",
        "Shots on target rate",
        "shots_on_target",
        "shots",
        "Percentage of recorded shots that were on target.",
    ),
    ApiFootballRatioSpec(
        "duel_win_rate",
        "Duel win rate",
        "duels_won",
        "duels",
        "Percentage of recorded duels won.",
    ),
    ApiFootballRatioSpec(
        "dribble_success_rate",
        "Dribble success rate",
        "dribbles_completed",
        "dribbles_attempted",
        "Percentage of recorded dribble attempts completed.",
    ),
)


@dataclass(frozen=True, slots=True)
class ApiFootballProfileResult:
    """Canonical, LLM-free output derived from API-Football aggregates."""

    profiles: list[PlayerSeasonProfile]
    evidence: list[PlayerMetricEvidence]
    definitions: list[MetricDefinition]
    limitations: list[str]
    identities: list[PlayerIdentity] = field(default_factory=list)
    stints: list[PlayerTeamSeasonStint] = field(default_factory=list)
    match_performances: list[PlayerMatchPerformance] = field(default_factory=list)
    recent_forms: list[PlayerRecentForm] = field(default_factory=list)
    season_trends: list[PlayerSeasonTrend] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class _PlayerAggregate:
    player_id: int
    player_name: str
    player: dict[str, Any]
    blocks: tuple[dict[str, Any], ...]


def build_api_football_profiles(
    payloads: list[dict[str, Any]],
    *,
    league_id: int,
    season_start_year: int,
    competition_name: str,
    minimum_minutes: float = 450,
    full_sample_minutes: float = 900,
    minimum_comparison_group_size: int = 3,
    comparison_scope: str | None = None,
    enable_percentiles: bool = True,
) -> ApiFootballProfileResult:
    """Build profiles from one league-season worth of ``/players`` response items.

    Multiple statistics blocks for one player are retained only when their league
    and season match the requested scope.  Counts are summed across unique team
    blocks (for example after a transfer).  If any included block omits a metric,
    the combined metric is withheld rather than silently treating the gap as zero.
    """
    _validate_settings(
        league_id,
        season_start_year,
        competition_name,
        minimum_minutes,
        full_sample_minutes,
        minimum_comparison_group_size,
    )
    aggregates = _collect_players(payloads, league_id, season_start_year)
    season_name = f"{season_start_year}/{season_start_year + 1}"
    season_id = f"api-football:{league_id}:{season_start_year}"

    prepared: list[PlayerSeasonProfile] = []
    metric_completeness: dict[str, float] = {}
    for aggregate in aggregates:
        profile, completeness = _prepare_profile(
            aggregate,
            competition_name=competition_name,
            season_name=season_name,
            full_sample_minutes=full_sample_minutes,
        )
        if profile is not None:
            profile = profile.model_copy(
                update={"profile_id": f"{season_id}:{aggregate.player_id}"}
            )
            prepared.append(profile)
            metric_completeness[profile.player_id] = completeness

    scope_label = comparison_scope or f"{competition_name} {season_name}"
    eligible_groups: defaultdict[str, list[PlayerSeasonProfile]] = defaultdict(list)
    for profile in prepared:
        if profile.minutes_played >= minimum_minutes:
            eligible_groups[profile.position_group].append(profile)

    profiles: list[PlayerSeasonProfile] = []
    evidence: list[PlayerMetricEvidence] = []
    percentile_specs = _percentile_specs()
    labels = {name: display_name for name, display_name, _, _ in percentile_specs}
    for profile in prepared:
        peers = eligible_groups[profile.position_group]
        percentiles: dict[str, float] = {}
        if (
            enable_percentiles
            and profile.minutes_played >= minimum_minutes
            and len(peers) >= minimum_comparison_group_size
        ):
            for metric_name, _, higher_is_better, _ in percentile_specs:
                if metric_name not in profile.structured_features:
                    continue
                peer_values = [
                    peer.structured_features[metric_name]
                    for peer in peers
                    if metric_name in peer.structured_features
                ]
                if len(peer_values) >= minimum_comparison_group_size:
                    percentiles[metric_name] = _percentile(
                        profile.structured_features[metric_name],
                        peer_values,
                        higher_is_better=higher_is_better,
                    )

        comparison_score = (
            min(len(peers) / minimum_comparison_group_size, 1) if enable_percentiles else 0
        )
        minutes_score = min(profile.minutes_played / full_sample_minutes, 1)
        quality = round(
            (0.50 * minutes_score)
            + (0.35 * metric_completeness[profile.player_id])
            + (0.15 * comparison_score),
            3,
        )
        summary_metrics = {
            name for name, _, _, summary_eligible in percentile_specs if summary_eligible
        }
        strongest = sorted(
            ((name, value) for name, value in percentiles.items() if name in summary_metrics),
            key=lambda item: (-item[1], item[0]),
        )[:3]
        percentile_text = (
            " Highest position-group percentiles: "
            + ", ".join(f"{labels[name]} P{value:.0f}" for name, value in strongest)
            + "."
            if strongest
            else (
                " No position percentile: the player or comparison group does not meet "
                "the configured evidence thresholds."
            )
        )
        enriched = profile.model_copy(
            update={
                "percentiles": percentiles,
                "data_quality": quality,
                "profile_text": (
                    f"{profile.player_name} | {' / '.join(profile.team_names)} | "
                    f"{competition_name} {season_name} | {profile.position_group} | "
                    f"{profile.minutes_played:.1f} minutes | "
                    f"API-Football aggregate | Evidence Quality Score {quality:.3f}."
                    f"{percentile_text}"
                ),
            }
        )
        profiles.append(enriched)

        comparison_group = (
            f"{scope_label} {profile.position_group} eligible API-Football profiles "
            f"(n={len(peers)}, min_minutes={minimum_minutes:g}, "
            f"percentiles={'enabled' if enable_percentiles else 'disabled'})"
        )
        player_number = profile.player_id.removeprefix("api-football:")
        base_source = (
            f"api-football:/players?league={league_id}&season={season_start_year}"
            f"&id={player_number}"
        )
        evidence.append(
            PlayerMetricEvidence(
                player_id=profile.player_id,
                season_id=season_id,
                metric_name="minutes_played",
                raw_value=profile.minutes_played,
                normalized_value=None,
                percentile=None,
                comparison_group=comparison_group,
                sample_size=profile.minutes_played,
                source_reference=f"{base_source}#statistics.games.minutes",
            )
        )
        if "age_at_season_start" in profile.structured_features:
            evidence.append(
                PlayerMetricEvidence(
                    player_id=profile.player_id,
                    season_id=season_id,
                    metric_name="age_at_season_start",
                    raw_value=profile.structured_features["age_at_season_start"],
                    normalized_value=None,
                    percentile=None,
                    comparison_group=comparison_group,
                    sample_size=None,
                    source_reference=f"{base_source}#player.birth.date",
                )
            )
        for count_spec in API_FOOTBALL_COUNTS:
            if count_spec.metric_name not in profile.structured_features:
                continue
            evidence.append(
                PlayerMetricEvidence(
                    player_id=profile.player_id,
                    season_id=season_id,
                    metric_name=count_spec.metric_name,
                    raw_value=profile.structured_features[count_spec.metric_name],
                    normalized_value=None,
                    percentile=None,
                    comparison_group=comparison_group,
                    sample_size=profile.minutes_played,
                    source_reference=(
                        f"{base_source}#statistics.{'.'.join(count_spec.source_path)}"
                    ),
                )
            )
        for aggregate_spec in API_FOOTBALL_AGGREGATES:
            if aggregate_spec.metric_name not in profile.structured_features:
                continue
            value = profile.structured_features[aggregate_spec.metric_name]
            evidence.append(
                PlayerMetricEvidence(
                    player_id=profile.player_id,
                    season_id=season_id,
                    metric_name=aggregate_spec.metric_name,
                    raw_value=value,
                    normalized_value=value,
                    percentile=percentiles.get(aggregate_spec.metric_name),
                    comparison_group=comparison_group,
                    sample_size=profile.structured_features.get("appearances"),
                    source_reference=(
                        f"{base_source}#statistics.{'.'.join(aggregate_spec.source_path)}"
                    ),
                )
            )
        for metric_spec in API_FOOTBALL_METRICS:
            if metric_spec.raw_metric not in profile.structured_features:
                continue
            evidence.append(
                PlayerMetricEvidence(
                    player_id=profile.player_id,
                    season_id=season_id,
                    metric_name=metric_spec.metric_name,
                    raw_value=profile.structured_features[metric_spec.raw_metric],
                    normalized_value=profile.structured_features.get(metric_spec.metric_name),
                    percentile=percentiles.get(metric_spec.metric_name),
                    comparison_group=comparison_group,
                    sample_size=profile.minutes_played,
                    source_reference=(
                        f"{base_source}#statistics.{'.'.join(metric_spec.source_path)}"
                    ),
                )
            )
        for ratio_spec in API_FOOTBALL_RATIOS:
            if ratio_spec.metric_name not in profile.structured_features:
                continue
            evidence.append(
                PlayerMetricEvidence(
                    player_id=profile.player_id,
                    season_id=season_id,
                    metric_name=ratio_spec.metric_name,
                    raw_value=None,
                    normalized_value=profile.structured_features[ratio_spec.metric_name],
                    percentile=percentiles.get(ratio_spec.metric_name),
                    comparison_group=comparison_group,
                    sample_size=profile.structured_features.get(ratio_spec.denominator_metric),
                    source_reference=(
                        f"{base_source}#derived."
                        f"{ratio_spec.numerator_metric}/{ratio_spec.denominator_metric}"
                    ),
                )
            )

    limitations: list[str] = []
    if not enable_percentiles:
        limitations.append(
            "Position-group percentiles were disabled because the supplied payload may not "
            "represent a complete competition comparison group."
        )
    elif comparison_scope is not None:
        limitations.append(f"Percentiles use only the declared comparison scope: {scope_label}.")
    profile_ids = {profile.player_id: profile.profile_id for profile in profiles}
    evidence = [
        item.model_copy(update={"profile_id": profile_ids[item.player_id]}) for item in evidence
    ]
    return ApiFootballProfileResult(
        profiles=sorted(profiles, key=lambda item: (item.player_name.casefold(), item.player_id)),
        evidence=sorted(
            evidence,
            key=lambda item: (item.player_id, item.metric_name),
        ),
        definitions=api_football_metric_definitions(),
        limitations=limitations,
    )


def api_football_metric_definitions() -> list[MetricDefinition]:
    """Return the provider-specific metric documentation in stable order."""
    return [
        MetricDefinition(
            metric_name="minutes_played",
            display_name="Minutes played",
            description="Minutes reported by API-Football for the selected league and season.",
            calculation_method="sum(players.statistics.games.minutes) across unique team blocks",
            required_event_types=["players.statistics.games.minutes"],
            limitations=[
                "Minutes are provider aggregates and are not reconstructed from lineups.",
                "Profiles without any reported minutes are excluded.",
            ],
        ),
        MetricDefinition(
            metric_name="age_at_season_start",
            display_name="Age at season start",
            description="Player age on 1 July of the selected season start year.",
            calculation_method="completed years between player.birth.date and season-start 1 July",
            required_event_types=["players.player.birth.date"],
            limitations=[
                "The fixed 1 July reference is a season convention, "
                "not a competition kickoff date.",
                "The current age returned by the API is deliberately not used.",
                "This context value is not used for percentile ranking.",
            ],
        ),
        *(spec.definition() for spec in API_FOOTBALL_COUNTS),
        *(spec.definition() for spec in API_FOOTBALL_AGGREGATES),
        *(spec.definition() for spec in API_FOOTBALL_METRICS),
        *(spec.definition() for spec in API_FOOTBALL_RATIOS),
    ]


class ApiFootballDatasetWriter:
    """Atomically persist canonical profile and evidence artifacts, never raw payloads."""

    def write(
        self,
        output_root: Path,
        *,
        result: ApiFootballProfileResult,
        league_id: int | list[int],
        season_start_year: int,
        competition_name: str,
        schema_version: str = "api-football-v2",
        source_endpoint: str = "/players",
        source_details: dict[str, Any] | None = None,
    ) -> list[Path]:
        if not result.profiles:
            raise ValueError("cannot write an API-Football dataset without profiles")
        if not result.evidence:
            raise ValueError("cannot write an API-Football dataset without evidence")
        output_root.mkdir(parents=True, exist_ok=True)

        profile_path = output_root / "player_season_profiles.parquet"
        evidence_path = output_root / "player_metric_evidence.parquet"
        definitions_path = output_root / "metric_definitions.json"
        manifest_path = output_root / "manifest.json"

        _atomic_write_parquet(
            profile_path,
            [_profile_record(profile) for profile in result.profiles],
        )
        _atomic_write_parquet(
            evidence_path,
            [item.model_dump(mode="json") for item in result.evidence],
        )
        _atomic_write_json(
            definitions_path,
            [item.model_dump(mode="json") for item in result.definitions],
        )
        artifact_paths = [profile_path, evidence_path, definitions_path]
        optional_parquet_artifacts: tuple[
            tuple[str, list[ScoutRAGModel]],
            ...,
        ] = (
            ("player_identities.parquet", list(result.identities)),
            ("player_team_season_stints.parquet", list(result.stints)),
            ("player_match_performances.parquet", list(result.match_performances)),
            ("player_recent_form.parquet", list(result.recent_forms)),
            ("player_season_trends.parquet", list(result.season_trends)),
        )
        for filename, records in optional_parquet_artifacts:
            if not records:
                continue
            path = output_root / filename
            _atomic_write_parquet(
                path,
                [_typed_artifact_record(item) for item in records],
            )
            artifact_paths.append(path)
        source = {
            "provider": "API-Football",
            "endpoint": source_endpoint,
            "league_id": league_id,
            "season_start_year": season_start_year,
            "competition_name": competition_name,
            "raw_responses_embedded_in_artifacts": False,
            "raw_cache_git_tracked": False,
        }
        if source_details is not None:
            protected_keys = set(source)
            conflicting_keys = protected_keys.intersection(source_details)
            if conflicting_keys:
                names = ", ".join(sorted(conflicting_keys))
                raise ValueError(f"source_details cannot replace protected keys: {names}")
            source.update(source_details)

        manifest = {
            "schema_version": schema_version,
            "generated_at": datetime.now(UTC).isoformat(),
            "source": source,
            "limitations": result.limitations,
            "artifacts": {
                path.name: {
                    "sha256": _sha256(path),
                    "bytes": path.stat().st_size,
                }
                for path in artifact_paths
            },
        }
        _atomic_write_json(manifest_path, manifest)
        return [*artifact_paths, manifest_path]


def _validate_settings(
    league_id: int,
    season_start_year: int,
    competition_name: str,
    minimum_minutes: float,
    full_sample_minutes: float,
    minimum_comparison_group_size: int,
) -> None:
    if league_id <= 0:
        raise ValueError("league_id must be positive")
    if season_start_year < 1900:
        raise ValueError("season_start_year must be a four-digit year")
    if not competition_name.strip():
        raise ValueError("competition_name must not be empty")
    if minimum_minutes < 0:
        raise ValueError("minimum_minutes must be non-negative")
    if full_sample_minutes <= 0:
        raise ValueError("full_sample_minutes must be positive")
    if minimum_comparison_group_size < 2:
        raise ValueError("minimum_comparison_group_size must be at least 2")


def _collect_players(
    payloads: list[dict[str, Any]],
    league_id: int,
    season_start_year: int,
) -> list[_PlayerAggregate]:
    grouped_blocks: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    names: defaultdict[int, list[str]] = defaultdict(list)
    player_records: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    seen_blocks: defaultdict[int, set[str]] = defaultdict(set)
    for item in payloads:
        player = item.get("player")
        statistics = item.get("statistics")
        if not isinstance(player, dict) or not isinstance(statistics, list):
            continue
        player_id = _as_int(player.get("id"))
        player_name = _display_player_name(player)
        if player_id is None or not isinstance(player_name, str) or not player_name.strip():
            continue
        names[player_id].append(player_name.strip())
        player_records[player_id].append(player)
        for block in statistics:
            if not isinstance(block, dict) or not _matches_scope(
                block,
                league_id,
                season_start_year,
            ):
                continue
            signature = json.dumps(block, ensure_ascii=False, sort_keys=True, default=str)
            if signature not in seen_blocks[player_id]:
                grouped_blocks[player_id].append(block)
                seen_blocks[player_id].add(signature)

    results: list[_PlayerAggregate] = []
    for player_id in sorted(grouped_blocks):
        blocks = grouped_blocks[player_id]
        if not blocks:
            continue
        name_counts = Counter(names[player_id])
        player_name = sorted(
            name_counts,
            key=lambda value: (-name_counts[value], value.casefold(), value),
        )[0]
        results.append(
            _PlayerAggregate(
                player_id=player_id,
                player_name=player_name,
                player=_best_player_record(player_records[player_id]),
                blocks=tuple(blocks),
            )
        )
    return results


def _matches_scope(block: dict[str, Any], league_id: int, season_start_year: int) -> bool:
    league = block.get("league")
    if not isinstance(league, dict):
        return False
    return (
        _as_int(league.get("id")) == league_id
        and _as_int(league.get("season")) == season_start_year
    )


def _display_player_name(player: dict[str, Any]) -> str | None:
    """Expand provider abbreviations such as ``J. Kimmich`` when safely possible."""
    displayed_name = player.get("name")
    if not isinstance(displayed_name, str) or not displayed_name.strip():
        return None
    clean_name = displayed_name.strip()
    abbreviated = re.fullmatch(r"[^\W\d_]\.\s+(.+)", clean_name, flags=re.UNICODE)
    if abbreviated is None:
        return clean_name

    firstname = player.get("firstname")
    if not isinstance(firstname, str) or not firstname.strip():
        return clean_name
    first_token = firstname.strip().split(maxsplit=1)[0]
    return f"{first_token} {abbreviated.group(1).strip()}"


def _prepare_profile(
    aggregate: _PlayerAggregate,
    *,
    competition_name: str,
    season_name: str,
    full_sample_minutes: float,
) -> tuple[PlayerSeasonProfile | None, float]:
    complete_minutes = _complete_sum(aggregate.blocks, ("games", "minutes"))
    if complete_minutes is None or complete_minutes <= 0:
        return None, 0
    minutes = round(complete_minutes, 3)

    team_minutes: defaultdict[str, float] = defaultdict(float)
    team_names: set[str] = set()
    for block in aggregate.blocks:
        team = block.get("team")
        if not isinstance(team, dict):
            continue
        name = team.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        clean_name = name.strip()
        team_names.add(clean_name)
        block_minutes = _number_at(block, ("games", "minutes"))
        if block_minutes is not None:
            team_minutes[clean_name] += block_minutes
    if not team_names:
        return None, 0
    ordered_teams = sorted(
        team_names,
        key=lambda name: (-team_minutes[name], name.casefold(), name),
    )

    position_minutes: defaultdict[str, float] = defaultdict(float)
    for block in aggregate.blocks:
        position = _position_group(_text_at(block, ("games", "position")))
        block_minutes = _number_at(block, ("games", "minutes")) or 0
        position_minutes[position] += block_minutes
    position_group = sorted(
        position_minutes,
        key=lambda value: (-position_minutes[value], value),
    )[0]

    features: dict[str, float] = {
        "teams_count": float(len(ordered_teams)),
    }
    for count_spec in API_FOOTBALL_COUNTS:
        if count_spec.aggregation == "sum":
            value = _complete_sum(aggregate.blocks, count_spec.source_path)
        elif count_spec.aggregation == "any":
            value = _complete_boolean_any(aggregate.blocks, count_spec.source_path)
        else:
            value = _consistent_number(aggregate.blocks, count_spec.source_path)
        if value is not None:
            features[count_spec.metric_name] = value

    for aggregate_spec in API_FOOTBALL_AGGREGATES:
        value = _complete_weighted_mean(
            aggregate.blocks,
            value_path=aggregate_spec.source_path,
            weight_path=aggregate_spec.weight_path,
        )
        if value is not None:
            features[aggregate_spec.metric_name] = value

    complete_metric_count = 0
    for metric_spec in API_FOOTBALL_METRICS:
        raw_value = _complete_sum(aggregate.blocks, metric_spec.source_path)
        if raw_value is None:
            continue
        complete_metric_count += 1
        features[metric_spec.raw_metric] = raw_value
        if minutes > 0:
            features[metric_spec.metric_name] = round((raw_value / minutes) * 90, 4)

    for ratio_spec in API_FOOTBALL_RATIOS:
        numerator = features.get(ratio_spec.numerator_metric)
        denominator = features.get(ratio_spec.denominator_metric)
        if numerator is not None and denominator is not None and denominator > 0:
            features[ratio_spec.metric_name] = round(
                (numerator / denominator) * 100,
                4,
            )

    birth_date = _date_at(aggregate.player, ("birth", "date"))
    if birth_date is not None:
        features["age_at_season_start"] = float(
            _age_on(birth_date, date(int(season_name[:4]), 7, 1))
        )

    tracked_metric_count = len(API_FOOTBALL_METRICS) + len(API_FOOTBALL_AGGREGATES)
    aggregate_completeness = sum(spec.metric_name in features for spec in API_FOOTBALL_AGGREGATES)
    metric_completeness = (complete_metric_count + aggregate_completeness) / tracked_metric_count
    provisional_quality = round(
        (0.6 * min(minutes / full_sample_minutes, 1)) + (0.4 * metric_completeness),
        3,
    )
    profile = PlayerSeasonProfile(
        player_id=f"api-football:{aggregate.player_id}",
        player_name=aggregate.player_name,
        date_of_birth=birth_date,
        birth_place=_text_at(aggregate.player, ("birth", "place")),
        birth_country=_text_at(aggregate.player, ("birth", "country")),
        nationality=_text_at(aggregate.player, ("nationality",)),
        height_cm=_measurement_at(aggregate.player, ("height",), unit="cm"),
        weight_kg=_measurement_at(aggregate.player, ("weight",), unit="kg"),
        photo_url=_text_at(aggregate.player, ("photo",)),
        team_name=ordered_teams[0],
        team_names=ordered_teams,
        competition_name=competition_name,
        season_name=season_name,
        position_group=position_group,
        minutes_played=minutes,
        structured_features=features,
        percentiles={},
        profile_text=(
            f"{aggregate.player_name} | {' / '.join(ordered_teams)} | "
            f"{competition_name} {season_name} | {position_group} | "
            f"{minutes:.1f} minutes | API-Football aggregate."
        ),
        data_quality=provisional_quality,
    )
    return profile, metric_completeness


def _percentile_specs() -> list[tuple[str, str, bool, bool]]:
    return [
        *[
            (
                spec.metric_name,
                spec.display_name,
                spec.higher_is_better,
                spec.summary_eligible,
            )
            for spec in API_FOOTBALL_METRICS
        ],
        *[
            (spec.metric_name, spec.display_name, True, spec.summary_eligible)
            for spec in API_FOOTBALL_AGGREGATES
        ],
        *[(spec.metric_name, spec.display_name, True, True) for spec in API_FOOTBALL_RATIOS],
    ]


def _best_player_record(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Pick the most complete duplicate player record deterministically."""

    def completeness(record: dict[str, Any]) -> int:
        values = (
            _text_at(record, ("birth", "date")),
            _text_at(record, ("birth", "place")),
            _text_at(record, ("birth", "country")),
            _text_at(record, ("nationality",)),
            _text_at(record, ("height",)),
            _text_at(record, ("weight",)),
            _text_at(record, ("photo",)),
        )
        return sum(value is not None for value in values)

    return sorted(
        records,
        key=lambda record: (
            -completeness(record),
            json.dumps(record, ensure_ascii=False, sort_keys=True, default=str),
        ),
    )[0]


def _complete_boolean_any(
    blocks: tuple[dict[str, Any], ...],
    path: tuple[str, ...],
) -> float | None:
    values = [_value_at(block, path) for block in blocks]
    if not values or any(not isinstance(value, bool) for value in values):
        return None
    return float(any(value is True for value in values))


def _consistent_number(
    blocks: tuple[dict[str, Any], ...],
    path: tuple[str, ...],
) -> float | None:
    values = [_number_at(block, path) for block in blocks]
    reported = [value for value in values if value is not None]
    if not reported or len(set(reported)) != 1:
        return None
    return reported[0]


def _complete_weighted_mean(
    blocks: tuple[dict[str, Any], ...],
    *,
    value_path: tuple[str, ...],
    weight_path: tuple[str, ...],
) -> float | None:
    weighted_values: list[tuple[float, float]] = []
    for block in blocks:
        weight = _number_at(block, weight_path)
        if weight is None:
            return None
        if weight == 0:
            continue
        value = _number_at(block, value_path)
        if value is None:
            return None
        weighted_values.append((value, weight))
    total_weight = sum(weight for _, weight in weighted_values)
    if total_weight <= 0:
        return None
    return round(
        sum(value * weight for value, weight in weighted_values) / total_weight,
        4,
    )


def _complete_sum(blocks: tuple[dict[str, Any], ...], path: tuple[str, ...]) -> float | None:
    values = [_number_at(block, path) for block in blocks]
    if not values or any(value is None for value in values):
        return None
    return round(sum(value for value in values if value is not None), 4)


def _number_at(record: dict[str, Any], path: tuple[str, ...]) -> float | None:
    value = _value_at(record, path)
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        number = float(value)
    elif isinstance(value, str):
        try:
            number = float(value.strip())
        except ValueError:
            return None
    else:
        return None
    return number if math.isfinite(number) and number >= 0 else None


def _text_at(record: dict[str, Any], path: tuple[str, ...]) -> str | None:
    value = _value_at(record, path)
    return value.strip() if isinstance(value, str) and value.strip() else None


def _value_at(record: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = record
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _date_at(record: dict[str, Any], path: tuple[str, ...]) -> date | None:
    value = _text_at(record, path)
    if value is None:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _measurement_at(
    record: dict[str, Any],
    path: tuple[str, ...],
    *,
    unit: str,
) -> float | None:
    value = _value_at(record, path)
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        number = float(value)
    elif isinstance(value, str):
        matched = re.fullmatch(
            rf"\s*(\d+(?:[.,]\d+)?)\s*(?:{re.escape(unit)})?\s*",
            value,
            flags=re.IGNORECASE,
        )
        if matched is None:
            return None
        number = float(matched.group(1).replace(",", "."))
    else:
        return None
    return number if math.isfinite(number) and number > 0 else None


def _age_on(birth_date: date, reference_date: date) -> int:
    return (
        reference_date.year
        - birth_date.year
        - ((reference_date.month, reference_date.day) < (birth_date.month, birth_date.day))
    )


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _position_group(position: str | None) -> str:
    if position is None:
        return "unknown"
    normalized = position.casefold()
    if "goalkeeper" in normalized:
        return "goalkeeper"
    if "defender" in normalized:
        return "defender"
    if "midfield" in normalized:
        return "midfielder"
    if "attacker" in normalized or "forward" in normalized:
        return "forward"
    return normalized.replace(" ", "_")


def _percentile(
    value: float,
    values: list[float],
    *,
    higher_is_better: bool,
) -> float:
    if len(values) == 1:
        return 50
    lower = sum(item < value for item in values)
    equal_others = sum(item == value for item in values) - 1
    raw = ((lower + (0.5 * equal_others)) / (len(values) - 1)) * 100
    return round(raw if higher_is_better else 100 - raw, 1)


def _profile_record(profile: PlayerSeasonProfile) -> dict[str, Any]:
    record = profile.model_dump(mode="json")
    record["structured_features_json"] = json.dumps(
        record.pop("structured_features"),
        ensure_ascii=False,
        sort_keys=True,
    )
    record["percentiles_json"] = json.dumps(
        record.pop("percentiles"),
        ensure_ascii=False,
        sort_keys=True,
    )
    return record


def _typed_artifact_record(item: ScoutRAGModel) -> dict[str, Any]:
    """Store dynamic mappings as JSON instead of sparse Arrow structs.

    Feature dictionaries intentionally have different keys for different player
    positions and matches. Arrow structs turn absent keys into null-valued keys
    when such rows are read again, which is not the same as an unavailable
    metric in the typed domain model.
    """

    record = item.model_dump(mode="json")
    for field_name, value in tuple(record.items()):
        if not isinstance(value, dict):
            continue
        record.pop(field_name)
        record[f"{field_name}_json"] = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
        )
    return record


def _atomic_write_parquet(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        raise ValueError(f"cannot write empty Parquet dataset: {path.name}")
    table = pa.Table.from_pylist(records)
    metadata = dict(table.schema.metadata or {})
    metadata[b"data_source"] = b"API-Football"
    metadata[b"raw_responses_embedded_in_artifacts"] = b"false"
    table = table.replace_schema_metadata(metadata)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        pq.write_table(
            table,
            temporary_path,
            compression="zstd",
            row_group_size=10_000,
        )
        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
            mode="w",
            encoding="utf-8",
        ) as temporary:
            temporary.write(json.dumps(value, ensure_ascii=False, indent=2))
            temporary.write("\n")
            temporary_path = Path(temporary.name)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
