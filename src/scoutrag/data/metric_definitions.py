"""Central definitions for every engineered scouting feature."""

from dataclasses import dataclass

from scoutrag.domain.player import MetricDefinition


@dataclass(frozen=True, slots=True)
class FeatureMetricSpec:
    """Machine-readable calculation contract for one normalized feature."""

    metric_name: str
    raw_metric: str
    display_name: str
    description: str
    calculation_method: str
    required_event_types: tuple[str, ...]
    limitations: tuple[str, ...]
    calculation: str = "per_90"

    def as_definition(self) -> MetricDefinition:
        return MetricDefinition(
            metric_name=self.metric_name,
            display_name=self.display_name,
            description=self.description,
            calculation_method=self.calculation_method,
            required_event_types=list(self.required_event_types),
            limitations=list(self.limitations),
        )


COMMON_LIMITATION = (
    "StatsBomb Open Data coverage may contain only one reference team's matches.",
    "The metric describes observed event volume, not causal player quality.",
)

FEATURE_METRICS = (
    FeatureMetricSpec(
        "passes_per_90",
        "passes",
        "Passes per 90",
        "Attempted passes per 90 observed minutes.",
        "passes / minutes_played * 90",
        ("Pass",),
        COMMON_LIMITATION,
    ),
    FeatureMetricSpec(
        "pass_completion_rate",
        "passes_completed",
        "Pass completion rate",
        "Share of attempted passes recorded without an incomplete outcome.",
        "passes_completed / passes * 100",
        ("Pass",),
        (
            *COMMON_LIMITATION,
            "Pass difficulty, location, pressure, and tactical intent are not adjusted.",
        ),
        calculation="rate",
    ),
    FeatureMetricSpec(
        "progressive_passes_per_90",
        "progressive_passes",
        "Progressive passes per 90",
        "Completed passes that move the ball at least ten coordinate units closer to goal.",
        "progressive_passes / minutes_played * 90",
        ("Pass",),
        (
            *COMMON_LIMITATION,
            "The geometric threshold is a transparent project definition, not a vendor metric.",
        ),
    ),
    FeatureMetricSpec(
        "carries_per_90",
        "carries",
        "Carries per 90",
        "Recorded ball carries per 90 observed minutes.",
        "carries / minutes_played * 90",
        ("Carry",),
        COMMON_LIMITATION,
    ),
    FeatureMetricSpec(
        "progressive_carries_per_90",
        "progressive_carries",
        "Progressive carries per 90",
        "Carries that move the ball at least ten coordinate units closer to goal.",
        "progressive_carries / minutes_played * 90",
        ("Carry",),
        (
            *COMMON_LIMITATION,
            "The geometric threshold is a transparent project definition, not a vendor metric.",
        ),
    ),
    FeatureMetricSpec(
        "pressures_per_90",
        "pressures",
        "Pressures per 90",
        "Recorded pressure events per 90 observed minutes.",
        "pressures / minutes_played * 90",
        ("Pressure",),
        (*COMMON_LIMITATION, "Pressure event collection depends on provider definitions."),
    ),
    FeatureMetricSpec(
        "ball_recoveries_per_90",
        "ball_recoveries",
        "Ball recoveries per 90",
        "Recorded ball recoveries per 90 observed minutes.",
        "ball_recoveries / minutes_played * 90",
        ("Ball Recovery",),
        COMMON_LIMITATION,
    ),
    FeatureMetricSpec(
        "interceptions_per_90",
        "interceptions",
        "Interceptions per 90",
        "Recorded interceptions per 90 observed minutes.",
        "interceptions / minutes_played * 90",
        ("Interception",),
        COMMON_LIMITATION,
    ),
    FeatureMetricSpec(
        "tackles_per_90",
        "tackles",
        "Tackles per 90",
        "Duel events typed as tackles per 90 observed minutes.",
        "tackles / minutes_played * 90",
        ("Duel",),
        (*COMMON_LIMITATION, "Tackle outcome and defensive opportunity are not adjusted."),
    ),
    FeatureMetricSpec(
        "shots_per_90",
        "shots",
        "Shots per 90",
        "Recorded shots per 90 observed minutes.",
        "shots / minutes_played * 90",
        ("Shot",),
        COMMON_LIMITATION,
    ),
    FeatureMetricSpec(
        "expected_goals_per_90",
        "expected_goals",
        "Expected goals per 90",
        "Sum of StatsBomb shot expected-goal values per 90 observed minutes.",
        "sum(statsbomb_xg) / minutes_played * 90",
        ("Shot",),
        (*COMMON_LIMITATION, "Expected goals is model-dependent and not a finishing guarantee."),
    ),
    FeatureMetricSpec(
        "dribbles_completed_per_90",
        "dribbles_completed",
        "Completed dribbles per 90",
        "Dribble events with a complete outcome per 90 observed minutes.",
        "dribbles_completed / minutes_played * 90",
        ("Dribble",),
        COMMON_LIMITATION,
    ),
    FeatureMetricSpec(
        "clearances_per_90",
        "clearances",
        "Clearances per 90",
        "Recorded clearances per 90 observed minutes.",
        "clearances / minutes_played * 90",
        ("Clearance",),
        (
            *COMMON_LIMITATION,
            "A higher value can reflect team context rather than stronger performance.",
        ),
    ),
)


def metric_definitions() -> list[MetricDefinition]:
    """Return stable, serializable definitions in feature order."""
    return [spec.as_definition() for spec in FEATURE_METRICS]
