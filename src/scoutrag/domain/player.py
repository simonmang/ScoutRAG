"""Typed football retrieval and evidence units."""

from datetime import date

from pydantic import Field, field_validator

from scoutrag.domain.base import ScoutRAGModel


class PlayerSeasonProfile(ScoutRAGModel):
    """A player's immutable profile for exactly one season and competition."""

    player_id: str = Field(min_length=1)
    player_name: str = Field(min_length=1)
    team_name: str = Field(min_length=1)
    competition_name: str = Field(min_length=1)
    season_name: str = Field(min_length=1)
    position_group: str = Field(min_length=1)
    minutes_played: float = Field(ge=0)
    structured_features: dict[str, float] = Field(default_factory=dict)
    percentiles: dict[str, float] = Field(default_factory=dict)
    profile_text: str = Field(min_length=1)
    data_quality: float = Field(ge=0, le=1)

    @field_validator("percentiles")
    @classmethod
    def validate_percentiles(cls, values: dict[str, float]) -> dict[str, float]:
        """Percentiles use the conventional inclusive 0..100 scale."""
        invalid = {name: value for name, value in values.items() if not 0 <= value <= 100}
        if invalid:
            raise ValueError(f"percentiles must be between 0 and 100: {invalid}")
        return values


class PlayerMetricEvidence(ScoutRAGModel):
    """A source-linked statistical observation for one player season."""

    player_id: str = Field(min_length=1)
    season_id: str = Field(min_length=1)
    metric_name: str = Field(min_length=1)
    raw_value: float | None = None
    normalized_value: float | None = None
    percentile: float | None = Field(default=None, ge=0, le=100)
    comparison_group: str = Field(min_length=1)
    sample_size: float | None = Field(default=None, ge=0)
    source_reference: str = Field(min_length=1)


class MatchEvidence(ScoutRAGModel):
    """Optional post-MVP evidence from a concrete match or event group."""

    player_id: str = Field(min_length=1)
    match_id: str = Field(min_length=1)
    competition_name: str = Field(min_length=1)
    match_date: date | None = None
    evidence_type: str = Field(min_length=1)
    description: str = Field(min_length=1)
    supporting_event_ids: list[str] = Field(default_factory=list)


class MetricDefinition(ScoutRAGModel):
    """Human-readable definition and provenance contract for a metric."""

    metric_name: str = Field(min_length=1)
    display_name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    calculation_method: str = Field(min_length=1)
    required_event_types: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
