"""Typed football retrieval and evidence units."""

from datetime import date
from enum import StrEnum

from pydantic import Field, field_validator, model_validator

from scoutrag.domain.base import ScoutRAGModel


class PlayerSeasonProfile(ScoutRAGModel):
    """A player's immutable profile for exactly one season and competition."""

    player_id: str = Field(min_length=1)
    profile_id: str | None = Field(default=None, min_length=1)
    player_name: str = Field(min_length=1)
    date_of_birth: date | None = None
    birth_place: str | None = Field(default=None, min_length=1, max_length=200)
    birth_country: str | None = Field(default=None, min_length=1, max_length=100)
    nationality: str | None = Field(default=None, min_length=1, max_length=100)
    height_cm: float | None = Field(default=None, ge=100, le=250)
    weight_kg: float | None = Field(default=None, ge=30, le=250)
    photo_url: str | None = Field(
        default=None,
        min_length=1,
        max_length=2048,
        pattern=r"^https?://\S+$",
    )
    team_name: str = Field(min_length=1)
    team_names: list[str] = Field(default_factory=list)
    competition_name: str = Field(min_length=1)
    season_name: str = Field(min_length=1)
    position_group: str = Field(min_length=1)
    minutes_played: float = Field(ge=0)
    structured_features: dict[str, float] = Field(default_factory=dict)
    percentiles: dict[str, float] = Field(default_factory=dict)
    profile_text: str = Field(min_length=1)
    data_quality: float = Field(ge=0, le=1)

    @field_validator(
        "birth_place",
        "birth_country",
        "nationality",
        "photo_url",
        mode="before",
    )
    @classmethod
    def empty_optional_text_is_missing(cls, value: object) -> object:
        """Normalize blank provider metadata without weakening non-empty validation."""
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("date_of_birth")
    @classmethod
    def validate_date_of_birth(cls, value: date | None) -> date | None:
        """Birth dates are historical facts and therefore cannot be in the future."""
        if value is not None and value > date.today():
            raise ValueError("date_of_birth must not be in the future")
        return value

    @field_validator("percentiles")
    @classmethod
    def validate_percentiles(cls, values: dict[str, float]) -> dict[str, float]:
        """Percentiles use the conventional inclusive 0..100 scale."""
        invalid = {name: value for name, value in values.items() if not 0 <= value <= 100}
        if invalid:
            raise ValueError(f"percentiles must be between 0 and 100: {invalid}")
        return values


class PlayerIdentity(ScoutRAGModel):
    """Stable biographical identity shared by every club and season record."""

    player_id: str = Field(min_length=1)
    player_name: str = Field(min_length=1)
    date_of_birth: date | None = None
    birth_place: str | None = Field(default=None, min_length=1, max_length=200)
    birth_country: str | None = Field(default=None, min_length=1, max_length=100)
    nationality: str | None = Field(default=None, min_length=1, max_length=100)
    height_cm: float | None = Field(default=None, ge=100, le=250)
    weight_kg: float | None = Field(default=None, ge=30, le=250)
    photo_url: str | None = Field(
        default=None,
        min_length=1,
        max_length=2048,
        pattern=r"^https?://\S+$",
    )
    source_reference: str = Field(min_length=1)

    @field_validator(
        "birth_place",
        "birth_country",
        "nationality",
        "photo_url",
        mode="before",
    )
    @classmethod
    def empty_optional_text_is_missing(cls, value: object) -> object:
        """Normalize blank provider metadata."""

        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("date_of_birth")
    @classmethod
    def validate_date_of_birth(cls, value: date | None) -> date | None:
        if value is not None and value > date.today():
            raise ValueError("date_of_birth must not be in the future")
        return value


class PlayerTeamSeasonStint(ScoutRAGModel):
    """One player's evidence for one club inside one competition-season."""

    stint_id: str = Field(min_length=1)
    player_id: str = Field(min_length=1)
    profile_id: str = Field(min_length=1)
    season_id: str = Field(min_length=1)
    season_name: str = Field(min_length=1)
    competition_name: str = Field(min_length=1)
    team_id: int | None = Field(default=None, ge=1)
    team_name: str = Field(min_length=1)
    position_group: str = Field(min_length=1)
    minutes_played: float = Field(ge=0)
    appearances: int = Field(ge=1)
    structured_features: dict[str, float] = Field(default_factory=dict)
    data_quality: float = Field(ge=0, le=1)
    source_reference: str = Field(min_length=1)


class PlayerMatchPerformance(ScoutRAGModel):
    """Typed player performance in one fixture for rolling-form analysis."""

    performance_id: str = Field(min_length=1)
    player_id: str = Field(min_length=1)
    profile_id: str = Field(min_length=1)
    season_id: str = Field(min_length=1)
    season_name: str = Field(min_length=1)
    competition_name: str = Field(min_length=1)
    fixture_id: int = Field(ge=1)
    match_date: date | None = None
    team_id: int | None = Field(default=None, ge=1)
    team_name: str = Field(min_length=1)
    opponent_id: int | None = Field(default=None, ge=1)
    opponent_name: str | None = Field(default=None, min_length=1)
    home_away: str | None = Field(default=None, pattern=r"^(home|away)$")
    position_group: str = Field(min_length=1)
    minutes_played: float = Field(gt=0)
    started: bool | None = None
    substitute: bool | None = None
    captain: bool | None = None
    structured_features: dict[str, float] = Field(default_factory=dict)
    data_quality: float = Field(ge=0, le=1)
    source_reference: str = Field(min_length=1)


class TrendDirection(StrEnum):
    """Descriptive direction of stored observations, never a prediction."""

    IMPROVING = "improving"
    STABLE = "stable"
    DECLINING = "declining"
    MIXED = "mixed"
    INSUFFICIENT = "insufficient"


class PlayerRecentForm(ScoutRAGModel):
    """Recent fixture window compared with the same season's prior baseline."""

    profile_id: str = Field(min_length=1)
    player_id: str = Field(min_length=1)
    as_of_date: date | None = None
    window_size: int = Field(ge=1)
    matches_in_window: int = Field(ge=1)
    minutes_in_window: float = Field(gt=0)
    fixture_ids: list[int] = Field(min_length=1)
    recent_features: dict[str, float] = Field(default_factory=dict)
    baseline_features: dict[str, float] = Field(default_factory=dict)
    relative_changes: dict[str, float] = Field(default_factory=dict)
    data_quality: float = Field(ge=0, le=1)
    limitations: list[str] = Field(default_factory=list)


class SeasonMetricObservation(ScoutRAGModel):
    """One league-season observation used in a transparent trend."""

    profile_id: str = Field(min_length=1)
    competition_name: str = Field(min_length=1)
    season_name: str = Field(min_length=1)
    team_names: list[str] = Field(min_length=1)
    minutes_played: float = Field(ge=0)
    normalized_value: float
    percentile: float | None = Field(default=None, ge=0, le=100)
    data_quality: float = Field(ge=0, le=1)


class PlayerSeasonTrend(ScoutRAGModel):
    """Current metric plus separate historical observations and direction."""

    trend_id: str = Field(min_length=1)
    player_id: str = Field(min_length=1)
    current_profile_id: str = Field(min_length=1)
    metric_name: str = Field(min_length=1)
    direction: TrendDirection
    latest_value: float
    previous_value: float | None = None
    relative_change: float | None = None
    historical_fallback_available: bool
    observations: list[SeasonMetricObservation] = Field(min_length=1)
    limitations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_fallback(self) -> "PlayerSeasonTrend":
        if self.historical_fallback_available != (len(self.observations) > 1):
            raise ValueError("historical_fallback_available must reflect multiple observations")
        return self


class PlayerTemporalContext(ScoutRAGModel):
    """Current-first player history exposed to evidence consumers."""

    player_id: str = Field(min_length=1)
    identity: PlayerIdentity | None = None
    season_profiles: list[PlayerSeasonProfile] = Field(default_factory=list)
    team_stints: list[PlayerTeamSeasonStint] = Field(default_factory=list)
    recent_forms: list[PlayerRecentForm] = Field(default_factory=list)
    season_trends: list[PlayerSeasonTrend] = Field(default_factory=list)
    latest_matches: list[PlayerMatchPerformance] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_player_identity(self) -> "PlayerTemporalContext":
        related_ids = {
            item.player_id
            for collection in (
                self.season_profiles,
                self.team_stints,
                self.recent_forms,
                self.season_trends,
                self.latest_matches,
            )
            for item in collection
        }
        if self.identity is not None:
            related_ids.add(self.identity.player_id)
        if related_ids - {self.player_id}:
            raise ValueError("all temporal context records must belong to player_id")
        return self


class PlayerMetricEvidence(ScoutRAGModel):
    """A source-linked statistical observation for one player season."""

    player_id: str = Field(min_length=1)
    profile_id: str | None = Field(default=None, min_length=1)
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


def profile_evidence_key(profile: PlayerSeasonProfile) -> str:
    """Return a competition-season-safe key while supporting older artifacts."""

    return profile.profile_id or profile.player_id
