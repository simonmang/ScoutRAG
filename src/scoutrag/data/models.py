"""Typed intermediate records used by the data and feature pipeline."""

from datetime import date

from pydantic import Field

from scoutrag.domain.base import ScoutRAGModel


class CompetitionSeason(ScoutRAGModel):
    """One exact StatsBomb competition-season source."""

    competition_id: int = Field(gt=0)
    season_id: int = Field(gt=0)
    country_name: str = Field(min_length=1)
    competition_name: str = Field(min_length=1)
    season_name: str = Field(min_length=1)
    competition_gender: str | None = None
    source_reference: str = Field(min_length=1)


class MatchRecord(ScoutRAGModel):
    """Normalized metadata and observed duration for one match."""

    match_id: int = Field(gt=0)
    competition_id: int = Field(gt=0)
    season_id: int = Field(gt=0)
    match_date: date
    match_week: int | None = Field(default=None, ge=1)
    home_team_id: int = Field(gt=0)
    home_team_name: str = Field(min_length=1)
    away_team_id: int = Field(gt=0)
    away_team_name: str = Field(min_length=1)
    home_score: int = Field(ge=0)
    away_score: int = Field(ge=0)
    duration_seconds: float = Field(gt=0)
    source_reference: str = Field(min_length=1)


class NormalizedEvent(ScoutRAGModel):
    """Flat, source-linked representation of a StatsBomb event."""

    event_id: str = Field(min_length=1)
    match_id: int = Field(gt=0)
    competition_id: int = Field(gt=0)
    season_id: int = Field(gt=0)
    event_index: int = Field(ge=1)
    period: int = Field(ge=1)
    timestamp: str = Field(min_length=1)
    match_second: float = Field(ge=0)
    event_type: str = Field(min_length=1)
    event_subtype: str | None = None
    outcome_name: str | None = None
    team_id: int | None = Field(default=None, gt=0)
    team_name: str | None = None
    player_id: int | None = Field(default=None, gt=0)
    player_name: str | None = None
    position_name: str | None = None
    location_x: float | None = None
    location_y: float | None = None
    end_location_x: float | None = None
    end_location_y: float | None = None
    expected_goals: float | None = Field(default=None, ge=0)
    pass_length: float | None = Field(default=None, ge=0)
    duration_seconds: float | None = Field(default=None, ge=0)
    under_pressure: bool = False
    counterpress: bool = False
    source_reference: str = Field(min_length=1)


class PlayerMatchParticipation(ScoutRAGModel):
    """Minutes and dominant position for a player in one match."""

    match_id: int = Field(gt=0)
    competition_id: int = Field(gt=0)
    season_id: int = Field(gt=0)
    player_id: int = Field(gt=0)
    player_name: str = Field(min_length=1)
    team_id: int = Field(gt=0)
    team_name: str = Field(min_length=1)
    primary_position: str = Field(min_length=1)
    position_group: str = Field(min_length=1)
    minutes_played: float = Field(gt=0)
    started: bool
    source_reference: str = Field(min_length=1)


class DataValidationReport(ScoutRAGModel):
    """Machine-readable validation result stored beside generated data."""

    valid: bool
    competition_id: int = Field(gt=0)
    season_id: int = Field(gt=0)
    match_count: int = Field(ge=0)
    event_count: int = Field(ge=0)
    participation_count: int = Field(ge=0)
    profile_count: int = Field(ge=0)
    evidence_count: int = Field(ge=0)
    metric_definition_count: int = Field(default=0, ge=0)
    percentile_profile_count: int = Field(default=0, ge=0)
    feature_version: str = "phase3-v1"
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class DownloadSummary(ScoutRAGModel):
    """Result of a reproducible StatsBomb download."""

    competition_id: int = Field(gt=0)
    season_id: int = Field(gt=0)
    match_ids: list[int] = Field(default_factory=list)
    files_downloaded: int = Field(ge=0)
    output_directory: str = Field(min_length=1)


class PipelineResult(ScoutRAGModel):
    """Serializable summary returned by the complete data build."""

    competition: CompetitionSeason
    validation: DataValidationReport
    output_files: list[str] = Field(default_factory=list)
