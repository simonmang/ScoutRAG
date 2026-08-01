"""Models produced by rule-based query analysis."""

from enum import StrEnum

from pydantic import Field, model_validator

from scoutrag.domain.base import ScoutRAGModel


class QueryIntent(StrEnum):
    """Supported user intentions."""

    PLAYER_DISCOVERY = "player_discovery"
    SIMILAR_PLAYER = "similar_player"
    PLAYER_COMPARISON = "player_comparison"
    EXACT_PLAYER_LOOKUP = "exact_player_lookup"
    AGGREGATION = "aggregation"
    OUT_OF_SCOPE = "out_of_scope"


class TemporalScope(StrEnum):
    """Requested time perspective; current season is always the default."""

    CURRENT = "current"
    RECENT_FORM = "recent_form"
    HISTORY = "history"
    TREND = "trend"


class QueryProfile(ScoutRAGModel):
    """Deterministic, retrieval-oriented representation of a user query."""

    original_query: str = Field(min_length=1)
    normalized_query: str = Field(min_length=1)
    intent: QueryIntent
    requested_positions: list[str] = Field(default_factory=list)
    requested_traits: list[str] = Field(default_factory=list)
    requested_metrics: list[str] = Field(default_factory=list)
    named_players: list[str] = Field(default_factory=list)
    team_filters: list[str] = Field(default_factory=list)
    competition_filters: list[str] = Field(default_factory=list)
    season_filters: list[str] = Field(default_factory=list)
    minimum_minutes: int | None = Field(default=None, ge=0)
    result_count: int = Field(default=10, ge=1, le=100)
    expected_evidence_types: list[str] = Field(default_factory=list)
    temporal_scope: TemporalScope = TemporalScope.CURRENT

    @model_validator(mode="after")
    def validate_intent_requirements(self) -> "QueryProfile":
        """Reject profiles that cannot represent their declared lookup intent."""
        if self.intent is QueryIntent.EXACT_PLAYER_LOOKUP and not self.named_players:
            raise ValueError("exact_player_lookup requires at least one named player")
        if self.intent is QueryIntent.PLAYER_COMPARISON and len(self.named_players) < 2:
            raise ValueError("player_comparison requires at least two named players")
        if self.intent is QueryIntent.SIMILAR_PLAYER and not self.named_players:
            raise ValueError("similar_player requires a reference player")
        return self
