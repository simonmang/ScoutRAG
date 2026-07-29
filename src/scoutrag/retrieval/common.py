"""Shared deterministic filtering and text projection for retrieval."""

import json
import re
from pathlib import Path

import pyarrow.parquet as pq  # type: ignore[import-untyped]

from scoutrag.domain.player import PlayerSeasonProfile
from scoutrag.domain.query import QueryIntent, QueryProfile

TOKEN_PATTERN = re.compile(r"[\wäöüÄÖÜß]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Produce stable Unicode-aware lowercase tokens."""
    return [token.casefold() for token in TOKEN_PATTERN.findall(text)]


def profile_key(profile: PlayerSeasonProfile) -> tuple[str, str, str]:
    """Identify one player-season without collapsing seasons during fusion."""
    return profile.player_id, profile.competition_name, profile.season_name


def matches_hard_filters(profile: PlayerSeasonProfile, query: QueryProfile) -> bool:
    """Apply only explicit user constraints shared by all recall strategies."""
    if query.intent in {
        QueryIntent.EXACT_PLAYER_LOOKUP,
        QueryIntent.PLAYER_COMPARISON,
    } and profile.player_name.casefold() not in {name.casefold() for name in query.named_players}:
        return False
    if query.requested_positions and profile.position_group not in query.requested_positions:
        return False
    if query.minimum_minutes is not None and profile.minutes_played < query.minimum_minutes:
        return False
    if query.team_filters and not any(
        _matches_any(team_name, query.team_filters) for team_name in profile.team_names
    ):
        return False
    if query.competition_filters and not _matches_any(
        profile.competition_name,
        query.competition_filters,
    ):
        return False
    return not (
        query.season_filters and not _matches_any(profile.season_name, query.season_filters)
    )


def query_search_text(query: QueryProfile) -> str:
    """Augment natural language with canonical terms discovered by query analysis."""
    fields = [
        query.normalized_query,
        *query.requested_positions,
        *query.requested_traits,
        *query.requested_metrics,
        *query.named_players,
        *query.team_filters,
        *query.competition_filters,
        *query.season_filters,
    ]
    return " ".join(field.replace("_", " ") for field in fields)


def profile_search_text(profile: PlayerSeasonProfile) -> str:
    """Project a typed profile into one retrieval-only lexical document."""
    metric_names = " ".join(name.replace("_", " ") for name in profile.percentiles)
    return " | ".join(
        [
            profile.player_name,
            " ".join(profile.team_names),
            profile.competition_name,
            profile.season_name,
            profile.position_group.replace("_", " "),
            metric_names,
            profile.profile_text,
        ]
    )


def load_profiles(path: Path) -> list[PlayerSeasonProfile]:
    """Load Phase 3 profiles while restoring deterministic JSON map columns."""
    rows = pq.read_table(path).to_pylist()
    profiles: list[PlayerSeasonProfile] = []
    for row in rows:
        record = dict(row)
        record["structured_features"] = json.loads(record.pop("structured_features_json"))
        record["percentiles"] = json.loads(record.pop("percentiles_json"))
        profiles.append(PlayerSeasonProfile.model_validate(record))
    return profiles


def _matches_any(value: str, filters: list[str]) -> bool:
    normalized = value.casefold()
    return any(item.casefold() in normalized or normalized in item.casefold() for item in filters)
