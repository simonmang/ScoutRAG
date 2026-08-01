"""Read-only access to separate current and historical player evidence."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, TypeVar

import pyarrow.parquet as pq  # type: ignore[import-untyped]

from scoutrag.domain.base import ScoutRAGModel
from scoutrag.domain.player import (
    PlayerIdentity,
    PlayerMatchPerformance,
    PlayerRecentForm,
    PlayerSeasonProfile,
    PlayerSeasonTrend,
    PlayerTeamSeasonStint,
    PlayerTemporalContext,
)

ModelT = TypeVar("ModelT", bound=ScoutRAGModel)


class PlayerHistoryStore:
    """Query a local multi-season artifact by stable API-Football player ID."""

    def __init__(self, root: Path) -> None:
        self.root = root
        required = (
            "player_season_profiles.parquet",
            "player_identities.parquet",
            "player_team_season_stints.parquet",
            "player_recent_form.parquet",
            "player_season_trends.parquet",
            "player_match_performances.parquet",
        )
        missing = [name for name in required if not (root / name).exists()]
        if missing:
            raise FileNotFoundError(
                f"ScoutRAG history artifacts are missing in {root}: {', '.join(missing)}"
            )

    def for_player(self, player_id: str, *, match_limit: int = 10) -> PlayerTemporalContext:
        """Return newest-first season data; match detail is deliberately bounded."""

        if not player_id.strip():
            raise ValueError("player_id must not be blank")
        return self.for_players([player_id], match_limit=match_limit)[player_id]

    def for_players(
        self,
        player_ids: list[str],
        *,
        match_limit: int = 10,
    ) -> dict[str, PlayerTemporalContext]:
        """Load a candidate set with one filtered scan per artifact."""

        if not 0 <= match_limit <= 50:
            raise ValueError("match_limit must be between zero and 50")
        unique_ids = list(dict.fromkeys(player_id.strip() for player_id in player_ids))
        if not unique_ids or any(not player_id for player_id in unique_ids):
            raise ValueError("player_ids must contain non-blank values")

        identities = _load_models_many(
            self.root / "player_identities.parquet",
            PlayerIdentity,
            unique_ids,
        )
        profiles = _load_models_many(
            self.root / "player_season_profiles.parquet",
            PlayerSeasonProfile,
            unique_ids,
        )
        stints = _load_models_many(
            self.root / "player_team_season_stints.parquet",
            PlayerTeamSeasonStint,
            unique_ids,
        )
        forms = _load_models_many(
            self.root / "player_recent_form.parquet",
            PlayerRecentForm,
            unique_ids,
        )
        trends = _load_models_many(
            self.root / "player_season_trends.parquet",
            PlayerSeasonTrend,
            unique_ids,
        )
        matches = _load_models_many(
            self.root / "player_match_performances.parquet",
            PlayerMatchPerformance,
            unique_ids,
        )
        return {
            player_id: _context(
                player_id,
                identities.get(player_id, []),
                profiles.get(player_id, []),
                stints.get(player_id, []),
                forms.get(player_id, []),
                trends.get(player_id, []),
                matches.get(player_id, []),
                match_limit=match_limit,
            )
            for player_id in unique_ids
        }


def _read_rows(path: Path, player_ids: list[str]) -> list[dict[str, Any]]:
    table = pq.read_table(path, filters=[("player_id", "in", player_ids)])
    rows: list[dict[str, Any]] = []
    for stored in table.to_pylist():
        row = dict(stored)
        for field_name, value in tuple(row.items()):
            if field_name.endswith("_json") and isinstance(value, str):
                row[field_name.removesuffix("_json")] = json.loads(value)
                row.pop(field_name)
            elif isinstance(value, dict):
                row[field_name] = {
                    key: nested for key, nested in value.items() if nested is not None
                }
        rows.append(row)
    return rows


def _load_models_many(
    path: Path,
    model: type[ModelT],
    player_ids: list[str],
) -> dict[str, list[ModelT]]:
    grouped: defaultdict[str, list[ModelT]] = defaultdict(list)
    for row in _read_rows(path, player_ids):
        item = model.model_validate(row)
        grouped[str(row["player_id"])].append(item)
    return dict(grouped)


def _context(
    player_id: str,
    identities: list[PlayerIdentity],
    profiles: list[PlayerSeasonProfile],
    stints: list[PlayerTeamSeasonStint],
    forms: list[PlayerRecentForm],
    trends: list[PlayerSeasonTrend],
    matches: list[PlayerMatchPerformance],
    *,
    match_limit: int,
) -> PlayerTemporalContext:
    matches.sort(key=lambda item: (item.match_date is not None, item.match_date, item.fixture_id))
    return PlayerTemporalContext(
        player_id=player_id,
        identity=identities[0] if identities else None,
        season_profiles=sorted(
            profiles,
            key=lambda item: (int(item.season_name[:4]), item.minutes_played),
            reverse=True,
        ),
        team_stints=sorted(
            stints,
            key=lambda item: (int(item.season_name[:4]), item.minutes_played),
            reverse=True,
        ),
        recent_forms=sorted(
            forms,
            key=lambda item: (item.as_of_date is not None, item.as_of_date),
            reverse=True,
        ),
        season_trends=sorted(
            trends,
            key=lambda item: (item.current_profile_id, item.metric_name),
        ),
        latest_matches=list(reversed(matches[-match_limit:])) if match_limit else [],
    )
