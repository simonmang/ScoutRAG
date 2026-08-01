"""Quota-aware, resumable API-Football fixture synchronization.

The season aggregate returned by ``/players`` is useful for player identity
metadata, but its team/statistics blocks can mix transfers and appearances.
ScoutRAG therefore treats completed fixture packages as the authoritative
input for season aggregation.  This module only downloads and validates those
packages; converting them into domain profiles is a separate responsibility.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from scoutrag.data.api_football import (
    ApiFootballClient,
    ApiFootballPageLimitExceeded,
    ApiFootballProtocolError,
    ApiFootballQuota,
    ApiFootballResponse,
    QueryValue,
)

COMPLETED_FIXTURE_STATUSES = "FT-AET-PEN"
MAX_FIXTURE_IDS_PER_REQUEST = 20


@dataclass(frozen=True, slots=True)
class ApiFootballFixtureSyncResult:
    """Validated fixture packages and download provenance."""

    league_id: int
    season_start_year: int
    fixtures: list[dict[str, Any]]
    fixture_ids: list[int]
    player_identities: list[dict[str, Any]]
    fixture_list_pages_fetched: int
    detail_batches_fetched: int
    player_pages_fetched: int
    network_requests_made: int
    quota: ApiFootballQuota | None

    @property
    def fixture_count(self) -> int:
        """Number of unique completed fixture packages."""

        return len(self.fixtures)

    @property
    def player_identity_count(self) -> int:
        """Number of unique player identity records."""

        return len(self.player_identities)


class ApiFootballFixtureSynchronizer:
    """Download completed fixture packages in API-supported batches.

    Every request goes through :class:`ApiFootballClient`, so an interrupted
    run can safely be repeated: already completed requests are read from the
    client's content-addressed cache and only missing batches consume quota.
    """

    def __init__(self, client: ApiFootballClient) -> None:
        self._client = client

    def sync(
        self,
        *,
        league: int,
        season: int,
        include_player_identities: bool = True,
        max_fixture_pages: int = 10,
        max_player_pages: int | None = None,
        batch_size: int = MAX_FIXTURE_IDS_PER_REQUEST,
        use_cache: bool = True,
    ) -> ApiFootballFixtureSyncResult:
        """Fetch and validate one league-season without silent truncation."""

        _validate_sync_options(
            league=league,
            season=season,
            max_fixture_pages=max_fixture_pages,
            max_player_pages=max_player_pages,
            batch_size=batch_size,
        )
        requests_before = self._client.network_requests_made
        listed_fixtures, fixture_list_pages = self._list_completed_fixtures(
            league=league,
            season=season,
            max_pages=max_fixture_pages,
            use_cache=use_cache,
        )
        fixture_ids = _unique_fixture_ids(listed_fixtures)
        fixtures = self._fetch_fixture_batches(
            fixture_ids,
            league=league,
            season=season,
            batch_size=batch_size,
            use_cache=use_cache,
        )

        player_identities: list[dict[str, Any]] = []
        player_pages_fetched = 0
        if include_player_identities:
            player_result = self._client.players(
                league=league,
                season=season,
                max_pages=max_player_pages,
                use_cache=use_cache,
            )
            player_identities = _extract_player_identities(player_result.players)
            player_pages_fetched = player_result.pages_fetched

        return ApiFootballFixtureSyncResult(
            league_id=league,
            season_start_year=season,
            fixtures=fixtures,
            fixture_ids=fixture_ids,
            player_identities=player_identities,
            fixture_list_pages_fetched=fixture_list_pages,
            detail_batches_fetched=_batch_count(len(fixture_ids), batch_size),
            player_pages_fetched=player_pages_fetched,
            network_requests_made=(self._client.network_requests_made - requests_before),
            quota=self._client.last_quota,
        )

    def _list_completed_fixtures(
        self,
        *,
        league: int,
        season: int,
        max_pages: int,
        use_cache: bool,
    ) -> tuple[list[dict[str, Any]], int]:
        base_params: dict[str, QueryValue] = {
            "league": league,
            "season": season,
            "status": COMPLETED_FIXTURE_STATUSES,
        }
        fixtures: list[dict[str, Any]] = []
        expected_total: int | None = None
        current_page = 1

        while expected_total is None or current_page <= expected_total:
            # API-Football normally returns this fixture query in one page.
            # Omitting page=1 also retains compatibility with existing caches.
            params = base_params if current_page == 1 else {**base_params, "page": current_page}
            response = self._client.get("/fixtures", params, use_cache=use_cache)
            _validate_page(response, expected_page=current_page, endpoint="/fixtures")
            if response.paging.total > max_pages:
                raise ApiFootballPageLimitExceeded(
                    "API-Football completed fixture query reported "
                    f"{response.paging.total} pages; configured maximum is {max_pages}"
                )
            expected_total = response.paging.total
            page_items = _response_objects(response, endpoint="/fixtures")
            for item in page_items:
                _validate_fixture_scope(
                    item,
                    league=league,
                    season=season,
                    require_completed=True,
                )
            fixtures.extend(page_items)
            current_page += 1

        return fixtures, current_page - 1

    def _fetch_fixture_batches(
        self,
        fixture_ids: list[int],
        *,
        league: int,
        season: int,
        batch_size: int,
        use_cache: bool,
    ) -> list[dict[str, Any]]:
        by_id: dict[int, dict[str, Any]] = {}
        for start in range(0, len(fixture_ids), batch_size):
            requested_ids = fixture_ids[start : start + batch_size]
            response = self._client.get(
                "/fixtures",
                {"ids": "-".join(str(fixture_id) for fixture_id in requested_ids)},
                use_cache=use_cache,
            )
            _validate_page(response, expected_page=1, endpoint="/fixtures?ids")
            if response.paging.total != 1:
                raise ApiFootballProtocolError(
                    "API-Football fixture-id batch unexpectedly returned multiple pages"
                )
            batch_items = _response_objects(response, endpoint="/fixtures?ids")
            batch_by_id: dict[int, dict[str, Any]] = {}
            for item in batch_items:
                fixture_id = _fixture_id(item)
                _validate_fixture_scope(
                    item,
                    league=league,
                    season=season,
                    require_completed=True,
                )
                if fixture_id not in batch_by_id:
                    batch_by_id[fixture_id] = item

            requested_set = set(requested_ids)
            returned_set = set(batch_by_id)
            if returned_set != requested_set:
                missing = sorted(requested_set - returned_set)
                unexpected = sorted(returned_set - requested_set)
                raise ApiFootballProtocolError(
                    "API-Football fixture-id batch did not match the request "
                    f"(missing={missing}, unexpected={unexpected})"
                )
            for fixture_id in requested_ids:
                by_id.setdefault(fixture_id, batch_by_id[fixture_id])

        return [by_id[fixture_id] for fixture_id in fixture_ids]


def write_fixture_sync_result(
    path: Path,
    result: ApiFootballFixtureSyncResult,
) -> Path:
    """Atomically write a portable raw fixture artifact for later builds."""

    document = {
        "schema_version": "api-football-fixtures-v1",
        "provider": "API-Football",
        "league_id": result.league_id,
        "season_start_year": result.season_start_year,
        "fixture_count": result.fixture_count,
        "fixture_ids": result.fixture_ids,
        "fixtures": result.fixtures,
        # Identity payloads deliberately exclude /players statistics/team blocks.
        "player_identities": result.player_identities,
        "download": {
            "fixture_list_pages_fetched": result.fixture_list_pages_fetched,
            "detail_batches_fetched": result.detail_batches_fetched,
            "player_pages_fetched": result.player_pages_fetched,
            "network_requests_made": result.network_requests_made,
            "quota": asdict(result.quota) if result.quota is not None else None,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            json.dump(document, temporary, ensure_ascii=False, separators=(",", ":"))
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_name = temporary.name
        Path(temporary_name).replace(path)
    finally:
        if temporary_name is not None:
            temporary_path = Path(temporary_name)
            if temporary_path.exists():
                temporary_path.unlink()
    return path


def _validate_sync_options(
    *,
    league: int,
    season: int,
    max_fixture_pages: int,
    max_player_pages: int | None,
    batch_size: int,
) -> None:
    if league < 1:
        raise ValueError("league must be a positive integer")
    if season < 1900:
        raise ValueError("season must be a four-digit start year")
    if max_fixture_pages < 1:
        raise ValueError("max_fixture_pages must be at least one")
    if max_player_pages is not None and max_player_pages < 1:
        raise ValueError("max_player_pages must be at least one")
    if batch_size < 1 or batch_size > MAX_FIXTURE_IDS_PER_REQUEST:
        raise ValueError(f"batch_size must be between 1 and {MAX_FIXTURE_IDS_PER_REQUEST}")


def _validate_page(
    response: ApiFootballResponse,
    *,
    expected_page: int,
    endpoint: str,
) -> None:
    if response.paging.current != expected_page:
        raise ApiFootballProtocolError(
            f"API-Football returned an unexpected page number for {endpoint}"
        )


def _response_objects(
    response: ApiFootballResponse,
    *,
    endpoint: str,
) -> list[dict[str, Any]]:
    if not isinstance(response.response, list) or not all(
        isinstance(item, dict) for item in response.response
    ):
        raise ApiFootballProtocolError(
            f"API-Football {endpoint} response must be an array of objects"
        )
    return cast(list[dict[str, Any]], response.response)


def _fixture_id(item: dict[str, Any]) -> int:
    fixture = item.get("fixture")
    fixture_id = fixture.get("id") if isinstance(fixture, dict) else None
    if isinstance(fixture_id, bool) or not isinstance(fixture_id, int) or fixture_id < 1:
        raise ApiFootballProtocolError("API-Football fixture record has no valid fixture.id")
    return fixture_id


def _validate_fixture_scope(
    item: dict[str, Any],
    *,
    league: int,
    season: int,
    require_completed: bool,
) -> None:
    _fixture_id(item)
    league_data = item.get("league")
    if not isinstance(league_data, dict):
        raise ApiFootballProtocolError("API-Football fixture record has no league metadata")
    if league_data.get("id") != league or league_data.get("season") != season:
        raise ApiFootballProtocolError(
            "API-Football fixture record belongs to a different league-season"
        )
    if require_completed:
        fixture = item["fixture"]
        status = fixture.get("status") if isinstance(fixture, dict) else None
        short = status.get("short") if isinstance(status, dict) else None
        if short not in {"FT", "AET", "PEN"}:
            raise ApiFootballProtocolError(
                "API-Football completed fixture query returned a non-completed record"
            )


def _unique_fixture_ids(fixtures: list[dict[str, Any]]) -> list[int]:
    unique: list[int] = []
    seen: set[int] = set()
    for item in fixtures:
        fixture_id = _fixture_id(item)
        if fixture_id not in seen:
            seen.add(fixture_id)
            unique.append(fixture_id)
    return unique


def _extract_player_identities(
    player_payloads: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    identities: list[dict[str, Any]] = []
    seen: set[int] = set()
    for payload in player_payloads:
        player = payload.get("player")
        if not isinstance(player, dict):
            raise ApiFootballProtocolError(
                "API-Football /players record has no player identity object"
            )
        player_id = player.get("id")
        if isinstance(player_id, bool) or not isinstance(player_id, int) or player_id < 1:
            raise ApiFootballProtocolError("API-Football /players identity has no valid player.id")
        if player_id not in seen:
            seen.add(player_id)
            # Preserve the official {"player": {...}} shape expected by profile
            # builders, but never retain payload["statistics"].
            identities.append({"player": dict(player)})
    return identities


def _batch_count(item_count: int, batch_size: int) -> int:
    return (item_count + batch_size - 1) // batch_size
