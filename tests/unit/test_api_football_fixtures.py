"""Tests for resumable API-Football fixture synchronization."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast
from urllib.request import Request

import pytest

from scoutrag.data.api_football import (
    ApiFootballClient,
    ApiFootballPaging,
    ApiFootballPlayersResult,
    ApiFootballProtocolError,
    ApiFootballQuota,
    ApiFootballResponse,
    ApiFootballTransportResponse,
    QueryValue,
)
from scoutrag.data.api_football_fixtures import (
    ApiFootballFixtureSynchronizer,
    write_fixture_sync_result,
)


def _fixture(fixture_id: int, *, status: str = "FT") -> dict[str, Any]:
    return {
        "fixture": {"id": fixture_id, "status": {"short": status}},
        "league": {"id": 78, "season": 2024},
        "players": [{"not": "validated by the downloader"}],
    }


def _response(
    payload: object,
    *,
    current: int = 1,
    total: int = 1,
    from_cache: bool = False,
) -> ApiFootballResponse:
    return ApiFootballResponse(
        endpoint="/fixtures",
        parameters={},
        results=len(payload) if isinstance(payload, list) else 1,
        paging=ApiFootballPaging(current=current, total=total),
        response=payload,
        quota=ApiFootballQuota(requests_limit=7500, requests_remaining=7490),
        from_cache=from_cache,
        cache_path=None,
    )


class FakeClient:
    """In-memory client stub; no network transport is used."""

    def __init__(
        self,
        listing: list[dict[str, Any]],
        batch_payloads: dict[str, list[dict[str, Any]]],
    ) -> None:
        self.listing = listing
        self.batch_payloads = batch_payloads
        self.network_requests_made = 0
        self.last_quota = ApiFootballQuota(
            requests_limit=7500,
            requests_remaining=7490,
        )
        self.calls: list[tuple[str, dict[str, QueryValue]]] = []

    def get(
        self,
        endpoint: str,
        params: Mapping[str, QueryValue] | None = None,
        *,
        use_cache: bool = True,
    ) -> ApiFootballResponse:
        del use_cache
        normalized = dict(params or {})
        self.calls.append((endpoint, normalized))
        self.network_requests_made += 1
        ids = normalized.get("ids")
        if isinstance(ids, str):
            return _response(self.batch_payloads[ids])
        return _response(self.listing)

    def players(
        self,
        *,
        league: int,
        season: int,
        team: int | None = None,
        player: int | None = None,
        search: str | None = None,
        max_pages: int | None = None,
        use_cache: bool = True,
    ) -> ApiFootballPlayersResult:
        del team, player, search, use_cache
        assert (league, season, max_pages) == (78, 2024, 50)
        self.network_requests_made += 2
        return ApiFootballPlayersResult(
            players=[
                {
                    "player": {"id": 9, "name": "Identity only"},
                    "statistics": [{"team": {"id": 999}, "games": {"minutes": 1}}],
                },
                {
                    "player": {"id": 9, "name": "Duplicate identity"},
                    "statistics": [{"must": "not survive"}],
                },
                {"player": {"id": 10, "name": "Second identity"}},
            ],
            pages_fetched=2,
            quota=self.last_quota,
        )


def _synchronizer(client: FakeClient) -> ApiFootballFixtureSynchronizer:
    # The production synchronizer deliberately depends on the concrete,
    # quota-aware client; this narrow cast keeps the mock fully offline.
    return ApiFootballFixtureSynchronizer(cast(Any, client))


def test_sync_batches_deduplicates_and_restores_listing_order() -> None:
    client = FakeClient(
        [_fixture(3), _fixture(1), _fixture(3), _fixture(2)],
        {
            "3-1": [_fixture(1), _fixture(3)],
            "2": [_fixture(2), _fixture(2)],
        },
    )

    result = _synchronizer(client).sync(
        league=78,
        season=2024,
        max_player_pages=50,
        batch_size=2,
    )

    assert result.fixture_ids == [3, 1, 2]
    assert [item["fixture"]["id"] for item in result.fixtures] == [3, 1, 2]
    assert result.fixture_count == 3
    assert result.detail_batches_fetched == 2
    assert result.fixture_list_pages_fetched == 1
    assert result.player_pages_fetched == 2
    assert result.network_requests_made == 5
    assert result.player_identities == [
        {"player": {"id": 9, "name": "Identity only"}},
        {"player": {"id": 10, "name": "Second identity"}},
    ]
    assert client.calls == [
        (
            "/fixtures",
            {"league": 78, "season": 2024, "status": "FT-AET-PEN"},
        ),
        ("/fixtures", {"ids": "3-1"}),
        ("/fixtures", {"ids": "2"}),
    ]


def test_sync_can_skip_player_identity_endpoint() -> None:
    client = FakeClient([_fixture(4)], {"4": [_fixture(4)]})

    result = _synchronizer(client).sync(
        league=78,
        season=2024,
        include_player_identities=False,
    )

    assert result.player_identities == []
    assert result.player_pages_fetched == 0
    assert result.network_requests_made == 2


def test_sync_rejects_missing_or_unexpected_batch_ids() -> None:
    client = FakeClient(
        [_fixture(1), _fixture(2)],
        {"1-2": [_fixture(1), _fixture(99)]},
    )

    with pytest.raises(ApiFootballProtocolError, match="did not match"):
        _synchronizer(client).sync(
            league=78,
            season=2024,
            include_player_identities=False,
        )


@pytest.mark.parametrize(
    ("listing", "message"),
    [
        ([_fixture(1, status="NS")], "non-completed"),
        (
            [
                {
                    "fixture": {"id": 1, "status": {"short": "FT"}},
                    "league": {"id": 39, "season": 2024},
                }
            ],
            "different league-season",
        ),
    ],
)
def test_sync_rejects_out_of_scope_listing(
    listing: list[dict[str, Any]],
    message: str,
) -> None:
    client = FakeClient(listing, {})

    with pytest.raises(ApiFootballProtocolError, match=message):
        _synchronizer(client).sync(
            league=78,
            season=2024,
            include_player_identities=False,
        )


def test_sync_validates_api_batch_limit_before_requests() -> None:
    client = FakeClient([], {})

    with pytest.raises(ValueError, match="between 1 and 20"):
        _synchronizer(client).sync(
            league=78,
            season=2024,
            batch_size=21,
        )

    assert client.calls == []


def test_writer_persists_portable_fixture_artifact(tmp_path: Path) -> None:
    client = FakeClient([_fixture(4)], {"4": [_fixture(4)]})
    result = _synchronizer(client).sync(
        league=78,
        season=2024,
        include_player_identities=False,
    )

    output = write_fixture_sync_result(tmp_path / "nested" / "fixtures.json", result)
    document = json.loads(output.read_text(encoding="utf-8"))

    assert document["schema_version"] == "api-football-fixtures-v1"
    assert document["fixture_count"] == 1
    assert document["fixture_ids"] == [4]
    assert document["download"]["network_requests_made"] == 2
    assert document["download"]["quota"]["requests_limit"] == 7500
    assert not list(tmp_path.rglob("*.tmp"))


def test_interrupted_rerun_reuses_content_addressed_request_cache(
    tmp_path: Path,
) -> None:
    responses = iter(
        [
            _envelope_bytes([_fixture(7)]),
            _envelope_bytes([_fixture(7)]),
        ]
    )
    transport_calls: list[str] = []

    def recording_transport(
        request: Request,
        timeout: float,
    ) -> ApiFootballTransportResponse:
        assert timeout > 0
        transport_calls.append(request.full_url)
        return ApiFootballTransportResponse(
            status=200,
            headers={"x-ratelimit-requests-remaining": "7498"},
            body=next(responses),
        )

    first_client = ApiFootballClient(
        "not-written-to-cache",
        cache_dir=tmp_path,
        transport=recording_transport,
        request_budget=2,
    )
    first = ApiFootballFixtureSynchronizer(first_client).sync(
        league=78,
        season=2024,
        include_player_identities=False,
    )

    def forbidden_transport(
        request: Request,
        timeout: float,
    ) -> ApiFootballTransportResponse:
        del request, timeout
        raise AssertionError("a completed cached request must not be repeated")

    resumed_client = ApiFootballClient(
        "a-different-key-is-safe",
        cache_dir=tmp_path,
        transport=forbidden_transport,
        request_budget=0,
    )
    resumed = ApiFootballFixtureSynchronizer(resumed_client).sync(
        league=78,
        season=2024,
        include_player_identities=False,
    )

    assert first.fixture_ids == resumed.fixture_ids == [7]
    assert len(transport_calls) == 2
    assert resumed.network_requests_made == 0
    assert not any(
        b"not-written-to-cache" in path.read_bytes() for path in tmp_path.rglob("*.json")
    )


def _envelope_bytes(response: list[dict[str, Any]]) -> bytes:
    return json.dumps(
        {
            "get": "fixtures",
            "parameters": {},
            "errors": [],
            "results": len(response),
            "paging": {"current": 1, "total": 1},
            "response": response,
        }
    ).encode()
