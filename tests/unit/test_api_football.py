"""Focused tests for the secure, quota-aware API-Football client."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from urllib.request import Request

import pytest

from scoutrag.data.api_football import (
    ApiFootballBudgetExceeded,
    ApiFootballClient,
    ApiFootballPageLimitExceeded,
    ApiFootballProtocolError,
    ApiFootballQuotaExceeded,
    ApiFootballResponseError,
    ApiFootballTransportResponse,
)


def _envelope(
    response: object,
    *,
    current: int = 1,
    total: int = 1,
    errors: object | None = None,
) -> bytes:
    return json.dumps(
        {
            "get": "players",
            "parameters": {},
            "errors": {} if errors is None else errors,
            "results": len(response) if isinstance(response, list) else 1,
            "paging": {"current": current, "total": total},
            "response": response,
        }
    ).encode()


class RecordingTransport:
    def __init__(
        self,
        bodies: list[bytes],
        *,
        headers: Mapping[str, str] | None = None,
        status: int = 200,
    ) -> None:
        self.bodies = iter(bodies)
        self.headers = headers or {}
        self.status = status
        self.requests: list[Request] = []

    def __call__(self, request: Request, timeout: float) -> ApiFootballTransportResponse:
        assert timeout > 0
        self.requests.append(request)
        return ApiFootballTransportResponse(
            status=self.status,
            headers=self.headers,
            body=next(self.bodies),
        )


def test_get_uses_header_only_and_collects_quota_metadata() -> None:
    secret = "super-secret-api-key"
    transport = RecordingTransport(
        [_envelope([{"player": {"id": 42}}])],
        headers={
            "X-RateLimit-Requests-Limit": "100",
            "x-ratelimit-requests-remaining": "99",
            "x-ratelimit-limit": "10",
            "x-ratelimit-remaining": "9",
        },
    )
    client = ApiFootballClient(secret, transport=transport)

    result = client.get("/players", {"season": 2023, "league": 78})

    request = transport.requests[0]
    assert request.get_method() == "GET"
    assert request.headers["X-apisports-key"] == secret
    assert secret not in request.full_url
    assert "league=78&season=2023" in request.full_url
    assert result.quota.requests_limit == 100
    assert result.quota.requests_remaining == 99
    assert result.quota.rate_limit == 10
    assert result.quota.rate_remaining == 9
    assert secret not in repr(client)


def test_valid_response_is_cached_atomically_and_reused(tmp_path: Path) -> None:
    body = _envelope([{"player": {"id": 42}}])
    transport = RecordingTransport([body])
    client = ApiFootballClient(
        "not-in-cache",
        cache_dir=tmp_path,
        transport=transport,
    )

    first = client.get("/players", {"league": 78, "season": 2023})
    second = client.get("/players", {"season": 2023, "league": 78})

    assert first.from_cache is False
    assert second.from_cache is True
    assert second.response == first.response
    assert client.network_requests_made == 1
    assert first.cache_path is not None
    assert first.cache_path.read_bytes() == body
    assert not list(tmp_path.rglob("*.tmp"))
    assert b"not-in-cache" not in first.cache_path.read_bytes()


def test_invalid_cache_is_not_returned_as_data(tmp_path: Path) -> None:
    transport = RecordingTransport([_envelope([{"fresh": True}])])
    client = ApiFootballClient("secret", cache_dir=tmp_path, transport=transport)
    cache_path = client._cache_path("/players", {"league": 78, "season": 2023})
    assert cache_path is not None
    cache_path.parent.mkdir(parents=True)
    cache_path.write_text("not-json", encoding="utf-8")

    result = client.get("/players", {"season": 2023, "league": 78})

    assert result.response == [{"fresh": True}]
    assert result.from_cache is False
    assert client.network_requests_made == 1


def test_players_collects_every_page_and_preserves_records() -> None:
    transport = RecordingTransport(
        [
            _envelope([{"player": {"id": 1}}], current=1, total=2),
            _envelope([{"player": {"id": 2}}], current=2, total=2),
        ],
        headers={"x-ratelimit-requests-remaining": "98"},
    )
    client = ApiFootballClient("secret", transport=transport, max_pages=5)

    result = client.players(league=78, season=2023)

    assert [item["player"]["id"] for item in result.players] == [1, 2]
    assert result.pages_fetched == 2
    assert result.quota is not None
    assert result.quota.requests_remaining == 98
    assert [
        parse_qs(urlparse(request.full_url).query)["page"][0] for request in transport.requests
    ] == ["1", "2"]


def test_players_obeys_minimum_network_request_interval() -> None:
    transport = RecordingTransport(
        [
            _envelope([{"player": {"id": 1}}], current=1, total=2),
            _envelope([{"player": {"id": 2}}], current=2, total=2),
        ]
    )
    current_time = [100.0]
    waits: list[float] = []

    def clock() -> float:
        return current_time[0]

    def sleeper(seconds: float) -> None:
        waits.append(seconds)
        current_time[0] += seconds

    client = ApiFootballClient(
        "secret",
        transport=transport,
        min_request_interval_seconds=6.2,
        sleeper=sleeper,
        monotonic_clock=clock,
    )

    result = client.players(league=78, season=2023)

    assert result.pages_fetched == 2
    assert waits == pytest.approx([6.2])


def test_client_rejects_invalid_request_interval() -> None:
    with pytest.raises(ValueError, match="finite nonnegative"):
        ApiFootballClient("secret", min_request_interval_seconds=-0.1)
    with pytest.raises(ValueError, match="finite nonnegative"):
        ApiFootballClient("secret", min_request_interval_seconds=float("inf"))


def test_players_search_requires_four_characters() -> None:
    client = ApiFootballClient("secret", transport=RecordingTransport([]))

    with pytest.raises(ValueError, match="at least four"):
        client.players(league=78, season=2023, search="abc")


def test_players_refuses_silent_truncation_at_page_guard() -> None:
    transport = RecordingTransport([_envelope([{"player": {"id": 1}}], current=1, total=20)])
    client = ApiFootballClient("secret", transport=transport, max_pages=3)

    with pytest.raises(ApiFootballPageLimitExceeded, match="reported 20 pages"):
        client.players(league=78, season=2023)

    assert client.network_requests_made == 1


def test_request_budget_is_checked_before_transport_call() -> None:
    transport = RecordingTransport([_envelope([])])
    client = ApiFootballClient("secret", transport=transport, request_budget=1)

    client.status()

    with pytest.raises(ApiFootballBudgetExceeded, match="budget of 1"):
        client.status()
    assert len(transport.requests) == 1


def test_reported_daily_quota_stops_follow_up_request() -> None:
    transport = RecordingTransport(
        [_envelope([])],
        headers={"x-ratelimit-requests-remaining": "0"},
    )
    client = ApiFootballClient("secret", transport=transport, request_budget=10)

    client.status()

    with pytest.raises(ApiFootballQuotaExceeded, match="quota is exhausted"):
        client.status()
    assert len(transport.requests) == 1


@pytest.mark.parametrize(
    "body, expected",
    [
        (b"[]", "must be an object"),
        (
            json.dumps(
                {
                    "parameters": {},
                    "errors": {},
                    "results": "one",
                    "paging": {"current": 1, "total": 1},
                    "response": [],
                }
            ).encode(),
            "invalid results",
        ),
        (
            json.dumps(
                {
                    "parameters": {},
                    "errors": {},
                    "results": 0,
                    "paging": {"current": 0, "total": 1},
                    "response": [],
                }
            ).encode(),
            "invalid page values",
        ),
    ],
)
def test_get_rejects_invalid_standard_envelopes(body: bytes, expected: str) -> None:
    client = ApiFootballClient("secret", transport=RecordingTransport([body]))

    with pytest.raises(ApiFootballProtocolError, match=expected):
        client.get("/players")


def test_api_errors_are_validated_and_secret_is_redacted() -> None:
    secret = "super-secret-api-key"
    body = _envelope([], errors={"auth": f"invalid {secret}"})
    client = ApiFootballClient(secret, transport=RecordingTransport([body]))

    with pytest.raises(ApiFootballResponseError) as exc_info:
        client.get("/players")

    assert secret not in str(exc_info.value)
    assert "[REDACTED]" in str(exc_info.value)


def test_status_uses_live_status_endpoint_by_default() -> None:
    status_payload = {
        "account": {"firstname": "Scout"},
        "subscription": {"plan": "Free"},
        "requests": {"current": 1, "limit_day": 100},
    }
    transport = RecordingTransport([_envelope(status_payload)])
    client = ApiFootballClient("secret", transport=transport)

    result = client.status()

    assert result.response == status_payload
    assert transport.requests[0].full_url.endswith("/status")


def test_status_accepts_official_empty_parameters_array() -> None:
    body = json.dumps(
        {
            "get": "status",
            "parameters": [],
            "errors": [],
            "results": 1,
            "paging": {"current": 1, "total": 1},
            "response": {"requests": {"current": 1, "limit_day": 100}},
        }
    ).encode()
    client = ApiFootballClient("secret", transport=RecordingTransport([body]))

    result = client.status()

    assert result.parameters == {}


def test_nonempty_parameters_array_remains_invalid() -> None:
    body = json.dumps(
        {
            "get": "status",
            "parameters": ["unexpected"],
            "errors": [],
            "results": 1,
            "paging": {"current": 1, "total": 1},
            "response": {},
        }
    ).encode()
    client = ApiFootballClient("secret", transport=RecordingTransport([body]))

    with pytest.raises(ApiFootballProtocolError, match="invalid parameters"):
        client.status()


def test_response_containing_api_key_is_never_cached(tmp_path: Path) -> None:
    secret = "server-echoed-secret"
    transport = RecordingTransport([_envelope([{"echo": secret}])])
    client = ApiFootballClient(secret, cache_dir=tmp_path, transport=transport)

    with pytest.raises(ApiFootballProtocolError, match="credential material"):
        client.get("/players")

    assert not list(tmp_path.rglob("*.json"))
