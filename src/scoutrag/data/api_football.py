"""Small, secure API-Football v3 client with quota and cache guards.

The client intentionally uses only the Python standard library.  Its transport
is injectable so importing and unit testing data never requires live network
access.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, cast
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

API_FOOTBALL_BASE_URL = "https://v3.football.api-sports.io"
_USER_AGENT = "ScoutRAG/1.0"

QueryValue = str | int | float | bool


class ApiFootballError(RuntimeError):
    """Base error for API-Football client failures."""


class ApiFootballProtocolError(ApiFootballError):
    """Raised when API-Football returns an invalid standard envelope."""


class ApiFootballResponseError(ApiFootballError):
    """Raised when the standard response envelope contains API errors."""


class ApiFootballBudgetExceeded(ApiFootballError):
    """Raised before a network request would exceed the configured budget."""


class ApiFootballQuotaExceeded(ApiFootballBudgetExceeded):
    """Raised before a request when the most recent quota reports exhaustion."""


class ApiFootballPageLimitExceeded(ApiFootballBudgetExceeded):
    """Raised when an endpoint reports more pages than the configured guard."""


@dataclass(frozen=True, slots=True)
class ApiFootballTransportResponse:
    """Raw response returned by an injectable HTTP transport."""

    status: int
    headers: Mapping[str, str]
    body: bytes


Transport = Callable[[Request, float], ApiFootballTransportResponse]


@dataclass(frozen=True, slots=True)
class ApiFootballQuota:
    """Quota values reported by API-Sports response headers.

    API-Sports uses ``requests_*`` for the daily allowance and the shorter
    ``rate_*`` pair for the current rate-limit window.
    """

    requests_limit: int | None = None
    requests_remaining: int | None = None
    rate_limit: int | None = None
    rate_remaining: int | None = None


@dataclass(frozen=True, slots=True)
class ApiFootballPaging:
    """Page metadata from an API-Football response envelope."""

    current: int
    total: int


@dataclass(frozen=True, slots=True)
class ApiFootballResponse:
    """Validated API-Football response including provenance metadata."""

    endpoint: str
    parameters: dict[str, Any]
    results: int
    paging: ApiFootballPaging
    response: object
    quota: ApiFootballQuota
    from_cache: bool
    cache_path: Path | None


@dataclass(frozen=True, slots=True)
class ApiFootballPlayersResult:
    """All validated player records collected across the reported pages."""

    players: list[dict[str, Any]]
    pages_fetched: int
    quota: ApiFootballQuota | None


class UrlLibApiFootballTransport:
    """Default GET-only HTTP transport based on :mod:`urllib`."""

    def __call__(self, request: Request, timeout: float) -> ApiFootballTransportResponse:
        try:
            with urlopen(request, timeout=timeout) as response:
                return ApiFootballTransportResponse(
                    status=int(response.status),
                    headers=dict(response.headers.items()),
                    body=cast(bytes, response.read()),
                )
        except HTTPError as exc:
            return ApiFootballTransportResponse(
                status=int(exc.code),
                headers=dict(exc.headers.items()) if exc.headers is not None else {},
                body=exc.read(),
            )


class ApiFootballClient:
    """Quota-aware API-Football v3 client.

    The API key is kept in a private attribute, sent only in the
    ``x-apisports-key`` header, and omitted from URLs, cache keys, exceptions,
    and representations.
    """

    def __init__(
        self,
        api_key: str,
        *,
        cache_dir: Path | None = None,
        transport: Transport | None = None,
        base_url: str = API_FOOTBALL_BASE_URL,
        timeout: float = 30.0,
        request_budget: int = 100,
        max_pages: int = 10,
        min_request_interval_seconds: float = 0.0,
        sleeper: Callable[[float], None] = time.sleep,
        monotonic_clock: Callable[[], float] = time.monotonic,
    ) -> None:
        normalized_key = api_key.strip()
        if not normalized_key:
            raise ValueError("api_key must not be empty")
        if timeout <= 0:
            raise ValueError("timeout must be greater than zero")
        if request_budget < 0:
            raise ValueError("request_budget must not be negative")
        if max_pages < 1:
            raise ValueError("max_pages must be at least one")
        if not math.isfinite(min_request_interval_seconds) or min_request_interval_seconds < 0:
            raise ValueError("min_request_interval_seconds must be a finite nonnegative number")
        if not base_url.lower().startswith("https://"):
            raise ValueError("base_url must use HTTPS")

        self._api_key = normalized_key
        self._cache_dir = cache_dir
        self._transport = transport or UrlLibApiFootballTransport()
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._request_budget = request_budget
        self._max_pages = max_pages
        self._min_request_interval_seconds = min_request_interval_seconds
        self._sleeper = sleeper
        self._monotonic_clock = monotonic_clock
        self._last_request_started_at: float | None = None
        self._network_requests_made = 0
        self._last_quota: ApiFootballQuota | None = None
        self._budget_lock = Lock()

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(base_url={self._base_url!r}, "
            f"request_budget={self._request_budget}, max_pages={self._max_pages})"
        )

    @property
    def network_requests_made(self) -> int:
        """Number of attempted transport calls in this client instance."""

        return self._network_requests_made

    @property
    def last_quota(self) -> ApiFootballQuota | None:
        """Most recently observed server quota metadata, if any."""

        return self._last_quota

    def get(
        self,
        endpoint: str,
        params: Mapping[str, QueryValue] | None = None,
        *,
        use_cache: bool = True,
    ) -> ApiFootballResponse:
        """Perform one GET and return a validated standard response envelope."""

        normalized_endpoint = self._normalize_endpoint(endpoint)
        normalized_params = dict(sorted((params or {}).items()))
        url = self._build_url(normalized_endpoint, normalized_params)
        cache_path = self._cache_path(normalized_endpoint, normalized_params)

        if use_cache and cache_path is not None and cache_path.is_file():
            try:
                cached = cache_path.read_bytes()
                return self._decode_response(
                    cached,
                    endpoint=normalized_endpoint,
                    status=200,
                    quota=ApiFootballQuota(),
                    from_cache=True,
                    cache_path=cache_path,
                )
            except (OSError, ApiFootballProtocolError):
                # A stale or interrupted cache must never be treated as API data.
                pass

        self._reserve_network_request()
        request = Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": _USER_AGENT,
                "x-apisports-key": self._api_key,
            },
            method="GET",
        )
        try:
            transport_response = self._transport(request, self._timeout)
        except (OSError, URLError) as exc:
            raise ApiFootballError(
                f"API-Football request failed: {self._redact(str(exc))}"
            ) from None
        except Exception as exc:
            raise ApiFootballError(
                f"API-Football transport failed: {self._redact(str(exc))}"
            ) from None

        quota = _quota_from_headers(transport_response.headers)
        self._last_quota = quota
        decoded = self._decode_response(
            transport_response.body,
            endpoint=normalized_endpoint,
            status=transport_response.status,
            quota=quota,
            from_cache=False,
            cache_path=cache_path,
        )
        if use_cache and cache_path is not None:
            self._atomic_write(cache_path, transport_response.body)
        return decoded

    def status(self, *, use_cache: bool = False) -> ApiFootballResponse:
        """Return account, subscription, and current request status."""

        return self.get("/status", use_cache=use_cache)

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
        """Collect all pages from the ``/players`` endpoint.

        The method refuses to return a silently truncated data set.  If the API
        reports more pages than the configured guard, it raises before fetching
        the excess pages.
        """

        if league < 1:
            raise ValueError("league must be a positive integer")
        if season < 1900:
            raise ValueError("season must be a four-digit start year")
        if team is not None and team < 1:
            raise ValueError("team must be a positive integer")
        if player is not None and player < 1:
            raise ValueError("player must be a positive integer")
        if search is not None and len(search.strip()) < 4:
            raise ValueError("search must contain at least four characters")
        if max_pages is not None and max_pages < 1:
            raise ValueError("max_pages must be at least one")

        page_limit = min(max_pages, self._max_pages) if max_pages is not None else self._max_pages
        base_params: dict[str, QueryValue] = {"league": league, "season": season}
        if team is not None:
            base_params["team"] = team
        if player is not None:
            base_params["id"] = player
        if search is not None:
            base_params["search"] = search.strip()

        records: list[dict[str, Any]] = []
        pages_fetched = 0
        latest_quota: ApiFootballQuota | None = None
        expected_total: int | None = None
        current_page = 1

        while expected_total is None or current_page <= expected_total:
            page_result = self.get(
                "/players",
                {**base_params, "page": current_page},
                use_cache=use_cache,
            )
            if page_result.paging.current != current_page:
                raise ApiFootballProtocolError(
                    "API-Football returned an unexpected player page number"
                )
            if not isinstance(page_result.response, list) or not all(
                isinstance(item, dict) for item in page_result.response
            ):
                raise ApiFootballProtocolError(
                    "API-Football /players response must be an array of objects"
                )

            expected_total = page_result.paging.total
            if expected_total > page_limit:
                raise ApiFootballPageLimitExceeded(
                    f"API-Football /players reported {expected_total} pages; "
                    f"configured maximum is {page_limit}"
                )

            records.extend(cast(list[dict[str, Any]], page_result.response))
            pages_fetched += 1
            if not page_result.from_cache:
                latest_quota = page_result.quota
            current_page += 1

        return ApiFootballPlayersResult(
            players=records,
            pages_fetched=pages_fetched,
            quota=latest_quota,
        )

    def _reserve_network_request(self) -> None:
        with self._budget_lock:
            if (
                self._last_quota is not None
                and self._last_quota.requests_remaining is not None
                and self._last_quota.requests_remaining <= 0
            ):
                raise ApiFootballQuotaExceeded("API-Football daily request quota is exhausted")
            if self._network_requests_made >= self._request_budget:
                raise ApiFootballBudgetExceeded(
                    f"API-Football request budget of {self._request_budget} is exhausted"
                )
            now = self._monotonic_clock()
            if self._last_request_started_at is not None:
                elapsed = now - self._last_request_started_at
                wait_seconds = self._min_request_interval_seconds - elapsed
                if wait_seconds > 0:
                    self._sleeper(wait_seconds)
                    now = self._monotonic_clock()
            self._last_request_started_at = now
            self._network_requests_made += 1

    def _decode_response(
        self,
        body: bytes,
        *,
        endpoint: str,
        status: int,
        quota: ApiFootballQuota,
        from_cache: bool,
        cache_path: Path | None,
    ) -> ApiFootballResponse:
        try:
            raw = cast(object, json.loads(body))
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise ApiFootballProtocolError(
                f"API-Football returned invalid JSON for {endpoint}"
            ) from None
        if not isinstance(raw, dict):
            raise ApiFootballProtocolError(
                f"API-Football envelope for {endpoint} must be an object"
            )

        errors = raw.get("errors")
        if not isinstance(errors, (dict, list)):
            raise ApiFootballProtocolError(
                f"API-Football envelope for {endpoint} has invalid errors metadata"
            )
        if errors:
            error_text = self._redact(_format_api_errors(errors))
            raise ApiFootballResponseError(f"API-Football rejected {endpoint}: {error_text}")

        if self._api_key.encode("utf-8") in body:
            raise ApiFootballProtocolError(
                "API-Football response unexpectedly contained credential material"
            )
        if status < 200 or status >= 300:
            raise ApiFootballResponseError(f"API-Football returned HTTP {status} for {endpoint}")

        parameters = raw.get("parameters")
        results = raw.get("results")
        paging = raw.get("paging")
        if parameters == []:
            parameters = {}
        if not isinstance(parameters, dict):
            raise ApiFootballProtocolError(
                f"API-Football envelope for {endpoint} has invalid parameters metadata"
            )
        if isinstance(results, bool) or not isinstance(results, int) or results < 0:
            raise ApiFootballProtocolError(
                f"API-Football envelope for {endpoint} has invalid results metadata"
            )
        if not isinstance(paging, dict):
            raise ApiFootballProtocolError(
                f"API-Football envelope for {endpoint} has invalid paging metadata"
            )
        current = paging.get("current")
        total = paging.get("total")
        if (
            isinstance(current, bool)
            or not isinstance(current, int)
            or current < 1
            or isinstance(total, bool)
            or not isinstance(total, int)
            or total < 1
        ):
            raise ApiFootballProtocolError(
                f"API-Football envelope for {endpoint} has invalid page values"
            )
        if "response" not in raw:
            raise ApiFootballProtocolError(
                f"API-Football envelope for {endpoint} is missing response data"
            )

        return ApiFootballResponse(
            endpoint=endpoint,
            parameters=cast(dict[str, Any], parameters),
            results=results,
            paging=ApiFootballPaging(current=current, total=total),
            response=raw["response"],
            quota=quota,
            from_cache=from_cache,
            cache_path=cache_path,
        )

    def _build_url(
        self,
        endpoint: str,
        params: Mapping[str, QueryValue],
    ) -> str:
        query = urlencode(list(params.items()))
        return f"{self._base_url}{endpoint}" + (f"?{query}" if query else "")

    @staticmethod
    def _normalize_endpoint(endpoint: str) -> str:
        normalized = "/" + endpoint.strip().lstrip("/")
        if (
            normalized == "/"
            or "://" in normalized
            or "?" in normalized
            or "#" in normalized
            or ".." in normalized.split("/")
        ):
            raise ValueError("endpoint must be a relative API path without a query string")
        return normalized

    def _cache_path(
        self,
        endpoint: str,
        params: Mapping[str, QueryValue],
    ) -> Path | None:
        if self._cache_dir is None:
            return None
        canonical = json.dumps(
            {"endpoint": endpoint, "parameters": params},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        digest = hashlib.sha256(canonical).hexdigest()
        endpoint_name = endpoint.strip("/").replace("/", "_")
        return self._cache_dir / "api_football" / endpoint_name / f"{digest}.json"

    @staticmethod
    def _atomic_write(path: Path, body: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary.write(body)
                temporary.flush()
                os.fsync(temporary.fileno())
                temporary_name = temporary.name
            Path(temporary_name).replace(path)
        finally:
            if temporary_name is not None:
                temporary_path = Path(temporary_name)
                if temporary_path.exists():
                    temporary_path.unlink()

    def _redact(self, text: str) -> str:
        return text.replace(self._api_key, "[REDACTED]")


def _quota_from_headers(headers: Mapping[str, str]) -> ApiFootballQuota:
    normalized = {key.lower(): value for key, value in headers.items()}
    return ApiFootballQuota(
        requests_limit=_optional_int(normalized.get("x-ratelimit-requests-limit")),
        requests_remaining=_optional_int(normalized.get("x-ratelimit-requests-remaining")),
        rate_limit=_optional_int(normalized.get("x-ratelimit-limit")),
        rate_remaining=_optional_int(normalized.get("x-ratelimit-remaining")),
    )


def _optional_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _format_api_errors(errors: dict[object, object] | list[object]) -> str:
    if isinstance(errors, dict):
        return "; ".join(f"{key}: {value}" for key, value in errors.items())
    return "; ".join(str(value) for value in errors)
