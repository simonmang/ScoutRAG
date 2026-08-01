"""Wikidata enrichment matching and merging, with no live network access."""

from datetime import date
from pathlib import Path

import pytest

from scoutrag.data.wikidata_enrichment import (
    CachedSparqlClient,
    WikidataCandidate,
    WikidataTransientError,
    enrich_players,
)


def _binding(**fields: str) -> dict[str, dict[str, str]]:
    return {key: {"type": "literal", "value": value} for key, value in fields.items()}


def _entity_binding(qid: str, **fields: str) -> dict[str, dict[str, str]]:
    row = {"player": {"type": "uri", "value": f"http://www.wikidata.org/entity/{qid}"}}
    row.update(_binding(**fields))
    return row


def _sparql_result(bindings: list[dict[str, dict[str, str]]]) -> dict[str, object]:
    return {"results": {"bindings": bindings}}


class FakeTransport:
    """Dispatches canned responses by inspecting which property a query targets."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.identity_bindings: list[dict[str, dict[str, str]]] = []
        self.footedness_bindings: list[dict[str, dict[str, str]]] = []
        self.national_team_bindings: list[dict[str, dict[str, str]]] = []
        self.honours_bindings: list[dict[str, dict[str, str]]] = []
        self.club_bindings: list[dict[str, dict[str, str]]] = []

    def __call__(self, query: str) -> dict[str, object]:
        self.calls.append(query)
        if "P8006" in query:
            return _sparql_result(self.footedness_bindings)
        if "P1350" in query:
            return _sparql_result(self.national_team_bindings)
        if "P166" in query:
            return _sparql_result(self.honours_bindings)
        if "FILTER NOT EXISTS" in query:
            return _sparql_result(self.club_bindings)
        return _sparql_result(self.identity_bindings)


def _client(transport: FakeTransport, tmp_path: Path | None = None) -> CachedSparqlClient:
    return CachedSparqlClient(transport=transport, cache_dir=tmp_path, throttle_seconds=0.0)


def test_confirmed_match_on_exact_name_and_birthdate() -> None:
    transport = FakeTransport()
    transport.identity_bindings = [
        _entity_binding("Q1", playerLabel="Joshua Kimmich", dob="1995-02-08T00:00:00Z"),
    ]
    transport.national_team_bindings = [
        _entity_binding("Q1", teamLabel="Germany", caps="106"),
        _entity_binding("Q1", teamLabel="Germany under-21", caps="14"),
    ]
    candidates = [
        WikidataCandidate(
            player_id="api-football:502",
            player_name="Joshua Kimmich",
            date_of_birth=date(1995, 2, 8),
        ),
    ]

    results = enrich_players(candidates, client=_client(transport))

    assert len(results) == 1
    assert results[0].wikidata_id == "Q1"
    assert results[0].national_team_name == "Germany"
    assert results[0].national_team_caps == 106  # Highest-caps team wins, not first row.


def test_birthdate_mismatch_is_not_enriched() -> None:
    transport = FakeTransport()
    transport.identity_bindings = [
        _entity_binding("Q1", playerLabel="Ryan Merlen", dob="2002-01-01T00:00:00Z"),
    ]
    candidates = [
        WikidataCandidate(
            player_id="api-football:9", player_name="Ryan Merlen", date_of_birth=date(2002, 5, 11)
        ),
    ]

    results = enrich_players(candidates, client=_client(transport))

    assert results == []


def test_current_club_is_excluded_from_earlier_clubs() -> None:
    transport = FakeTransport()
    transport.identity_bindings = [
        _entity_binding("Q1", playerLabel="Sample Player", dob="2000-01-01T00:00:00Z"),
    ]
    transport.club_bindings = [
        _entity_binding("Q1", teamLabel="Youth Club"),
        _entity_binding("Q1", teamLabel="Sample FC"),
    ]
    candidates = [
        WikidataCandidate(
            player_id="api-football:9",
            player_name="Sample Player",
            date_of_birth=date(2000, 1, 1),
            current_team_names=frozenset({"Sample FC"}),
        ),
    ]

    results = enrich_players(candidates, client=_client(transport))

    assert results[0].earlier_clubs == ["Youth Club"]


def test_ambiguous_name_with_no_birthdate_agreement_is_skipped() -> None:
    transport = FakeTransport()
    transport.identity_bindings = [
        _entity_binding("Q1", playerLabel="Common Name", dob="1990-01-01T00:00:00Z"),
        _entity_binding("Q2", playerLabel="Common Name", dob="1995-06-15T00:00:00Z"),
    ]
    candidates = [
        WikidataCandidate(
            player_id="api-football:1", player_name="Common Name", date_of_birth=date(2001, 3, 3)
        ),
    ]

    results = enrich_players(candidates, client=_client(transport))

    assert results == []


def test_no_footedness_leaves_the_field_none_not_missing() -> None:
    transport = FakeTransport()
    transport.identity_bindings = [
        _entity_binding("Q1", playerLabel="Joshua Kimmich", dob="1995-02-08T00:00:00Z"),
    ]
    candidates = [
        WikidataCandidate(
            player_id="api-football:502",
            player_name="Joshua Kimmich",
            date_of_birth=date(1995, 2, 8),
        ),
    ]

    results = enrich_players(candidates, client=_client(transport))

    assert results[0].footedness is None
    assert results[0].honours == []


def test_rate_limit_is_retried_and_eventually_succeeds() -> None:
    attempts = {"count": 0}
    canned = _sparql_result([_binding(dummy="ok")])

    def flaky_transport(query: str) -> dict[str, object]:
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise WikidataTransientError(status=429, retry_after=0.0)
        return canned

    client = CachedSparqlClient(transport=flaky_transport, throttle_seconds=0.0)

    result = client.query("SELECT * WHERE {}")

    assert attempts["count"] == 2
    assert result == canned


def test_transient_gateway_error_is_also_retried() -> None:
    attempts = {"count": 0}
    canned = _sparql_result([_binding(dummy="ok")])

    def flaky_transport(query: str) -> dict[str, object]:
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise WikidataTransientError(status=502, retry_after=None)
        return canned

    client = CachedSparqlClient(transport=flaky_transport, throttle_seconds=0.0)

    result = client.query("SELECT * WHERE {}")

    assert attempts["count"] == 2
    assert result == canned


def test_network_timeout_is_also_retried() -> None:
    attempts = {"count": 0}
    canned = _sparql_result([_binding(dummy="ok")])

    def flaky_transport(query: str) -> dict[str, object]:
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise WikidataTransientError(status=0, retry_after=None)
        return canned

    client = CachedSparqlClient(transport=flaky_transport, throttle_seconds=0.0)

    result = client.query("SELECT * WHERE {}")

    assert attempts["count"] == 2
    assert result == canned


def test_rate_limit_raises_after_exhausting_retries() -> None:
    def always_limited(query: str) -> dict[str, object]:
        raise WikidataTransientError(status=429, retry_after=0.0)

    client = CachedSparqlClient(transport=always_limited, throttle_seconds=0.0, max_retries=1)
    candidates = [
        WikidataCandidate(
            player_id="api-football:1", player_name="X", date_of_birth=date(2000, 1, 1)
        ),
    ]

    with pytest.raises(WikidataTransientError):
        enrich_players(candidates, client=client)


def test_repeated_query_is_served_from_cache_without_a_second_transport_call(
    tmp_path: Path,
) -> None:
    transport = FakeTransport()
    transport.identity_bindings = [
        _entity_binding("Q1", playerLabel="Joshua Kimmich", dob="1995-02-08T00:00:00Z"),
    ]
    candidates = [
        WikidataCandidate(
            player_id="api-football:502",
            player_name="Joshua Kimmich",
            date_of_birth=date(1995, 2, 8),
        ),
    ]
    client = _client(transport, tmp_path)

    enrich_players(candidates, client=client)
    call_count_after_first_run = len(transport.calls)
    enrich_players(
        candidates,
        client=CachedSparqlClient(transport=transport, cache_dir=tmp_path, throttle_seconds=0.0),
    )

    assert len(transport.calls) == call_count_after_first_run
