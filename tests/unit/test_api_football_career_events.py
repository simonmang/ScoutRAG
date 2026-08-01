"""Transfers/trophies/injuries via the existing licensed API-Football client."""

from __future__ import annotations

import json
from collections.abc import Mapping
from urllib.request import Request

from scoutrag.data.api_football import ApiFootballClient, ApiFootballTransportResponse
from scoutrag.data.api_football_career_events import fetch_career_events


def _envelope(response: object) -> bytes:
    return json.dumps(
        {
            "get": "x",
            "parameters": {},
            "errors": {},
            "results": len(response) if isinstance(response, list) else 1,
            "paging": {"current": 1, "total": 1},
            "response": response,
        }
    ).encode()


class QueuedTransport:
    """Serves canned bodies in call order, like the api_football test suite's own fake."""

    def __init__(self, bodies: list[bytes], *, headers: Mapping[str, str] | None = None) -> None:
        self.bodies = iter(bodies)
        self.headers = headers or {}
        self.requests: list[Request] = []

    def __call__(self, request: Request, timeout: float) -> ApiFootballTransportResponse:
        self.requests.append(request)
        return ApiFootballTransportResponse(
            status=200, headers=self.headers, body=next(self.bodies)
        )


_TRANSFERS = [
    {
        "player": {"id": 502},
        "transfers": [
            {
                "date": "2015-07-01",
                "type": "€ 8.5M",
                "teams": {"in": {"name": "Bayern München"}, "out": {"name": "VfB Stuttgart"}},
            },
            {
                "date": "2013-07-01",
                "type": "Loan",
                "teams": {"in": {"name": "RB Leipzig"}, "out": {"name": "VfB Stuttgart"}},
            },
        ],
    }
]
_TROPHIES = [
    {"league": "Bundesliga", "country": "Germany", "season": "2024/2025", "place": "Winner"},
    {"league": "Super Cup", "country": "Germany", "season": "2023", "place": "2nd Place"},
]
_SIDELINED = [
    {"type": "Hamstring", "start": "2025-02-24", "end": "2025-03-04"},
]


def test_fetches_and_parses_transfers_trophies_and_injuries() -> None:
    transport = QueuedTransport(
        [_envelope(_TRANSFERS), _envelope(_TROPHIES), _envelope(_SIDELINED)]
    )
    client = ApiFootballClient("secret", transport=transport, request_budget=10)

    completed, remaining = fetch_career_events(client, ["api-football:502"])

    assert remaining == []
    assert len(completed) == 1
    events = completed[0]
    assert events.player_id == "api-football:502"
    assert len(events.transfers) == 2
    assert events.transfers[0].fee_text == "€ 8.5M"
    assert events.transfers[0].from_team == "VfB Stuttgart"
    assert events.transfers[0].to_team == "Bayern München"
    assert len(events.trophies) == 2
    assert events.trophies[0].place == "Winner"
    assert len(events.injury_spells) == 1
    assert events.injury_spells[0].injury_type == "Hamstring"


def test_stops_cleanly_and_reports_remaining_players_once_budget_is_exhausted() -> None:
    # Budget of 2 requests: only the first player's /transfers and /trophies calls fit.
    transport = QueuedTransport(
        [_envelope(_TRANSFERS), _envelope(_TROPHIES), _envelope(_SIDELINED)]
    )
    client = ApiFootballClient("secret", transport=transport, request_budget=2)

    completed, remaining = fetch_career_events(client, ["api-football:502", "api-football:999"])

    assert completed == []
    assert remaining == ["api-football:502", "api-football:999"]


def test_malformed_entries_are_skipped_not_raised() -> None:
    bad_transfers = [{"player": {"id": 1}, "transfers": [{"date": "2020-01-01", "type": None}]}]
    bad_trophies = [{"league": "X", "country": "Y", "season": "2020", "place": None}]
    bad_sidelined = [{"start": "2020-01-01", "end": "2020-01-10"}]  # no "type"
    transport = QueuedTransport(
        [_envelope(bad_transfers), _envelope(bad_trophies), _envelope(bad_sidelined)]
    )
    client = ApiFootballClient("secret", transport=transport, request_budget=10)

    completed, remaining = fetch_career_events(client, ["api-football:1"])

    assert remaining == []
    assert completed[0].transfers == []
    assert completed[0].trophies == []
    assert completed[0].injury_spells == []
