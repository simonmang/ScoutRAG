"""Focused tests for the quota-safe API-Football data CLI."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest

from scoutrag.data import cli
from scoutrag.data.api_football import (
    ApiFootballPaging,
    ApiFootballPlayersResult,
    ApiFootballQuota,
    ApiFootballResponse,
    QueryValue,
)


def _response(endpoint: str, response: object) -> ApiFootballResponse:
    return ApiFootballResponse(
        endpoint=endpoint,
        parameters={},
        results=1,
        paging=ApiFootballPaging(current=1, total=1),
        response=response,
        quota=ApiFootballQuota(
            requests_limit=100,
            requests_remaining=98,
            rate_limit=10,
            rate_remaining=9,
        ),
        from_cache=False,
        cache_path=None,
    )


class FakeApiFootballClient:
    """Record orchestration without performing HTTP calls."""

    instances: ClassVar[list[FakeApiFootballClient]] = []
    events: ClassVar[list[str]] = []
    player_coverage: ClassVar[bool] = True

    def __init__(self, api_key: str, **options: object) -> None:
        self.api_key = api_key
        self.options = options
        self.network_requests_made = 2
        self.last_quota = ApiFootballQuota(
            requests_limit=100,
            requests_remaining=98,
        )
        type(self).instances.append(self)

    def status(self) -> ApiFootballResponse:
        type(self).events.append("status")
        return _response(
            "/status",
            {
                "account": {
                    "email": "private@example.test",
                    "api_key": self.api_key,
                },
                "subscription": {
                    "plan": "Free",
                    "end": "2026-08-01",
                    "active": True,
                    "api_key": self.api_key,
                },
                "requests": {"current": 2, "limit": 100},
            },
        )

    def get(
        self,
        endpoint: str,
        params: Mapping[str, QueryValue] | None = None,
        *,
        use_cache: bool = True,
    ) -> ApiFootballResponse:
        del params, use_cache
        type(self).events.append("coverage")
        assert endpoint == "/leagues"
        return _response(
            endpoint,
            [
                {
                    "league": {"id": 78},
                    "seasons": [
                        {
                            "year": 2024,
                            "coverage": {"players": type(self).player_coverage},
                        }
                    ],
                }
            ],
        )

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
        del player, search, use_cache
        type(self).events.append("players")
        assert (league, season, team, max_pages) == (78, 2024, 157, 3)
        return ApiFootballPlayersResult(
            players=[{"player": {"id": 1, "name": "Test Player"}}],
            pages_fetched=1,
            quota=self.last_quota,
        )


@pytest.fixture(autouse=True)
def _reset_fake_client() -> None:
    FakeApiFootballClient.instances = []
    FakeApiFootballClient.events = []
    FakeApiFootballClient.player_coverage = True


def test_api_football_sync_parser_defaults_to_bayern_and_small_budget() -> None:
    args = cli.build_parser().parse_args(["api-football-sync"])

    assert args.league_id == 78
    assert args.season == 2024
    assert args.team_id == 157
    assert args.all_teams is False
    assert args.request_budget == 10
    assert args.max_pages == 3
    assert args.cache == Path("data/raw")
    assert args.output == Path("data/processed/api-football-bayern-2024-2025")


def test_fixture_sync_parser_uses_pro_safe_full_league_defaults() -> None:
    args = cli.build_parser().parse_args(["api-football-fixture-sync"])

    assert args.league_id == 78
    assert args.season == 2024
    assert args.request_budget == 100
    assert args.max_fixture_pages == 10
    assert args.max_player_pages == 50
    assert args.batch_size == 20
    assert args.skip_player_identities is False
    assert args.throttle_seconds == 0.25
    assert args.output == Path("data/raw/api-football-bundesliga-2024-2025-fixtures.json")


def test_fixture_build_parser_targets_canonical_bundesliga_dataset() -> None:
    args = cli.build_parser().parse_args(["api-football-fixture-build"])

    assert args.input == Path("data/raw/api-football-bundesliga-2024-2025-fixtures.json")
    assert args.output == Path("data/processed/api-football-bundesliga-2024-2025")
    assert args.round_prefix == "Regular Season"


def test_fixture_merge_parser_uses_explicit_quality_gates() -> None:
    args = cli.build_parser().parse_args(["api-football-fixture-merge"])

    assert args.input == Path("data/processed/scouting-2025-2026")
    assert args.output == Path("data/processed/scouting-2025-2026/combined")
    assert args.minimum_profiles == 250
    assert args.minimum_teams == 10
    assert args.minimum_full_sample_profiles == 100
    assert args.minimum_median_quality == 0.75


def test_fixture_sync_cli_writes_validated_raw_artifact(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    secret = "fixture-sync-secret"
    monkeypatch.setenv("API_FOOTBALL_KEY", secret)
    monkeypatch.setattr(cli, "ApiFootballClient", FakeApiFootballClient)
    sync_calls: list[dict[str, object]] = []

    class FakeSynchronizer:
        def __init__(self, client: object) -> None:
            assert isinstance(client, FakeApiFootballClient)

        def sync(self, **options: object) -> SimpleNamespace:
            sync_calls.append(options)
            return SimpleNamespace(
                league_id=78,
                season_start_year=2024,
                fixture_count=308,
                player_identity_count=511,
                fixture_list_pages_fetched=1,
                detail_batches_fetched=16,
                player_pages_fetched=38,
                network_requests_made=0,
                quota=ApiFootballQuota(
                    requests_limit=7500,
                    requests_remaining=7499,
                ),
            )

    written: list[tuple[Path, object]] = []

    def fake_write(path: Path, result: object) -> Path:
        written.append((path, result))
        return path

    monkeypatch.setattr(cli, "ApiFootballFixtureSynchronizer", FakeSynchronizer)
    monkeypatch.setattr(cli, "write_fixture_sync_result", fake_write)
    output_path = tmp_path / "fixtures.json"

    exit_code = cli.main(
        [
            "api-football-fixture-sync",
            "--output",
            str(output_path),
            "--cache",
            str(tmp_path / "cache"),
            "--throttle-seconds",
            "0",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert secret not in captured.out
    assert sync_calls == [
        {
            "league": 78,
            "season": 2024,
            "include_player_identities": True,
            "max_fixture_pages": 10,
            "max_player_pages": 50,
            "batch_size": 20,
            "use_cache": True,
        }
    ]
    assert written[0][0] == output_path
    output = json.loads(captured.out)
    assert output["completed_fixtures"] == 308
    assert output["player_identities"] == 511
    assert output["network_requests_made"] == 0


def test_fixture_build_cli_uses_fixture_provenance(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "fixtures.json"
    input_path.write_text(
        json.dumps(
            {
                "schema_version": "api-football-fixtures-v1",
                "league_id": 78,
                "season_start_year": 2024,
                "fixtures": [{"fixture": {"id": 1}}],
                "player_identities": [{"player": {"id": 10}}],
            }
        ),
        encoding="utf-8",
    )
    build_calls: list[dict[str, object]] = []
    write_calls: list[dict[str, object]] = []

    def fake_build(
        payloads: list[dict[str, Any]],
        **options: object,
    ) -> SimpleNamespace:
        build_calls.append({"payloads": payloads, **options})
        return SimpleNamespace(
            profiles=[object()],
            evidence=[object(), object()],
            definitions=[object()],
            limitations=["fixture limitation"],
        )

    class FakeWriter:
        def write(self, output_root: Path, **options: object) -> list[Path]:
            write_calls.append({"output_root": output_root, **options})
            return [output_root / "manifest.json"]

    monkeypatch.setattr(cli, "build_api_football_fixture_profiles", fake_build)
    monkeypatch.setattr(cli, "ApiFootballDatasetWriter", FakeWriter)
    output_root = tmp_path / "processed"

    exit_code = cli.main(
        [
            "api-football-fixture-build",
            "--input",
            str(input_path),
            "--output",
            str(output_root),
        ]
    )

    assert exit_code == 0
    assert build_calls[0]["round_prefix"] == "Regular Season"
    assert build_calls[0]["player_identity_payloads"] == [{"player": {"id": 10}}]
    assert write_calls[0]["schema_version"] == "api-football-fixture-v1"
    assert write_calls[0]["source_endpoint"] == "/fixtures?ids"
    output = json.loads(capsys.readouterr().out)
    assert output["profiles"] == 1
    assert output["metric_evidence"] == 2


def test_status_reports_safe_subscription_and_quota_without_key(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = "status-secret-must-not-leak"
    monkeypatch.setenv("API_FOOTBALL_KEY", secret)
    monkeypatch.setattr(cli, "ApiFootballClient", FakeApiFootballClient)

    exit_code = cli.main(["api-football-status"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert secret not in captured.out
    assert secret not in captured.err
    assert "private@example.test" not in captured.out
    output = json.loads(captured.out)
    assert output["connected"] is True
    assert output["subscription"]["plan"] == "Free"
    assert output["requests"] == {"current": 2, "limit": 100}
    assert output["quota"]["requests_remaining"] == 98


def test_team_sync_checks_coverage_then_disables_percentiles(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    secret = "sync-secret-must-not-leak"
    monkeypatch.setenv("API_FOOTBALL_KEY", secret)
    monkeypatch.setattr(cli, "ApiFootballClient", FakeApiFootballClient)
    builder_calls: list[dict[str, object]] = []

    def fake_build(
        payloads: list[dict[str, Any]],
        **options: object,
    ) -> SimpleNamespace:
        FakeApiFootballClient.events.append("build")
        builder_calls.append({"payloads": payloads, **options})
        return SimpleNamespace(profiles=[object()], evidence=[object(), object()])

    class FakeWriter:
        def write(
            self,
            output_root: Path,
            *,
            result: object,
            league_id: int,
            season_start_year: int,
            competition_name: str,
        ) -> list[Path]:
            del result, league_id, season_start_year, competition_name
            FakeApiFootballClient.events.append("write")
            return [output_root / "player_season_profiles.parquet"]

    monkeypatch.setattr(cli, "build_api_football_profiles", fake_build)
    monkeypatch.setattr(cli, "ApiFootballDatasetWriter", FakeWriter)

    exit_code = cli.main(
        [
            "api-football-sync",
            "--cache",
            str(tmp_path / "raw"),
            "--output",
            str(tmp_path / "processed"),
            "--throttle-seconds",
            "0",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert secret not in captured.out
    assert FakeApiFootballClient.events == ["coverage", "players", "build", "write"]
    assert builder_calls[0]["enable_percentiles"] is False
    assert builder_calls[0]["comparison_scope"] == "FC Bayern München team-filtered sample"
    assert FakeApiFootballClient.instances[0].options["cache_dir"] == tmp_path / "raw"
    output = json.loads(captured.out)
    assert output["team_id"] == 157
    assert output["percentiles_enabled"] is False
    assert output["profiles"] == 1


def test_sync_refuses_missing_player_coverage_before_player_request(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("API_FOOTBALL_KEY", "coverage-test-secret")
    monkeypatch.setattr(cli, "ApiFootballClient", FakeApiFootballClient)
    FakeApiFootballClient.player_coverage = False

    exit_code = cli.main(["api-football-sync", "--throttle-seconds", "0"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert FakeApiFootballClient.events == ["coverage"]
    assert "no player coverage" in captured.err


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        (
            ["--all-teams", "--max-pages", "24", "--request-budget", "100"],
            "requires --max-pages of at least 25",
        ),
        (
            ["--all-teams", "--max-pages", "25", "--request-budget", "25"],
            "requires --request-budget of at least --max-pages + 1",
        ),
    ],
)
def test_full_league_sync_requires_explicit_sensible_guards(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    arguments: list[str],
    expected: str,
) -> None:
    monkeypatch.setenv("API_FOOTBALL_KEY", "full-league-test-secret")
    monkeypatch.setattr(cli, "ApiFootballClient", FakeApiFootballClient)

    exit_code = cli.main(["api-football-sync", *arguments])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert expected in captured.err
    assert not FakeApiFootballClient.instances


def test_missing_key_fails_without_constructing_client(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("API_FOOTBALL_KEY", raising=False)
    monkeypatch.setattr(cli, "Settings", lambda: SimpleNamespace(api_football_key=None))
    monkeypatch.setattr(cli, "ApiFootballClient", FakeApiFootballClient)

    exit_code = cli.main(["api-football-status"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "API_FOOTBALL_KEY is not configured" in captured.err
    assert not FakeApiFootballClient.instances
