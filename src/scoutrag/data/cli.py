"""Command-line entry point for reproducible football data builds."""

import argparse
import json
import math
import sys
from collections.abc import Sequence
from dataclasses import asdict
from datetime import date
from pathlib import Path
from statistics import median
from typing import Any

import pyarrow.parquet as pq  # type: ignore[import-untyped]

from scoutrag.config import Settings
from scoutrag.data.api_football import (
    ApiFootballClient,
    ApiFootballError,
    ApiFootballProtocolError,
    ApiFootballResponse,
)
from scoutrag.data.api_football_fixture_profiles import (
    build_api_football_fixture_profiles,
)
from scoutrag.data.api_football_fixtures import (
    ApiFootballFixtureSynchronizer,
    write_fixture_sync_result,
)
from scoutrag.data.api_football_profiles import (
    ApiFootballDatasetWriter,
    ApiFootballProfileResult,
    build_api_football_profiles,
)
from scoutrag.data.feature_engineering import FeatureEngineeringConfig
from scoutrag.data.models import DownloadSummary, PipelineResult
from scoutrag.data.pipeline import Phase3DataPipeline
from scoutrag.data.statsbomb import StatsBombOpenDataDownloader
from scoutrag.data.temporal import build_season_trends
from scoutrag.domain.player import (
    MetricDefinition,
    PlayerIdentity,
    PlayerMatchPerformance,
    PlayerRecentForm,
    PlayerSeasonTrend,
    PlayerTeamSeasonStint,
)
from scoutrag.governance.evidence import load_metric_evidence
from scoutrag.retrieval.common import load_profiles

DEFAULT_COMPETITION_ID = 9
DEFAULT_SEASON_ID = 281
DEFAULT_RAW_ROOT = Path("data/raw/statsbomb")
DEFAULT_PROCESSED_ROOT = Path("data/processed/bundesliga-2023-2024")
DEFAULT_API_FOOTBALL_LEAGUE_ID = 78
DEFAULT_API_FOOTBALL_SEASON = 2024
DEFAULT_API_FOOTBALL_TEAM_ID = 157
DEFAULT_API_FOOTBALL_COMPETITION_NAME = "Bundesliga"
DEFAULT_API_FOOTBALL_CACHE_ROOT = Path("data/raw")
DEFAULT_API_FOOTBALL_OUTPUT_ROOT = Path("data/processed/api-football-bayern-2024-2025")
DEFAULT_API_FOOTBALL_REQUEST_BUDGET = 10
DEFAULT_API_FOOTBALL_MAX_PAGES = 3
DEFAULT_API_FOOTBALL_THROTTLE_SECONDS = 6.1
MINIMUM_FULL_LEAGUE_MAX_PAGES = 25
DEFAULT_API_FOOTBALL_FIXTURE_OUTPUT = Path(
    "data/raw/api-football-bundesliga-2024-2025-fixtures.json"
)
DEFAULT_API_FOOTBALL_FIXTURE_BUILD_OUTPUT = Path("data/processed/api-football-bundesliga-2024-2025")
DEFAULT_API_FOOTBALL_SCOUTING_ROOT = Path("data/processed/scouting-2025-2026")
DEFAULT_API_FOOTBALL_SCOUTING_COMBINED = DEFAULT_API_FOOTBALL_SCOUTING_ROOT / "combined"
DEFAULT_API_FOOTBALL_FIXTURE_REQUEST_BUDGET = 100
DEFAULT_API_FOOTBALL_FIXTURE_MAX_PAGES = 10
DEFAULT_API_FOOTBALL_PLAYER_MAX_PAGES = 50
DEFAULT_API_FOOTBALL_PRO_THROTTLE_SECONDS = 0.25


def _add_competition_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--competition-id",
        type=int,
        default=DEFAULT_COMPETITION_ID,
    )
    parser.add_argument("--season-id", type=int, default=DEFAULT_SEASON_ID)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scoutrag-data",
        description="Download and build typed ScoutRAG football evidence.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    download_parser = subparsers.add_parser(
        "download",
        help="Download one StatsBomb competition-season.",
    )
    _add_competition_arguments(download_parser)
    download_parser.add_argument("--output", type=Path, default=DEFAULT_RAW_ROOT)
    download_parser.add_argument("--match-limit", type=int)

    build_parser_command = subparsers.add_parser(
        "build",
        help="Normalize downloaded JSON and write Parquet artifacts.",
    )
    _add_competition_arguments(build_parser_command)
    build_parser_command.add_argument("--input", type=Path, default=DEFAULT_RAW_ROOT)
    build_parser_command.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_PROCESSED_ROOT,
    )
    build_parser_command.add_argument("--minimum-minutes", type=float, default=450)
    build_parser_command.add_argument("--full-sample-minutes", type=float, default=900)
    build_parser_command.add_argument(
        "--minimum-comparison-group-size",
        type=int,
        default=3,
    )
    build_parser_command.add_argument(
        "--minimum-source-coverage",
        type=float,
        default=0.8,
    )

    status_parser = subparsers.add_parser(
        "api-football-status",
        help="Check API-Football access and show safe quota information.",
    )
    status_parser.add_argument("--timeout", type=float, default=30.0)

    sync_parser = subparsers.add_parser(
        "api-football-sync",
        help="Build canonical player profiles from API-Football season aggregates.",
    )
    sync_parser.add_argument(
        "--league-id",
        type=int,
        default=DEFAULT_API_FOOTBALL_LEAGUE_ID,
    )
    sync_parser.add_argument(
        "--season",
        type=int,
        default=DEFAULT_API_FOOTBALL_SEASON,
        help="Four-digit season start year.",
    )
    sync_parser.add_argument(
        "--team-id",
        type=int,
        default=DEFAULT_API_FOOTBALL_TEAM_ID,
        help="Team filter used unless --all-teams is supplied.",
    )
    sync_parser.add_argument(
        "--all-teams",
        action="store_true",
        help="Explicitly download the full league instead of the default Bayern sample.",
    )
    sync_parser.add_argument(
        "--competition-name",
        default=DEFAULT_API_FOOTBALL_COMPETITION_NAME,
    )
    sync_parser.add_argument(
        "--cache",
        type=Path,
        default=DEFAULT_API_FOOTBALL_CACHE_ROOT,
    )
    sync_parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_API_FOOTBALL_OUTPUT_ROOT,
    )
    sync_parser.add_argument(
        "--request-budget",
        type=int,
        default=DEFAULT_API_FOOTBALL_REQUEST_BUDGET,
        help="Maximum network requests for this command.",
    )
    sync_parser.add_argument(
        "--max-pages",
        type=int,
        default=DEFAULT_API_FOOTBALL_MAX_PAGES,
        help="Maximum /players pages; the sync fails instead of truncating.",
    )
    sync_parser.add_argument(
        "--throttle-seconds",
        type=float,
        default=DEFAULT_API_FOOTBALL_THROTTLE_SECONDS,
        help="Minimum interval between network requests (6.1s is safe for 10/min).",
    )
    sync_parser.add_argument("--timeout", type=float, default=30.0)
    sync_parser.add_argument("--minimum-minutes", type=float, default=450)
    sync_parser.add_argument("--full-sample-minutes", type=float, default=900)
    sync_parser.add_argument(
        "--minimum-comparison-group-size",
        type=int,
        default=3,
    )

    fixture_sync_parser = subparsers.add_parser(
        "api-football-fixture-sync",
        help="Cache completed fixture packages for trustworthy season aggregation.",
    )
    fixture_sync_parser.add_argument(
        "--league-id",
        type=int,
        default=DEFAULT_API_FOOTBALL_LEAGUE_ID,
    )
    fixture_sync_parser.add_argument(
        "--season",
        type=int,
        default=DEFAULT_API_FOOTBALL_SEASON,
        help="Four-digit season start year.",
    )
    fixture_sync_parser.add_argument(
        "--cache",
        type=Path,
        default=DEFAULT_API_FOOTBALL_CACHE_ROOT,
    )
    fixture_sync_parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_API_FOOTBALL_FIXTURE_OUTPUT,
    )
    fixture_sync_parser.add_argument(
        "--request-budget",
        type=int,
        default=DEFAULT_API_FOOTBALL_FIXTURE_REQUEST_BUDGET,
        help="Maximum network requests; cached requests do not consume this budget.",
    )
    fixture_sync_parser.add_argument(
        "--max-fixture-pages",
        type=int,
        default=DEFAULT_API_FOOTBALL_FIXTURE_MAX_PAGES,
    )
    fixture_sync_parser.add_argument(
        "--max-player-pages",
        type=int,
        default=DEFAULT_API_FOOTBALL_PLAYER_MAX_PAGES,
        help="Fail instead of silently truncating /players identity metadata.",
    )
    fixture_sync_parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Fixture IDs per request (API-Football maximum: 20).",
    )
    fixture_sync_parser.add_argument(
        "--skip-player-identities",
        action="store_true",
        help="Do not download supplemental /players identity metadata.",
    )
    fixture_sync_parser.add_argument(
        "--throttle-seconds",
        type=float,
        default=DEFAULT_API_FOOTBALL_PRO_THROTTLE_SECONDS,
        help="Minimum interval between network requests (0.25s is below Pro's 300/min).",
    )
    fixture_sync_parser.add_argument("--timeout", type=float, default=30.0)

    fixture_build_parser = subparsers.add_parser(
        "api-football-fixture-build",
        help="Build canonical player profiles from a cached fixture sync artifact.",
    )
    fixture_build_parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_API_FOOTBALL_FIXTURE_OUTPUT,
    )
    fixture_build_parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_API_FOOTBALL_FIXTURE_BUILD_OUTPUT,
    )
    fixture_build_parser.add_argument(
        "--competition-name",
        default=DEFAULT_API_FOOTBALL_COMPETITION_NAME,
    )
    fixture_build_parser.add_argument("--minimum-minutes", type=float, default=450)
    fixture_build_parser.add_argument("--full-sample-minutes", type=float, default=900)
    fixture_build_parser.add_argument(
        "--minimum-comparison-group-size",
        type=int,
        default=3,
    )
    fixture_build_parser.add_argument(
        "--round-prefix",
        default="Regular Season",
        help="Only fixtures whose league round begins with this value are aggregated.",
    )
    fixture_build_parser.add_argument(
        "--season-name",
        help="Display label override, for example 2025 for calendar-year leagues.",
    )
    fixture_build_parser.add_argument(
        "--include-same-league-postseason",
        action="store_true",
        help=(
            "Include later rounds only when every team also appeared in the regular league phase."
        ),
    )

    fixture_merge_parser = subparsers.add_parser(
        "api-football-fixture-merge",
        help="Validate and merge league-specific fixture datasets for retrieval.",
    )
    fixture_merge_parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_API_FOOTBALL_SCOUTING_ROOT,
    )
    fixture_merge_parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_API_FOOTBALL_SCOUTING_COMBINED,
    )
    fixture_merge_parser.add_argument(
        "--season",
        type=int,
        default=DEFAULT_API_FOOTBALL_SEASON,
        help="Four-digit season start year stored in the combined manifest.",
    )
    fixture_merge_parser.add_argument("--minimum-profiles", type=int, default=250)
    fixture_merge_parser.add_argument("--minimum-teams", type=int, default=10)
    fixture_merge_parser.add_argument(
        "--minimum-full-sample-profiles",
        type=int,
        default=100,
    )
    fixture_merge_parser.add_argument(
        "--minimum-median-quality",
        type=float,
        default=0.75,
    )
    history_parser = subparsers.add_parser(
        "api-football-history-build",
        help="Combine quality-gated season datasets without averaging seasons.",
    )
    history_parser.add_argument(
        "--input",
        type=Path,
        action="append",
        required=True,
        help="A quality-gated season combined directory; repeat for each season.",
    )
    history_parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/scouting-history"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "api-football-status":
        return _api_football_status(args)
    if args.command == "api-football-sync":
        return _api_football_sync(args)
    if args.command == "api-football-fixture-sync":
        return _api_football_fixture_sync(args)
    if args.command == "api-football-fixture-build":
        return _api_football_fixture_build(args)
    if args.command == "api-football-fixture-merge":
        return _api_football_fixture_merge(args)
    if args.command == "api-football-history-build":
        return _api_football_history_build(args)
    return _statsbomb_command(args)


def _statsbomb_command(args: argparse.Namespace) -> int:
    result: DownloadSummary | PipelineResult
    if args.command == "download":
        result = StatsBombOpenDataDownloader().download(
            args.competition_id,
            args.season_id,
            args.output,
            match_limit=args.match_limit,
        )
    else:
        result = Phase3DataPipeline(
            feature_config=FeatureEngineeringConfig(
                minimum_minutes=args.minimum_minutes,
                full_sample_minutes=args.full_sample_minutes,
                minimum_comparison_group_size=args.minimum_comparison_group_size,
                minimum_source_coverage=args.minimum_source_coverage,
            )
        ).run(
            args.input,
            args.output,
            competition_id=args.competition_id,
            season_id=args.season_id,
        )
    print(result.model_dump_json(indent=2))
    if isinstance(result, PipelineResult) and not result.validation.valid:
        return 2
    return 0


def _api_football_status(args: argparse.Namespace) -> int:
    api_key = _api_football_key()
    if api_key is None:
        return _missing_api_football_key()
    try:
        client = ApiFootballClient(
            api_key,
            request_budget=1,
            max_pages=1,
            timeout=args.timeout,
        )
        response = client.status()
        _print_json(_status_summary(response))
    except (ApiFootballError, ValueError) as exc:
        return _report_api_error(exc, api_key)
    return 0


def _api_football_sync(args: argparse.Namespace) -> int:
    api_key = _api_football_key()
    if api_key is None:
        return _missing_api_football_key()
    try:
        _validate_api_football_sync_arguments(args)
        client = ApiFootballClient(
            api_key,
            cache_dir=args.cache,
            request_budget=args.request_budget,
            max_pages=args.max_pages,
            min_request_interval_seconds=args.throttle_seconds,
            timeout=args.timeout,
        )
        coverage = client.get(
            "/leagues",
            {"id": args.league_id, "season": args.season},
            use_cache=True,
        )
        if not _has_player_coverage(
            coverage,
            league_id=args.league_id,
            season_start_year=args.season,
        ):
            raise ApiFootballProtocolError(
                "API-Football reports no player coverage for the selected league-season"
            )

        selected_team_id = None if args.all_teams else args.team_id
        players = client.players(
            league=args.league_id,
            season=args.season,
            team=selected_team_id,
            max_pages=args.max_pages,
            use_cache=True,
        )
        result = build_api_football_profiles(
            players.players,
            league_id=args.league_id,
            season_start_year=args.season,
            competition_name=args.competition_name,
            minimum_minutes=args.minimum_minutes,
            full_sample_minutes=args.full_sample_minutes,
            minimum_comparison_group_size=args.minimum_comparison_group_size,
            enable_percentiles=args.all_teams,
            comparison_scope=_comparison_scope(args),
        )
        output_files = ApiFootballDatasetWriter().write(
            args.output,
            result=result,
            league_id=args.league_id,
            season_start_year=args.season,
            competition_name=args.competition_name,
        )
        _print_json(
            {
                "provider": "API-Football",
                "league_id": args.league_id,
                "season_start_year": args.season,
                "team_id": selected_team_id,
                "all_teams": args.all_teams,
                "profiles": len(result.profiles),
                "metric_evidence": len(result.evidence),
                "players_pages_fetched": players.pages_fetched,
                "network_requests_made": client.network_requests_made,
                "percentiles_enabled": args.all_teams,
                "cache_directory": str(args.cache),
                "output_directory": str(args.output),
                "output_files": [str(path) for path in output_files],
                "quota": asdict(client.last_quota) if client.last_quota is not None else None,
            }
        )
    except (ApiFootballError, OSError, ValueError) as exc:
        return _report_api_error(exc, api_key)
    return 0


def _api_football_fixture_sync(args: argparse.Namespace) -> int:
    api_key = _api_football_key()
    if api_key is None:
        return _missing_api_football_key()
    try:
        _validate_api_football_fixture_sync_arguments(args)
        client = ApiFootballClient(
            api_key,
            cache_dir=args.cache,
            request_budget=args.request_budget,
            max_pages=args.max_player_pages,
            min_request_interval_seconds=args.throttle_seconds,
            timeout=args.timeout,
        )
        result = ApiFootballFixtureSynchronizer(client).sync(
            league=args.league_id,
            season=args.season,
            include_player_identities=not args.skip_player_identities,
            max_fixture_pages=args.max_fixture_pages,
            max_player_pages=args.max_player_pages,
            batch_size=args.batch_size,
            use_cache=True,
        )
        output_path = write_fixture_sync_result(args.output, result)
        _print_json(
            {
                "provider": "API-Football",
                "league_id": result.league_id,
                "season_start_year": result.season_start_year,
                "completed_fixtures": result.fixture_count,
                "player_identities": result.player_identity_count,
                "fixture_list_pages_fetched": result.fixture_list_pages_fetched,
                "fixture_detail_batches": result.detail_batches_fetched,
                "player_pages_fetched": result.player_pages_fetched,
                "network_requests_made": result.network_requests_made,
                "cache_directory": str(args.cache),
                "output_file": str(output_path),
                "quota": asdict(result.quota) if result.quota is not None else None,
            }
        )
    except (ApiFootballError, OSError, ValueError) as exc:
        return _report_api_error(exc, api_key)
    return 0


def _api_football_fixture_build(args: argparse.Namespace) -> int:
    try:
        document = json.loads(args.input.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            raise ValueError("fixture sync artifact must contain a JSON object")
        if document.get("schema_version") != "api-football-fixtures-v1":
            raise ValueError("unsupported fixture sync schema_version")
        league_id = document.get("league_id")
        season_start_year = document.get("season_start_year")
        fixtures = document.get("fixtures")
        player_identities = document.get("player_identities")
        if not isinstance(league_id, int) or league_id < 1:
            raise ValueError("fixture sync artifact has an invalid league_id")
        if not isinstance(season_start_year, int) or season_start_year < 1900:
            raise ValueError("fixture sync artifact has an invalid season_start_year")
        if not isinstance(fixtures, list):
            raise ValueError("fixture sync artifact has no fixtures array")
        if not isinstance(player_identities, list):
            raise ValueError("fixture sync artifact has no player_identities array")

        result = build_api_football_fixture_profiles(
            fixtures,
            league_id=league_id,
            season_start_year=season_start_year,
            competition_name=args.competition_name,
            player_identity_payloads=player_identities,
            minimum_minutes=args.minimum_minutes,
            full_sample_minutes=args.full_sample_minutes,
            minimum_comparison_group_size=args.minimum_comparison_group_size,
            round_prefix=args.round_prefix,
            season_name=args.season_name,
            include_same_league_postseason=args.include_same_league_postseason,
        )
        output_files = ApiFootballDatasetWriter().write(
            args.output,
            result=result,
            league_id=league_id,
            season_start_year=season_start_year,
            competition_name=args.competition_name,
            schema_version="api-football-fixture-v1",
            source_endpoint="/fixtures?ids",
            source_details={
                "fixture_sync_schema_version": document["schema_version"],
                "completed_fixtures_in_source": len(fixtures),
                "round_filter": args.round_prefix,
                "include_same_league_postseason": args.include_same_league_postseason,
                "performance_source": "fixture player statistics",
                "identity_source": "/players player root fields only",
            },
        )
        _print_json(
            {
                "provider": "API-Football",
                "league_id": league_id,
                "season_start_year": season_start_year,
                "round_filter": args.round_prefix,
                "season_name": args.season_name or f"{season_start_year}/{season_start_year + 1}",
                "include_same_league_postseason": args.include_same_league_postseason,
                "profiles": len(result.profiles),
                "metric_evidence": len(result.evidence),
                "metric_definitions": len(result.definitions),
                "input_file": str(args.input),
                "output_directory": str(args.output),
                "output_files": [str(path) for path in output_files],
                "limitations": result.limitations,
            }
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


def _api_football_fixture_merge(args: argparse.Namespace) -> int:
    try:
        if args.minimum_profiles < 1 or args.minimum_teams < 1:
            raise ValueError("merge profile and team thresholds must be positive")
        if args.minimum_full_sample_profiles < 1:
            raise ValueError("--minimum-full-sample-profiles must be positive")
        if not 0 <= args.minimum_median_quality <= 1:
            raise ValueError("--minimum-median-quality must be between zero and one")

        profiles = []
        evidence = []
        identities_by_player: dict[str, PlayerIdentity] = {}
        stints: list[PlayerTeamSeasonStint] = []
        match_performances: list[PlayerMatchPerformance] = []
        recent_forms: list[PlayerRecentForm] = []
        definitions: list[MetricDefinition] | None = None
        league_ids: list[int] = []
        accepted: list[str] = []
        excluded: dict[str, list[str]] = {}
        quality_report: dict[str, dict[str, object]] = {}
        output_resolved = args.output.resolve()

        for dataset_root in sorted(args.input.iterdir(), key=lambda path: path.name):
            if not dataset_root.is_dir() or dataset_root.resolve() == output_resolved:
                continue
            profile_path = dataset_root / "player_season_profiles.parquet"
            evidence_path = dataset_root / "player_metric_evidence.parquet"
            definitions_path = dataset_root / "metric_definitions.json"
            manifest_path = dataset_root / "manifest.json"
            required = (profile_path, evidence_path, definitions_path, manifest_path)
            if not all(path.exists() for path in required):
                continue

            league_profiles = load_profiles(profile_path)
            league_evidence = load_metric_evidence(evidence_path)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            league_definitions = [
                MetricDefinition.model_validate(item)
                for item in json.loads(definitions_path.read_text(encoding="utf-8"))
            ]
            if definitions is None:
                definitions = league_definitions
            elif league_definitions != definitions:
                raise ValueError(f"metric definitions differ in dataset {dataset_root.name}")

            team_count = len({profile.team_name for profile in league_profiles})
            full_sample_count = sum(profile.minutes_played >= 900 for profile in league_profiles)
            median_quality = (
                round(median(profile.data_quality for profile in league_profiles), 3)
                if league_profiles
                else 0
            )
            reasons = []
            if len(league_profiles) < args.minimum_profiles:
                reasons.append(f"profiles={len(league_profiles)} < {args.minimum_profiles}")
            if team_count < args.minimum_teams:
                reasons.append(f"teams={team_count} < {args.minimum_teams}")
            if full_sample_count < args.minimum_full_sample_profiles:
                reasons.append(
                    "full_sample_profiles="
                    f"{full_sample_count} < {args.minimum_full_sample_profiles}"
                )
            if median_quality < args.minimum_median_quality:
                reasons.append(f"median_quality={median_quality} < {args.minimum_median_quality}")
            quality_report[dataset_root.name] = {
                "profiles": len(league_profiles),
                "teams": team_count,
                "full_sample_profiles": full_sample_count,
                "median_quality": median_quality,
                "accepted": not reasons,
                "reasons": reasons,
            }
            if reasons:
                excluded[dataset_root.name] = reasons
                continue

            source = manifest.get("source")
            league_id = source.get("league_id") if isinstance(source, dict) else None
            if not isinstance(league_id, int):
                raise ValueError(f"dataset {dataset_root.name} has no integer league_id")
            league_ids.append(league_id)
            accepted.append(dataset_root.name)
            profiles.extend(league_profiles)
            evidence.extend(league_evidence)
            identity_path = dataset_root / "player_identities.parquet"
            if identity_path.exists():
                for row in _typed_parquet_rows(identity_path):
                    identity = PlayerIdentity.model_validate(row)
                    current = identities_by_player.get(identity.player_id)
                    if current is None or _identity_quality(identity) > _identity_quality(current):
                        identities_by_player[identity.player_id] = identity
            stint_path = dataset_root / "player_team_season_stints.parquet"
            if stint_path.exists():
                stints.extend(
                    PlayerTeamSeasonStint.model_validate(row)
                    for row in _typed_parquet_rows(stint_path)
                )
            match_path = dataset_root / "player_match_performances.parquet"
            if match_path.exists():
                match_performances.extend(
                    PlayerMatchPerformance.model_validate(row)
                    for row in _typed_parquet_rows(match_path)
                )
            form_path = dataset_root / "player_recent_form.parquet"
            if form_path.exists():
                recent_forms.extend(
                    PlayerRecentForm.model_validate(row) for row in _typed_parquet_rows(form_path)
                )

        if not accepted or definitions is None:
            raise ValueError("no league dataset passed the merge quality thresholds")
        profile_ids = [profile.profile_id for profile in profiles]
        if any(profile_id is None for profile_id in profile_ids):
            raise ValueError("all merged profiles require a competition-season profile_id")
        if len(profile_ids) != len(set(profile_ids)):
            raise ValueError("merged datasets contain duplicate profile_id values")
        known_profile_ids = set(profile_ids)
        unknown_evidence = {
            item.profile_id or "<missing>"
            for item in evidence
            if item.profile_id not in known_profile_ids
        }
        if unknown_evidence:
            raise ValueError(
                f"metric evidence references unknown profile IDs: {sorted(unknown_evidence)[:5]}"
            )

        result = ApiFootballProfileResult(
            profiles=sorted(
                profiles,
                key=lambda item: (
                    item.player_name.casefold(),
                    item.profile_id or "",
                ),
            ),
            evidence=sorted(
                evidence,
                key=lambda item: (
                    item.profile_id or "",
                    item.metric_name,
                    item.source_reference,
                ),
            ),
            definitions=definitions,
            limitations=[
                "Position percentiles remain league- and position-specific; they are "
                "not recalculated across competitions.",
                "Datasets below the declared profile, team, full-sample, or median "
                "quality thresholds are quarantined from retrieval.",
            ],
            identities=sorted(
                identities_by_player.values(),
                key=lambda item: (item.player_name.casefold(), item.player_id),
            ),
            stints=sorted(stints, key=lambda item: item.stint_id),
            match_performances=sorted(
                match_performances,
                key=lambda item: (
                    item.match_date or date.min,
                    item.fixture_id,
                    item.player_id,
                ),
            ),
            recent_forms=sorted(
                recent_forms,
                key=lambda item: (item.profile_id, item.player_id),
            ),
        )
        output_files = ApiFootballDatasetWriter().write(
            args.output,
            result=result,
            league_id=sorted(league_ids),
            season_start_year=args.season,
            competition_name="European Scouting Universe",
            schema_version="api-football-scouting-v1",
            source_endpoint="/fixtures?ids",
            source_details={
                "accepted_datasets": accepted,
                "excluded_datasets": excluded,
                "quality_thresholds": {
                    "minimum_profiles": args.minimum_profiles,
                    "minimum_teams": args.minimum_teams,
                    "minimum_full_sample_profiles": (args.minimum_full_sample_profiles),
                    "minimum_median_quality": args.minimum_median_quality,
                },
                "quality_report": quality_report,
            },
        )
        _print_json(
            {
                "accepted_leagues": len(accepted),
                "excluded_leagues": excluded,
                "profiles": len(profiles),
                "unique_players": len({profile.player_id for profile in profiles}),
                "metric_evidence": len(evidence),
                "player_identities": len(identities_by_player),
                "team_stints": len(stints),
                "match_performances": len(match_performances),
                "recent_form_snapshots": len(recent_forms),
                "output_directory": str(args.output),
                "output_files": [str(path) for path in output_files],
                "quality_report": quality_report,
            }
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


def _api_football_history_build(args: argparse.Namespace) -> int:
    """Join accepted season datasets while keeping every season observation separate."""

    try:
        input_roots = [path.resolve() for path in args.input]
        if len(input_roots) < 2:
            raise ValueError("history build requires at least two season datasets")

        profiles_by_id = {}
        evidence_by_key = {}
        identities_by_player: dict[str, PlayerIdentity] = {}
        stints_by_id: dict[str, PlayerTeamSeasonStint] = {}
        matches_by_id: dict[str, PlayerMatchPerformance] = {}
        forms_by_profile: dict[str, PlayerRecentForm] = {}
        definitions: list[MetricDefinition] | None = None
        season_years: list[int] = []

        for root in input_roots:
            required = (
                root / "player_season_profiles.parquet",
                root / "player_metric_evidence.parquet",
                root / "metric_definitions.json",
                root / "manifest.json",
            )
            if not all(path.exists() for path in required):
                raise FileNotFoundError(f"incomplete season dataset: {root}")
            season_profiles = load_profiles(required[0])
            for profile in season_profiles:
                if profile.profile_id is None:
                    raise ValueError(f"history profile has no profile_id: {profile.player_id}")
                profiles_by_id[profile.profile_id] = profile
            for item in load_metric_evidence(required[1]):
                key = (
                    item.profile_id or item.player_id,
                    item.metric_name,
                    item.source_reference,
                )
                evidence_by_key[key] = item
            season_definitions = [
                MetricDefinition.model_validate(item)
                for item in json.loads(required[2].read_text(encoding="utf-8"))
            ]
            if definitions is None:
                definitions = season_definitions
            elif definitions != season_definitions:
                raise ValueError(f"metric definitions differ in history dataset {root}")
            manifest = json.loads(required[3].read_text(encoding="utf-8"))
            source = manifest.get("source", {})
            season_year = source.get("season_start_year")
            if not isinstance(season_year, int):
                raise ValueError(f"history dataset has no season_start_year: {root}")
            season_years.append(season_year)

            for row in _typed_parquet_rows(root / "player_identities.parquet"):
                identity = PlayerIdentity.model_validate(row)
                current = identities_by_player.get(identity.player_id)
                if current is None or _identity_quality(identity) > _identity_quality(current):
                    identities_by_player[identity.player_id] = identity
            for row in _typed_parquet_rows(root / "player_team_season_stints.parquet"):
                stint = PlayerTeamSeasonStint.model_validate(row)
                stints_by_id[stint.stint_id] = stint
            for row in _typed_parquet_rows(root / "player_match_performances.parquet"):
                performance = PlayerMatchPerformance.model_validate(row)
                matches_by_id[performance.performance_id] = performance
            for row in _typed_parquet_rows(root / "player_recent_form.parquet"):
                form = PlayerRecentForm.model_validate(row)
                forms_by_profile[form.profile_id] = form

        if definitions is None:
            raise ValueError("history build found no metric definitions")
        profiles = list(profiles_by_id.values())
        trends: list[PlayerSeasonTrend] = build_season_trends(profiles)
        result = ApiFootballProfileResult(
            profiles=sorted(
                profiles,
                key=lambda item: (
                    item.player_id,
                    -int(item.season_name[:4]),
                    item.profile_id or "",
                ),
            ),
            evidence=sorted(
                evidence_by_key.values(),
                key=lambda item: (
                    item.profile_id or "",
                    item.metric_name,
                    item.source_reference,
                ),
            ),
            definitions=definitions,
            limitations=[
                "Current-season profiles remain separate from historical observations.",
                "Historical values are context and fallback evidence, never an automatic average.",
                "Trend directions are descriptive and are not performance forecasts.",
            ],
            identities=sorted(
                identities_by_player.values(),
                key=lambda item: item.player_id,
            ),
            stints=sorted(
                stints_by_id.values(),
                key=lambda item: (
                    item.player_id,
                    -int(item.season_name[:4]),
                    item.stint_id,
                ),
            ),
            match_performances=sorted(
                matches_by_id.values(),
                key=lambda item: (
                    item.player_id,
                    item.match_date or date.min,
                    item.fixture_id,
                ),
            ),
            recent_forms=sorted(
                forms_by_profile.values(),
                key=lambda item: (item.player_id, item.profile_id),
            ),
            season_trends=sorted(
                trends,
                key=lambda item: (
                    item.player_id,
                    item.current_profile_id,
                    item.metric_name,
                ),
            ),
        )
        output_files = ApiFootballDatasetWriter().write(
            args.output,
            result=result,
            league_id=[],
            season_start_year=max(season_years),
            competition_name="European Scouting History",
            schema_version="api-football-history-v1",
            source_endpoint="local-quality-gated-season-artifacts",
            source_details={
                "season_start_years": sorted(set(season_years), reverse=True),
                "input_datasets": [str(path) for path in input_roots],
                "season_aggregation": "none",
                "current_season_priority": max(season_years),
            },
        )
        _print_json(
            {
                "seasons": sorted(set(season_years), reverse=True),
                "profiles": len(result.profiles),
                "unique_players": len({item.player_id for item in result.profiles}),
                "metric_evidence": len(result.evidence),
                "team_stints": len(result.stints),
                "match_performances": len(result.match_performances),
                "recent_form_snapshots": len(result.recent_forms),
                "season_trends": len(result.season_trends),
                "output_directory": str(args.output),
                "output_files": [str(path) for path in output_files],
            }
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


def _api_football_key() -> str | None:
    secret = Settings().api_football_key
    if secret is None:
        return None
    value = secret.get_secret_value().strip()
    return value or None


def _identity_quality(identity: PlayerIdentity) -> int:
    return sum(
        value is not None
        for value in (
            identity.date_of_birth,
            identity.birth_place,
            identity.birth_country,
            identity.nationality,
            identity.height_cm,
            identity.weight_kg,
            identity.photo_url,
        )
    )


def _typed_parquet_rows(path: Path) -> list[dict[str, Any]]:
    """Read new JSON mappings and repair legacy sparse Arrow mappings."""

    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for stored_row in pq.read_table(path).to_pylist():
        row = dict(stored_row)
        for field_name, value in tuple(row.items()):
            if field_name.endswith("_json") and isinstance(value, str):
                row[field_name.removesuffix("_json")] = json.loads(value)
                row.pop(field_name)
            elif isinstance(value, dict):
                row[field_name] = {
                    key: nested_value
                    for key, nested_value in value.items()
                    if nested_value is not None
                }
        rows.append(row)
    return rows


def _missing_api_football_key() -> int:
    print(
        "error: API_FOOTBALL_KEY is not configured; add it to the local .env file",
        file=sys.stderr,
    )
    return 2


def _report_api_error(exc: Exception, api_key: str) -> int:
    safe_message = str(exc).replace(api_key, "[REDACTED]")
    print(f"error: {safe_message}", file=sys.stderr)
    return 2


def _validate_api_football_sync_arguments(args: argparse.Namespace) -> None:
    if args.league_id < 1:
        raise ValueError("--league-id must be positive")
    if args.season < 1900:
        raise ValueError("--season must be a four-digit start year")
    if args.team_id < 1:
        raise ValueError("--team-id must be positive")
    if args.request_budget < 1:
        raise ValueError("--request-budget must be at least one")
    if args.max_pages < 1:
        raise ValueError("--max-pages must be at least one")
    if not math.isfinite(args.throttle_seconds) or args.throttle_seconds < 0:
        raise ValueError("--throttle-seconds must be a finite nonnegative number")
    if args.all_teams:
        if args.max_pages < MINIMUM_FULL_LEAGUE_MAX_PAGES:
            raise ValueError(
                f"full-league sync requires --max-pages of at least {MINIMUM_FULL_LEAGUE_MAX_PAGES}"
            )
        if args.request_budget < args.max_pages + 1:
            raise ValueError(
                "full-league sync requires --request-budget of at least "
                "--max-pages + 1 for the coverage check"
            )


def _validate_api_football_fixture_sync_arguments(
    args: argparse.Namespace,
) -> None:
    if args.league_id < 1:
        raise ValueError("--league-id must be positive")
    if args.season < 1900:
        raise ValueError("--season must be a four-digit start year")
    if args.request_budget < 1:
        raise ValueError("--request-budget must be at least one")
    if args.max_fixture_pages < 1:
        raise ValueError("--max-fixture-pages must be at least one")
    if args.max_player_pages < 1:
        raise ValueError("--max-player-pages must be at least one")
    if args.batch_size < 1 or args.batch_size > 20:
        raise ValueError("--batch-size must be between 1 and 20")
    if not math.isfinite(args.throttle_seconds) or args.throttle_seconds < 0:
        raise ValueError("--throttle-seconds must be a finite nonnegative number")


def _has_player_coverage(
    response: ApiFootballResponse,
    *,
    league_id: int,
    season_start_year: int,
) -> bool:
    if not isinstance(response.response, list):
        raise ApiFootballProtocolError("API-Football /leagues response must be an array")
    for item in response.response:
        if not isinstance(item, dict):
            continue
        league = item.get("league")
        if not isinstance(league, dict) or league.get("id") != league_id:
            continue
        seasons = item.get("seasons")
        if not isinstance(seasons, list):
            continue
        for season in seasons:
            if not isinstance(season, dict) or season.get("year") != season_start_year:
                continue
            coverage = season.get("coverage")
            if isinstance(coverage, dict) and coverage.get("players") is True:
                return True
    return False


def _comparison_scope(args: argparse.Namespace) -> str | None:
    if args.all_teams:
        return None
    if args.team_id == DEFAULT_API_FOOTBALL_TEAM_ID:
        return "FC Bayern München team-filtered sample"
    return f"API-Football team-filtered sample (team_id={args.team_id})"


def _status_summary(response: ApiFootballResponse) -> dict[str, Any]:
    if not isinstance(response.response, dict):
        raise ApiFootballProtocolError("API-Football /status response must be an object")
    subscription = response.response.get("subscription")
    requests = response.response.get("requests")
    return {
        "provider": "API-Football",
        "connected": True,
        "subscription": _selected_fields(subscription, ("plan", "end", "active")),
        "requests": _selected_fields(requests, ("current", "limit")),
        "quota": asdict(response.quota),
    }


def _selected_fields(value: object, names: tuple[str, ...]) -> dict[str, object]:
    if not isinstance(value, dict):
        return {}
    return {name: value[name] for name in names if name in value}


def _print_json(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))
