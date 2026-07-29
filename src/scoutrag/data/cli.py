"""Command-line entry point for reproducible Phase 3 builds."""

import argparse
from collections.abc import Sequence
from pathlib import Path

from scoutrag.data.feature_engineering import FeatureEngineeringConfig
from scoutrag.data.models import DownloadSummary, PipelineResult
from scoutrag.data.pipeline import Phase3DataPipeline
from scoutrag.data.statsbomb import StatsBombOpenDataDownloader

DEFAULT_COMPETITION_ID = 9
DEFAULT_SEASON_ID = 281
DEFAULT_RAW_ROOT = Path("data/raw/statsbomb")
DEFAULT_PROCESSED_ROOT = Path("data/processed/bundesliga-2023-2024")


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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
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
