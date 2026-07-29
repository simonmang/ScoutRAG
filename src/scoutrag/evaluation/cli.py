"""Run Phase 5 retrieval ablations from the command line."""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from scoutrag.evaluation.dataset import load_golden_dataset
from scoutrag.evaluation.runner import AblationRunner
from scoutrag.retrieval.common import load_profiles
from scoutrag.retrieval.dense import (
    DEFAULT_MODEL_NAME,
    DensePlayerRetriever,
    SentenceTransformerEmbeddingModel,
)

DEFAULT_GOLDEN_PATH = Path("evaluation/golden_queries.json")
DEFAULT_PROFILES_PATH = Path("data/processed/bundesliga-2023-2024/player_season_profiles.parquet")
DEFAULT_DENSE_INDEX_PATH = Path("data/processed/bundesliga-2023-2024/dense_index.json")
DEFAULT_OUTPUT_PATH = Path("evaluation/results/phase5_ablation.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scoutrag-evaluate",
        description="Evaluate ScoutRAG retrieval variants against the golden dataset.",
    )
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN_PATH)
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES_PATH)
    parser.add_argument("--dense-index", type=Path, default=DEFAULT_DENSE_INDEX_PATH)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--candidate-pool-size", type=int, default=40)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--rebuild-dense-index", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset = load_golden_dataset(args.golden)
    profiles = load_profiles(args.profiles)
    dense = DensePlayerRetriever(
        profiles,
        SentenceTransformerEmbeddingModel(
            args.model_name,
            local_files_only=args.local_files_only,
        ),
        index_path=args.dense_index,
        rebuild_index=args.rebuild_dense_index,
    )
    report = AblationRunner(
        profiles,
        dense,
        candidate_pool_size=args.candidate_pool_size,
    ).run(dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    if args.summary_only:
        print(
            json.dumps(
                {
                    item.variant_name: item.aggregate.model_dump(mode="json")
                    for item in report.reports
                },
                indent=2,
            )
        )
    else:
        print(report.model_dump_json(indent=2))
    return 0
