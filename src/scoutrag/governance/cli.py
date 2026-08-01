"""Produce an LLM-free RecommendationEvidencePack from local artifacts."""

import argparse
from collections.abc import Sequence
from pathlib import Path

from scoutrag.governance.evidence import load_metric_evidence
from scoutrag.governance.factory import build_governed_pipeline
from scoutrag.retrieval.common import load_profiles
from scoutrag.retrieval.dense import (
    DEFAULT_MODEL_NAME,
    DensePlayerRetriever,
    SentenceTransformerEmbeddingModel,
)

DEFAULT_PROFILES_PATH = Path(
    "data/processed/scouting-2025-2026/combined/player_season_profiles.parquet"
)
DEFAULT_EVIDENCE_PATH = Path(
    "data/processed/scouting-2025-2026/combined/player_metric_evidence.parquet"
)
DEFAULT_DENSE_INDEX_PATH = Path("data/processed/scouting-2025-2026/combined/dense_index.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scoutrag-govern",
        description="Retrieve players and return a governed evidence pack without an LLM.",
    )
    parser.add_argument("query")
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES_PATH)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE_PATH)
    parser.add_argument("--dense-index", type=Path, default=DEFAULT_DENSE_INDEX_PATH)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--candidate-pool-size", type=int, default=40)
    parser.add_argument("--disable-dense", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    profiles = load_profiles(args.profiles)
    evidence = load_metric_evidence(args.evidence)
    dense = None
    if not args.disable_dense:
        dense = DensePlayerRetriever(
            profiles,
            SentenceTransformerEmbeddingModel(
                args.model_name,
                local_files_only=args.local_files_only,
            ),
            index_path=args.dense_index,
        )
    pack = build_governed_pipeline(
        profiles,
        evidence,
        dense_retriever=dense,
        candidate_pool_size=args.candidate_pool_size,
    ).search(args.query)
    print(pack.model_dump_json(indent=2, ensure_ascii=True))
    return 0
