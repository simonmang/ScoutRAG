"""Run committed Phase 7 governance and abstention cases."""

import argparse
from collections.abc import Sequence
from pathlib import Path

from scoutrag.evaluation.governance import GovernanceEvaluator, load_governance_dataset
from scoutrag.governance.cli import (
    DEFAULT_DENSE_INDEX_PATH,
    DEFAULT_EVIDENCE_PATH,
    DEFAULT_PROFILES_PATH,
)
from scoutrag.governance.evidence import load_metric_evidence
from scoutrag.governance.factory import build_governed_pipeline
from scoutrag.retrieval.common import load_profiles
from scoutrag.retrieval.dense import (
    DEFAULT_MODEL_NAME,
    DensePlayerRetriever,
    SentenceTransformerEmbeddingModel,
)

DEFAULT_GOVERNANCE_CASES_PATH = Path("evaluation/governance_cases.json")
DEFAULT_OUTPUT_PATH = Path("evaluation/results/phase7_governance.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scoutrag-govern-evaluate",
        description="Evaluate false recommendations, abstention, and limited verdicts.",
    )
    parser.add_argument("--cases", type=Path, default=DEFAULT_GOVERNANCE_CASES_PATH)
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES_PATH)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE_PATH)
    parser.add_argument("--dense-index", type=Path, default=DEFAULT_DENSE_INDEX_PATH)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--candidate-pool-size", type=int, default=40)
    parser.add_argument("--disable-dense", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset = load_governance_dataset(args.cases)
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
    pipeline = build_governed_pipeline(
        profiles,
        evidence,
        dense_retriever=dense,
        candidate_pool_size=args.candidate_pool_size,
    )
    report = GovernanceEvaluator().evaluate(pipeline.search, dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    print(report.model_dump_json(indent=2, ensure_ascii=True))
    return 0
