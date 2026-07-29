"""Run Phase 6 cross-encoder reranking evaluation from the command line."""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from scoutrag.evaluation.dataset import load_golden_dataset
from scoutrag.evaluation.reranking import RerankingEvaluator
from scoutrag.reranking.cross_encoder import (
    DEFAULT_CROSS_ENCODER_MODEL,
    CrossEncoderPlayerReranker,
    SentenceTransformerCrossEncoderModel,
)
from scoutrag.retrieval.common import load_profiles
from scoutrag.retrieval.dense import (
    DEFAULT_MODEL_NAME,
    DensePlayerRetriever,
    SentenceTransformerEmbeddingModel,
)
from scoutrag.retrieval.exact import ExactPlayerRetriever
from scoutrag.retrieval.fusion import WeightedRetrievalFusion
from scoutrag.retrieval.pipeline import HybridRetrievalPipeline
from scoutrag.retrieval.query_analysis import RuleBasedQueryAnalyzer
from scoutrag.retrieval.sparse import BM25PlayerRetriever
from scoutrag.retrieval.structured import StructuredFeaturePlayerRetriever

DEFAULT_GOLDEN_PATH = Path("evaluation/golden_queries.json")
DEFAULT_PROFILES_PATH = Path("data/processed/bundesliga-2023-2024/player_season_profiles.parquet")
DEFAULT_DENSE_INDEX_PATH = Path("data/processed/bundesliga-2023-2024/dense_index.json")
DEFAULT_OUTPUT_PATH = Path("evaluation/results/phase6_reranking.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scoutrag-rerank-evaluate",
        description="Compare fused order with cross-encoder order over identical candidates.",
    )
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN_PATH)
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES_PATH)
    parser.add_argument("--dense-index", type=Path, default=DEFAULT_DENSE_INDEX_PATH)
    parser.add_argument("--bi-encoder-model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--cross-encoder-model", default=DEFAULT_CROSS_ENCODER_MODEL)
    parser.add_argument("--candidate-pool-size", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--backend", choices=("torch", "onnx"), default="torch")
    parser.add_argument("--onnx-file-name", default="onnx/model.onnx")
    parser.add_argument("--device")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--skip-warmup", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset = load_golden_dataset(args.golden)
    profiles = load_profiles(args.profiles)
    dense = DensePlayerRetriever(
        profiles,
        SentenceTransformerEmbeddingModel(
            args.bi_encoder_model,
            local_files_only=args.local_files_only,
        ),
        index_path=args.dense_index,
    )
    pipeline = HybridRetrievalPipeline(
        RuleBasedQueryAnalyzer(profiles),
        (
            ExactPlayerRetriever(profiles),
            StructuredFeaturePlayerRetriever(profiles),
            BM25PlayerRetriever(profiles),
            dense,
        ),
        WeightedRetrievalFusion(),
        candidate_pool_size=args.candidate_pool_size,
    )
    scoring_model = SentenceTransformerCrossEncoderModel(
        args.cross_encoder_model,
        batch_size=args.batch_size,
        local_files_only=args.local_files_only,
        backend=args.backend,
        onnx_file_name=args.onnx_file_name,
        device=args.device,
    )
    if not args.skip_warmup:
        scoring_model.warmup()
    report = RerankingEvaluator().compare(
        pipeline,
        CrossEncoderPlayerReranker(scoring_model),
        dataset,
        model_name=scoring_model.model_name,
        backend=args.backend,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "model_name": report.model_name,
                "backend": report.backend,
                "baseline": report.baseline.aggregate.model_dump(mode="json"),
                "reranked": report.reranked.aggregate.model_dump(mode="json"),
                "delta": report.delta.model_dump(mode="json"),
                "latency": report.latency.model_dump(mode="json"),
                "report": str(args.output),
            },
            indent=2,
        )
    )
    return 0
