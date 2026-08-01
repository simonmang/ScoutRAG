"""Command-line hybrid retrieval over generated Phase 3 profiles."""

import argparse
from collections.abc import Sequence
from pathlib import Path

from scoutrag.ports.retrieval import PlayerRetriever
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
from scoutrag.retrieval.fusion import FusionWeights, WeightedRetrievalFusion
from scoutrag.retrieval.pipeline import HybridRetrievalPipeline
from scoutrag.retrieval.query_analysis import RuleBasedQueryAnalyzer
from scoutrag.retrieval.sparse import BM25PlayerRetriever
from scoutrag.retrieval.structured import StructuredFeaturePlayerRetriever

DEFAULT_PROFILES_PATH = Path(
    "data/processed/scouting-2025-2026/combined/player_season_profiles.parquet"
)
DEFAULT_DENSE_INDEX_PATH = Path("data/processed/scouting-2025-2026/combined/dense_index.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scoutrag-retrieve",
        description="Run auditable hybrid player retrieval without an LLM.",
    )
    parser.add_argument("query")
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES_PATH)
    parser.add_argument("--candidate-pool-size", type=int, default=40)
    parser.add_argument("--disable-dense", action="store_true")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--dense-index", type=Path, default=DEFAULT_DENSE_INDEX_PATH)
    parser.add_argument("--rebuild-dense-index", action="store_true")
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--cross-encoder-model", default=DEFAULT_CROSS_ENCODER_MODEL)
    parser.add_argument("--reranker-batch-size", type=int, default=16)
    parser.add_argument("--reranker-backend", choices=("torch", "onnx"), default="torch")
    parser.add_argument("--onnx-file-name", default="onnx/model.onnx")
    parser.add_argument("--device")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--dense-weight", type=float, default=0.30)
    parser.add_argument("--sparse-weight", type=float, default=0.25)
    parser.add_argument("--structured-weight", type=float, default=0.30)
    parser.add_argument("--exact-weight", type=float, default=0.15)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    profiles = load_profiles(args.profiles)
    retrievers: list[PlayerRetriever] = [
        ExactPlayerRetriever(profiles),
        StructuredFeaturePlayerRetriever(profiles),
        BM25PlayerRetriever(profiles),
    ]
    weights = FusionWeights(
        dense=args.dense_weight,
        sparse=args.sparse_weight,
        structured=args.structured_weight,
        exact=args.exact_weight,
    )
    if not args.disable_dense:
        retrievers.append(
            DensePlayerRetriever(
                profiles,
                SentenceTransformerEmbeddingModel(
                    args.model_name,
                    local_files_only=args.local_files_only,
                ),
                index_path=args.dense_index,
                rebuild_index=args.rebuild_dense_index,
            )
        )
    elif weights.dense:
        weights = _without_dense(weights)

    player_reranker = None
    if args.rerank:
        player_reranker = CrossEncoderPlayerReranker(
            SentenceTransformerCrossEncoderModel(
                args.cross_encoder_model,
                batch_size=args.reranker_batch_size,
                local_files_only=args.local_files_only,
                backend=args.reranker_backend,
                onnx_file_name=args.onnx_file_name,
                device=args.device,
            )
        )
    pipeline = HybridRetrievalPipeline(
        RuleBasedQueryAnalyzer(profiles),
        tuple(retrievers),
        WeightedRetrievalFusion(weights),
        player_reranker=player_reranker,
        candidate_pool_size=args.candidate_pool_size,
    )
    result = pipeline.search(args.query)
    # ASCII escaping keeps JSON portable on legacy Windows consoles using cp1252.
    print(result.model_dump_json(indent=2, ensure_ascii=True))
    return 0


def _without_dense(weights: FusionWeights) -> FusionWeights:
    available = weights.sparse + weights.structured + weights.exact
    if not available:
        raise ValueError("at least one enabled retrieval strategy needs a positive weight")
    return FusionWeights(
        dense=0,
        sparse=weights.sparse / available,
        structured=weights.structured / available,
        exact=weights.exact / available,
    )
