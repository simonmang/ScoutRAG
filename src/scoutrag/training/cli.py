"""Mine, fine-tune, and evaluate the football bi-encoder."""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from scoutrag.evaluation.dataset import load_golden_dataset
from scoutrag.retrieval.common import load_profiles
from scoutrag.retrieval.dense import (
    DEFAULT_MODEL_NAME,
    SentenceTransformerEmbeddingModel,
)
from scoutrag.training.dataset import load_mined_dataset, load_training_dataset
from scoutrag.training.evaluation import BiEncoderEvaluator
from scoutrag.training.mining import FootballHardNegativeMiner
from scoutrag.training.trainer import (
    BiEncoderTrainingConfig,
    SentenceTransformerBiEncoderTrainer,
)

DEFAULT_SPECS = Path("evaluation/bi_encoder_training_queries.json")
DEFAULT_GOLDEN = Path("evaluation/golden_queries.json")
DEFAULT_PROFILES = Path("data/processed/bundesliga-2023-2024/player_season_profiles.parquet")
DEFAULT_MINED = Path("evaluation/results/phase9_mined_triplets.json")
DEFAULT_MODEL_OUTPUT = Path("models/scoutrag-football-bi-encoder-v1")
DEFAULT_REPORT = Path("evaluation/results/phase9_bi_encoder_comparison.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scoutrag-bi-encoder",
        description="Mine football hard negatives, fine-tune, and evaluate the bi-encoder.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    mine = subparsers.add_parser("mine", help="Resolve typed positive/hard/easy tuples.")
    _add_data_arguments(mine)
    mine.add_argument("--specs", type=Path, default=DEFAULT_SPECS)
    mine.add_argument("--mined-output", type=Path, default=DEFAULT_MINED)
    _add_base_model_arguments(mine)

    train = subparsers.add_parser("train", help="Fine-tune from mined tuples.")
    train.add_argument("--mined", type=Path, default=DEFAULT_MINED)
    train.add_argument("--model-output", type=Path, default=DEFAULT_MODEL_OUTPUT)
    _add_training_arguments(train)
    _add_base_model_arguments(train)

    evaluate = subparsers.add_parser("evaluate", help="Compare baseline and fine-tuned models.")
    _add_data_arguments(evaluate)
    evaluate.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    evaluate.add_argument("--mined", type=Path, default=DEFAULT_MINED)
    evaluate.add_argument("--fine-tuned-model", type=Path, default=DEFAULT_MODEL_OUTPUT)
    evaluate.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    evaluate.add_argument("--candidate-pool-size", type=int, default=40)
    _add_base_model_arguments(evaluate)

    all_steps = subparsers.add_parser("all", help="Run mining, training, and comparison.")
    _add_data_arguments(all_steps)
    all_steps.add_argument("--specs", type=Path, default=DEFAULT_SPECS)
    all_steps.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    all_steps.add_argument("--mined-output", type=Path, default=DEFAULT_MINED)
    all_steps.add_argument("--model-output", type=Path, default=DEFAULT_MODEL_OUTPUT)
    all_steps.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    all_steps.add_argument("--candidate-pool-size", type=int, default=40)
    _add_training_arguments(all_steps)
    _add_base_model_arguments(all_steps)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "mine":
        mined = _mine(args)
        _write_model(args.mined_output, mined)
        _print_mining_summary(mined)
    elif args.command == "train":
        summary = _train(args, load_mined_dataset(args.mined))
        print(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2))
    elif args.command == "evaluate":
        report = _evaluate(args, load_mined_dataset(args.mined))
        _write_model(args.report, report)
        _print_evaluation_summary(report)
    else:
        mined = _mine(args)
        _write_model(args.mined_output, mined)
        _train(args, mined)
        report = _evaluate(args, mined)
        _write_model(args.report, report)
        _print_evaluation_summary(report)
    return 0


def _mine(args: argparse.Namespace) -> Any:
    profiles = load_profiles(args.profiles)
    specs = load_training_dataset(args.specs)
    model = SentenceTransformerEmbeddingModel(
        args.base_model,
        local_files_only=args.local_files_only,
    )
    return FootballHardNegativeMiner(profiles, model).mine(specs)


def _train(args: argparse.Namespace, mined: Any) -> Any:
    config = BiEncoderTrainingConfig(
        base_model_name=args.base_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        max_steps=args.max_steps,
        seed=args.seed,
        use_cpu=not args.allow_accelerator,
        local_files_only=args.local_files_only,
    )
    return SentenceTransformerBiEncoderTrainer().train(mined, args.model_output, config)


def _evaluate(args: argparse.Namespace, mined: Any) -> Any:
    profiles = load_profiles(args.profiles)
    evaluator = BiEncoderEvaluator(
        profiles,
        load_golden_dataset(args.golden),
        mined,
        candidate_pool_size=args.candidate_pool_size,
    )
    return evaluator.compare(
        SentenceTransformerEmbeddingModel(
            args.base_model,
            local_files_only=args.local_files_only,
        ),
        SentenceTransformerEmbeddingModel(
            str(args.fine_tuned_model if hasattr(args, "fine_tuned_model") else args.model_output),
            local_files_only=True,
        ),
    )


def _add_data_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES)


def _add_base_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base-model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--local-files-only", action="store_true")


def _add_training_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-accelerator", action="store_true")


def _write_model(path: Path, model: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(model.model_dump_json(indent=2), encoding="utf-8")


def _print_mining_summary(mined: Any) -> None:
    print(
        json.dumps(
            {
                "schema_version": mined.schema_version,
                "embedding_model": mined.embedding_model,
                "train_examples": sum(item.split == "train" for item in mined.examples),
                "validation_examples": sum(item.split == "validation" for item in mined.examples),
            },
            indent=2,
        )
    )


def _print_evaluation_summary(report: Any) -> None:
    print(
        json.dumps(
            {
                "baseline": _summary_metrics(report.baseline),
                "fine_tuned": _summary_metrics(report.fine_tuned),
                "delta": report.delta.model_dump(mode="json"),
            },
            indent=2,
        )
    )


def _summary_metrics(evaluation: Any) -> dict[str, Any]:
    aggregate = evaluation.golden_retrieval.aggregate
    return {
        "candidate_recall": aggregate.candidate_recall,
        "mrr": aggregate.mean_reciprocal_rank,
        "ndcg_at_5": aggregate.at_k[5].ndcg,
        "hard_negative_accuracy": evaluation.pairwise.hard_negative_accuracy,
        "language_accuracy": evaluation.pairwise.language_accuracy,
        "bilingual_pair_stability": evaluation.pairwise.bilingual_pair_stability,
    }


def _jsonable(value: Any) -> dict[str, Any]:
    from dataclasses import asdict

    return asdict(value)
