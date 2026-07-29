"""CLI for the committed Phase 10 answer-grounding benchmark."""

import argparse
from collections.abc import Sequence
from pathlib import Path

from scoutrag.evaluation.answer_grounding import (
    AnswerGroundingEvaluator,
    load_answer_grounding_dataset,
)

DEFAULT_CASES_PATH = Path("evaluation/answer_grounding_cases.json")
DEFAULT_OUTPUT_PATH = Path("evaluation/results/phase10_answer_grounding.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scoutrag-answer-evaluate",
        description="Measure groundedness, hallucination blocking, and safe abstention.",
    )
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset = load_answer_grounding_dataset(args.cases)
    report = AnswerGroundingEvaluator().evaluate(dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    print(report.model_dump_json(indent=2))
    return 0
