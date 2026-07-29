"""Regression tests for the committed Phase 10 safety benchmark."""

from pathlib import Path

from scoutrag.evaluation.answer_grounding import (
    AnswerGroundingEvaluator,
    load_answer_grounding_dataset,
)


def test_answer_grounding_dataset_blocks_every_adversarial_case() -> None:
    dataset = load_answer_grounding_dataset(
        Path("evaluation/answer_grounding_cases.json")
    )

    report = AnswerGroundingEvaluator().evaluate(dataset)

    assert report.case_count == 10
    assert report.metrics.groundedness_pass_rate == 1
    assert report.metrics.hallucination_block_rate == 1
    assert report.metrics.false_grounded_rate == 0
    assert report.metrics.fallback_precision == 1
    assert report.metrics.fallback_recall == 1
    assert report.metrics.safe_answer_coverage == 1
    assert report.metrics.abstention_compliance == 1
    assert report.metrics.case_accuracy == 1
    assert all(case.passed for case in report.cases)
