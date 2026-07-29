"""Reproducible hallucination and groundedness evaluation."""

import json
from pathlib import Path

from pydantic import Field, model_validator

from scoutrag.answering.generator import GroundedAnswerGenerator
from scoutrag.answering.models import GroundedAnswerDraft
from scoutrag.domain.base import ScoutRAGModel
from scoutrag.domain.evidence import (
    EvidenceVerdict,
    GenerationMode,
    RecommendationEvidencePack,
    RecommendationGovernance,
)


class AnswerGroundingCase(ScoutRAGModel):
    """One safe-generation, hallucination, or abstention regression case."""

    case_id: str = Field(min_length=1)
    category: str = Field(min_length=1)
    verdict: EvidenceVerdict
    draft: GroundedAnswerDraft
    evaluates_draft: bool
    expected_grounded: bool
    expect_backend_call: bool
    rationale: str = Field(min_length=1)


class AnswerGroundingDataset(ScoutRAGModel):
    """Versioned fixture and adversarial structured outputs."""

    schema_version: str = Field(min_length=1)
    evidence_pack: RecommendationEvidencePack
    cases: list[AnswerGroundingCase] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_case_ids(self) -> "AnswerGroundingDataset":
        case_ids = [case.case_id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("answer grounding case IDs must be unique")
        return self


class AnswerGroundingCaseResult(ScoutRAGModel):
    """Expected and observed safety behavior for one generated draft."""

    case_id: str
    category: str
    expected_grounded: bool
    backend_called: bool
    generation_mode: GenerationMode
    validation_passed: bool
    fallback_used: bool
    violations: list[str] = Field(default_factory=list)
    passed: bool


class AnswerGroundingMetrics(ScoutRAGModel):
    """Safety metrics with false acceptance as the primary failure signal."""

    groundedness_pass_rate: float = Field(ge=0, le=1)
    hallucination_block_rate: float = Field(ge=0, le=1)
    false_grounded_rate: float = Field(ge=0, le=1)
    fallback_precision: float = Field(ge=0, le=1)
    fallback_recall: float = Field(ge=0, le=1)
    safe_answer_coverage: float = Field(ge=0, le=1)
    abstention_compliance: float = Field(ge=0, le=1)
    case_accuracy: float = Field(ge=0, le=1)


class AnswerGroundingReport(ScoutRAGModel):
    """Complete Phase 10 answer-safety report."""

    dataset_version: str
    case_count: int = Field(ge=1)
    metrics: AnswerGroundingMetrics
    cases: list[AnswerGroundingCaseResult] = Field(min_length=1)


class _CaseBackend:
    def __init__(self, draft: GroundedAnswerDraft) -> None:
        self._draft = draft
        self.called = False

    @property
    def model_name(self) -> str:
        return "evaluation-fixture"

    def generate_draft(self, *, instructions: str, input_text: str) -> GroundedAnswerDraft:
        del instructions, input_text
        self.called = True
        return self._draft


class AnswerGroundingEvaluator:
    """Run committed model-output fixtures through the production safety boundary."""

    def evaluate(self, dataset: AnswerGroundingDataset) -> AnswerGroundingReport:
        results: list[AnswerGroundingCaseResult] = []
        for case in dataset.cases:
            pack = _pack_with_verdict(dataset.evidence_pack, case.verdict)
            backend = _CaseBackend(case.draft)
            answer = GroundedAnswerGenerator(backend).generate(pack)
            accepted = answer.generation_mode is GenerationMode.GROUNDED_MODEL
            passed = backend.called == case.expect_backend_call and (
                accepted == case.expected_grounded
                if case.evaluates_draft
                else answer.generation_mode is GenerationMode.TEMPLATE
            )
            results.append(
                AnswerGroundingCaseResult(
                    case_id=case.case_id,
                    category=case.category,
                    expected_grounded=case.expected_grounded,
                    backend_called=backend.called,
                    generation_mode=answer.generation_mode,
                    validation_passed=answer.grounding.validation_passed,
                    fallback_used=answer.grounding.fallback_used,
                    violations=answer.grounding.violations,
                    passed=passed,
                )
            )
        return AnswerGroundingReport(
            dataset_version=dataset.schema_version,
            case_count=len(results),
            metrics=_metrics(dataset.cases, results),
            cases=results,
        )


def load_answer_grounding_dataset(path: Path) -> AnswerGroundingDataset:
    """Load and strictly validate the committed answer-safety dataset."""
    return AnswerGroundingDataset.model_validate(json.loads(path.read_text(encoding="utf-8")))


def _pack_with_verdict(
    pack: RecommendationEvidencePack,
    verdict: EvidenceVerdict,
) -> RecommendationEvidencePack:
    governance_data = pack.governance.model_dump()
    governance_data["verdict"] = verdict
    if verdict is EvidenceVerdict.SUFFICIENT:
        governance_data["evidence_quality_score"] = 0.9
    governance = RecommendationGovernance.model_validate(governance_data)
    return pack.model_copy(update={"governance": governance})


def _metrics(
    cases: list[AnswerGroundingCase],
    results: list[AnswerGroundingCaseResult],
) -> AnswerGroundingMetrics:
    paired = list(zip(cases, results, strict=True))
    safe = [
        (case, result) for case, result in paired if case.evaluates_draft and case.expected_grounded
    ]
    unsafe = [
        (case, result)
        for case, result in paired
        if case.evaluates_draft and not case.expected_grounded
    ]
    fallbacks = [
        (case, result) for case, result in paired if case.evaluates_draft and result.fallback_used
    ]
    abstentions = [(case, result) for case, result in paired if not case.expect_backend_call]
    accepted_safe = sum(
        result.generation_mode is GenerationMode.GROUNDED_MODEL for _, result in safe
    )
    blocked_unsafe = sum(result.fallback_used for _, result in unsafe)
    false_grounded = sum(
        result.generation_mode is GenerationMode.GROUNDED_MODEL for _, result in unsafe
    )
    correct_fallbacks = sum(not case.expected_grounded for case, _ in fallbacks)
    compliant_abstentions = sum(
        not result.backend_called and result.generation_mode is GenerationMode.TEMPLATE
        for _, result in abstentions
    )
    return AnswerGroundingMetrics(
        groundedness_pass_rate=_ratio(accepted_safe, len(safe)),
        hallucination_block_rate=_ratio(blocked_unsafe, len(unsafe)),
        false_grounded_rate=_ratio(false_grounded, len(unsafe)),
        fallback_precision=_ratio(correct_fallbacks, len(fallbacks)),
        fallback_recall=_ratio(blocked_unsafe, len(unsafe)),
        safe_answer_coverage=_ratio(accepted_safe, len(safe)),
        abstention_compliance=_ratio(compliant_abstentions, len(abstentions)),
        case_accuracy=_ratio(sum(result.passed for result in results), len(results)),
    )


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0
