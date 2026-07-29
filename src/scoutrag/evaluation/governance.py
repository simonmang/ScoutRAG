"""Safety-oriented evaluation of governance verdicts and abstention."""

import json
from collections.abc import Callable
from pathlib import Path

from pydantic import Field, model_validator

from scoutrag.domain.base import ScoutRAGModel
from scoutrag.domain.evidence import EvidenceVerdict, RecommendationEvidencePack


class GovernanceGoldenCase(ScoutRAGModel):
    """One expected safety decision for a natural-language query."""

    case_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    expected_verdict: EvidenceVerdict
    should_abstain: bool
    rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_abstention_label(self) -> "GovernanceGoldenCase":
        abstention_verdicts = {
            EvidenceVerdict.INSUFFICIENT,
            EvidenceVerdict.CONFLICTING,
            EvidenceVerdict.OUT_OF_SCOPE,
        }
        if self.should_abstain != (self.expected_verdict in abstention_verdicts):
            raise ValueError("should_abstain must match the expected verdict semantics")
        return self


class GovernanceGoldenDataset(ScoutRAGModel):
    """Versioned governance and abstention regression cases."""

    schema_version: str = Field(min_length=1)
    cases: list[GovernanceGoldenCase] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_case_ids(self) -> "GovernanceGoldenDataset":
        case_ids = [case.case_id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("governance case IDs must be unique")
        return self


class GovernanceCaseResult(ScoutRAGModel):
    """Auditable expected and actual verdict for one case."""

    case_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    expected_verdict: EvidenceVerdict
    actual_verdict: EvidenceVerdict
    should_abstain: bool
    did_abstain: bool
    evidence_quality_score: float = Field(ge=0, le=1)
    reasons: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class GovernanceMetrics(ScoutRAGModel):
    """Macro safety metrics for governance decisions."""

    false_recommendation_rate: float = Field(ge=0, le=1)
    abstention_recall: float = Field(ge=0, le=1)
    abstention_precision: float = Field(ge=0, le=1)
    coverage: float = Field(ge=0, le=1)
    selective_accuracy: float = Field(ge=0, le=1)
    limited_case_recall: float = Field(ge=0, le=1)
    verdict_accuracy: float = Field(ge=0, le=1)


class GovernanceEvaluationReport(ScoutRAGModel):
    """Complete Phase 7 safety evaluation report."""

    dataset_version: str = Field(min_length=1)
    case_count: int = Field(ge=1)
    metrics: GovernanceMetrics
    cases: list[GovernanceCaseResult] = Field(min_length=1)


class GovernanceEvaluator:
    """Evaluate a governed search callable without coupling to one pipeline."""

    def evaluate(
        self,
        search: Callable[[str], RecommendationEvidencePack],
        dataset: GovernanceGoldenDataset,
    ) -> GovernanceEvaluationReport:
        results: list[GovernanceCaseResult] = []
        for case in dataset.cases:
            pack = search(case.query)
            verdict = pack.governance.verdict
            did_abstain = _is_abstention(verdict)
            results.append(
                GovernanceCaseResult(
                    case_id=case.case_id,
                    query=case.query,
                    expected_verdict=case.expected_verdict,
                    actual_verdict=verdict,
                    should_abstain=case.should_abstain,
                    did_abstain=did_abstain,
                    evidence_quality_score=pack.governance.evidence_quality_score,
                    reasons=pack.governance.reasons,
                    missing_evidence=pack.governance.missing_evidence,
                    warnings=pack.governance.warnings,
                )
            )
        return GovernanceEvaluationReport(
            dataset_version=dataset.schema_version,
            case_count=len(results),
            metrics=_metrics(results),
            cases=results,
        )


def load_governance_dataset(path: Path) -> GovernanceGoldenDataset:
    """Validate committed governance cases before running safety evaluation."""
    return GovernanceGoldenDataset.model_validate(json.loads(path.read_text(encoding="utf-8")))


def _metrics(results: list[GovernanceCaseResult]) -> GovernanceMetrics:
    unsafe = [result for result in results if result.should_abstain]
    abstentions = [result for result in results if result.did_abstain]
    true_abstentions = [
        result for result in results if result.should_abstain and result.did_abstain
    ]
    covered = [result for result in results if not result.did_abstain]
    limited = [result for result in results if result.expected_verdict is EvidenceVerdict.LIMITED]
    false_recommendations = [
        result for result in unsafe if result.actual_verdict is EvidenceVerdict.SUFFICIENT
    ]
    return GovernanceMetrics(
        false_recommendation_rate=_ratio(len(false_recommendations), len(unsafe)),
        abstention_recall=_ratio(len(true_abstentions), len(unsafe)),
        abstention_precision=_ratio(len(true_abstentions), len(abstentions)),
        coverage=_ratio(len(covered), len(results)),
        selective_accuracy=_ratio(
            sum(result.actual_verdict is result.expected_verdict for result in covered),
            len(covered),
        ),
        limited_case_recall=_ratio(
            sum(result.actual_verdict is EvidenceVerdict.LIMITED for result in limited),
            len(limited),
        ),
        verdict_accuracy=_ratio(
            sum(result.actual_verdict is result.expected_verdict for result in results),
            len(results),
        ),
    )


def _is_abstention(verdict: EvidenceVerdict) -> bool:
    return verdict in {
        EvidenceVerdict.INSUFFICIENT,
        EvidenceVerdict.CONFLICTING,
        EvidenceVerdict.OUT_OF_SCOPE,
    }


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0
