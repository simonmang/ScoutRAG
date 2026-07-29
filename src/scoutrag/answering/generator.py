"""Governance-gated, grounding-validated answer generation."""

from scoutrag.answering.facts import build_fact_catalog
from scoutrag.answering.grounding import GroundednessValidator
from scoutrag.answering.models import GroundedAnswerDraft
from scoutrag.answering.prompting import SYSTEM_INSTRUCTIONS, build_generation_input
from scoutrag.answering.templates import TemplateAnswerGenerator
from scoutrag.domain.evidence import (
    EvidenceVerdict,
    GeneratedAnswer,
    GenerationMode,
    GroundingReport,
    RecommendationEvidencePack,
)
from scoutrag.ports.answering import StructuredAnswerBackend

_GENERATIVE_VERDICTS = {EvidenceVerdict.SUFFICIENT, EvidenceVerdict.LIMITED}


class GroundedAnswerGenerator:
    """Use a model only when governance permits and every claim passes validation."""

    def __init__(
        self,
        backend: StructuredAnswerBackend,
        *,
        validator: GroundednessValidator | None = None,
        fallback: TemplateAnswerGenerator | None = None,
    ) -> None:
        self._backend = backend
        self._validator = validator or GroundednessValidator()
        self._fallback = fallback or TemplateAnswerGenerator()

    def generate(self, evidence_pack: RecommendationEvidencePack) -> GeneratedAnswer:
        if evidence_pack.governance.verdict not in _GENERATIVE_VERDICTS:
            return self._fallback.generate(evidence_pack)

        catalog = build_fact_catalog(evidence_pack)
        try:
            draft = self._backend.generate_draft(
                instructions=SYSTEM_INSTRUCTIONS,
                input_text=build_generation_input(evidence_pack, catalog),
            )
            if not isinstance(draft, GroundedAnswerDraft):
                draft = GroundedAnswerDraft.model_validate(draft)
        except Exception as exc:  # A failed optional model must never break safe answers.
            return self._safe_fallback(
                evidence_pack,
                violations=[f"generation backend failed: {type(exc).__name__}"],
            )

        report = self._validator.validate(
            evidence_pack,
            catalog,
            draft,
            generator=self._backend.model_name,
        )
        if not report.validation_passed:
            return self._safe_fallback(
                evidence_pack,
                violations=report.violations,
                report=report,
            )
        return self._render(evidence_pack, draft, report)

    @staticmethod
    def _render(
        pack: RecommendationEvidencePack,
        draft: GroundedAnswerDraft,
        report: GroundingReport,
    ) -> GeneratedAnswer:
        governance = pack.governance
        german = _is_german(pack.query_profile.original_query)
        warnings = list(dict.fromkeys([*governance.warnings, *pack.limitations]))
        if governance.verdict is EvidenceVerdict.LIMITED:
            prefix = (
                f"Ergebnisse mit Einschränkungen. Evidence Quality Score: "
                f"{governance.evidence_quality_score:.3f}."
                if german
                else (
                    f"Results with limitations. Evidence Quality Score: "
                    f"{governance.evidence_quality_score:.3f}."
                )
            )
        else:
            prefix = (
                f"Belastbare Evidenz. Evidence Quality Score: "
                f"{governance.evidence_quality_score:.3f}."
                if german
                else (
                    f"Sufficient evidence. Evidence Quality Score: "
                    f"{governance.evidence_quality_score:.3f}."
                )
            )
        statements = [f"{index}. {claim.text}" for index, claim in enumerate(draft.claims, start=1)]
        suffix = ""
        if governance.verdict is EvidenceVerdict.LIMITED and warnings:
            label = "Einschränkungen" if german else "Limitations"
            suffix = f" {label}: {'; '.join(warnings)}"
        return GeneratedAnswer(
            query_id=pack.retrieval_trace.query_id,
            verdict=governance.verdict,
            text=" ".join([prefix, *statements]) + suffix,
            cited_player_ids=list(dict.fromkeys(claim.player_id for claim in draft.claims)),
            warnings=warnings,
            generation_mode=GenerationMode.GROUNDED_MODEL,
            grounding=report,
        )

    def _safe_fallback(
        self,
        pack: RecommendationEvidencePack,
        *,
        violations: list[str],
        report: GroundingReport | None = None,
    ) -> GeneratedAnswer:
        answer = self._fallback.generate(pack)
        failed_report = report or GroundingReport(
            validation_passed=False,
            grounding_score=0,
            violations=violations,
            generator=self._backend.model_name,
            fallback_used=True,
        )
        if not failed_report.fallback_used:
            failed_report = failed_report.model_copy(update={"fallback_used": True})
        return answer.model_copy(
            update={
                "generation_mode": GenerationMode.SAFE_FALLBACK,
                "grounding": failed_report,
            }
        )


def _is_german(query: str) -> bool:
    normalized = f" {query.casefold()} "
    return any(
        marker in normalized
        for marker in (
            " zeige ",
            " spieler",
            " wer ",
            " mit ",
            " von ",
            " vergleiche ",
            " sechser",
            " zehner",
            " pressingstark",
            "ä",
            "ö",
            "ü",
            "ß",
        )
    )
