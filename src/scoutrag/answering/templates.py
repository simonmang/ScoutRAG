"""Deterministic safe answer rendering directly from an Evidence Pack."""

from scoutrag.domain.evidence import (
    EvidenceVerdict,
    GeneratedAnswer,
    GenerationMode,
    GroundingReport,
    RecommendationEvidencePack,
)
from scoutrag.domain.player import PlayerMetricEvidence
from scoutrag.domain.retrieval import RankedPlayerCandidate


class TemplateAnswerGenerator:
    """Render an explanation without an LLM, new facts, or calculations."""

    def generate(self, evidence_pack: RecommendationEvidencePack) -> GeneratedAnswer:
        governance = evidence_pack.governance
        german = _is_german(evidence_pack.query_profile.original_query)
        warnings = [*governance.warnings, *evidence_pack.limitations]
        warnings = list(dict.fromkeys(warnings))

        if governance.verdict is EvidenceVerdict.OUT_OF_SCOPE:
            text = (
                "ScoutRAG unterstützt Spielersuche, Spielerprofile und evidenzbasierte "
                "Vergleiche, aber keine Ergebnisvorhersagen."
                if german
                else (
                    "ScoutRAG supports player discovery, player profiles, and evidence-based "
                    "comparisons, but not result predictions."
                )
            )
            cited_ids: list[str] = []
        elif governance.verdict is EvidenceVerdict.INSUFFICIENT:
            missing = "; ".join(governance.missing_evidence) or "Unspecified evidence is missing."
            text = (
                f"Keine belastbare Spielerempfehlung. Fehlende Evidenz: {missing}"
                if german
                else f"No reliable player recommendation. Missing evidence: {missing}"
            )
            cited_ids = []
        elif governance.verdict is EvidenceVerdict.CONFLICTING:
            conflicts = "; ".join(governance.reasons)
            text = (
                f"Die Evidenz ist widersprüchlich. Konflikte: {conflicts}"
                if german
                else f"The evidence is conflicting. Conflicts: {conflicts}"
            )
            cited_ids = []
        else:
            lines = [
                self._candidate_line(candidate, evidence_pack, german)
                for candidate in evidence_pack.candidates
            ]
            score = governance.evidence_quality_score
            if governance.verdict is EvidenceVerdict.LIMITED:
                prefix = (
                    f"Ergebnisse mit Einschränkungen. Evidence Quality Score: {score:.3f}."
                    if german
                    else f"Results with limitations. Evidence Quality Score: {score:.3f}."
                )
                limitation_text = "; ".join(warnings)
                suffix = f" Einschränkungen: {limitation_text}" if limitation_text else ""
            else:
                prefix = (
                    f"Belastbare Evidenz. Evidence Quality Score: {score:.3f}."
                    if german
                    else f"Sufficient evidence. Evidence Quality Score: {score:.3f}."
                )
                suffix = ""
            text = " ".join([prefix, *lines]) + suffix
            cited_ids = [candidate.profile.player_id for candidate in evidence_pack.candidates]

        return GeneratedAnswer(
            query_id=evidence_pack.retrieval_trace.query_id,
            verdict=governance.verdict,
            text=text,
            cited_player_ids=cited_ids,
            warnings=warnings,
            generation_mode=GenerationMode.TEMPLATE,
            grounding=GroundingReport(generator="template"),
        )

    @staticmethod
    def _candidate_line(
        candidate: RankedPlayerCandidate,
        pack: RecommendationEvidencePack,
        german: bool,
    ) -> str:
        profile = candidate.profile
        requested = set(pack.query_profile.requested_metrics)
        evidence = [
            item
            for item in pack.metric_evidence.get(profile.player_id, [])
            if item.metric_name in requested
        ]
        facts = ", ".join(_metric_fact(item) for item in evidence)
        base = (
            f"{candidate.rank}. {profile.player_name} — {profile.team_name}, {profile.season_name}"
        )
        if not facts:
            return f"{base}."
        label = "Evidenz" if german else "Evidence"
        return f"{base}. {label}: {facts}."


def _metric_fact(item: PlayerMetricEvidence) -> str:
    value = item.normalized_value if item.normalized_value is not None else item.raw_value
    parts = [item.metric_name.replace("_", " ")]
    if value is not None:
        parts.append(f"{value:g}")
    if item.percentile is not None:
        parts.append(f"P{item.percentile:g}")
    return " ".join(parts)


def _is_german(query: str) -> bool:
    normalized = f" {query.casefold()} "
    markers = (
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
    return any(marker in normalized for marker in markers)
