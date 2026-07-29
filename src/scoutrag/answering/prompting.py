"""Prompt construction that treats the fact catalog as a strict allowlist."""

import json

from scoutrag.answering.models import EvidenceFactCatalog
from scoutrag.domain.evidence import RecommendationEvidencePack

SYSTEM_INSTRUCTIONS = """\
You verbalize a ScoutRAG Recommendation Evidence Pack.
Return only the requested structured GroundedAnswerDraft.
Use only player IDs and facts present in the supplied fact catalog.
Every claim must cite all supporting fact_ids.
Copy every number exactly from a cited fact; do not calculate or compare values.
Do not add players, seasons, metrics, missing values, predictions, or tactical inferences.
Use short factual sentences and the language of the user's query.
If a fact does not support a statement directly, do not make that statement.
"""


def build_generation_input(
    pack: RecommendationEvidencePack,
    catalog: EvidenceFactCatalog,
) -> str:
    """Serialize the complete and auditable generation boundary."""
    payload = {
        "query": pack.query_profile.original_query,
        "verdict": pack.governance.verdict.value,
        "evidence_quality_score": pack.governance.evidence_quality_score,
        "governance_reasons": pack.governance.reasons,
        "warnings": [*pack.governance.warnings, *pack.limitations],
        "allowed_player_ids": [
            candidate.profile.player_id for candidate in pack.candidates
        ],
        "facts": [fact.model_dump(mode="json") for fact in catalog.facts],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)
