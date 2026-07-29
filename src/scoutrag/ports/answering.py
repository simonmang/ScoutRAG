"""Port for optional, governance-aware natural-language generation."""

from typing import Protocol

from scoutrag.answering.models import GroundedAnswerDraft
from scoutrag.domain.evidence import GeneratedAnswer, RecommendationEvidencePack


class AnswerGenerator(Protocol):
    """Generate only from the immutable evidence pack."""

    def generate(self, evidence_pack: RecommendationEvidencePack) -> GeneratedAnswer:
        """Render a governed answer without inventing facts."""
        ...


class StructuredAnswerBackend(Protocol):
    """Vendor-neutral port for schema-constrained text models."""

    @property
    def model_name(self) -> str:
        """Stable model identifier used in audit output."""
        ...

    def generate_draft(
        self,
        *,
        instructions: str,
        input_text: str,
    ) -> GroundedAnswerDraft:
        """Return schema-validated claims; grounding is checked separately."""
        ...
