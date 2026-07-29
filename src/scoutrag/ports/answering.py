"""Port for optional, governance-aware natural-language generation."""

from typing import Protocol

from scoutrag.domain.evidence import GeneratedAnswer, RecommendationEvidencePack


class AnswerGenerator(Protocol):
    """Generate only from the immutable evidence pack."""

    def generate(self, evidence_pack: RecommendationEvidencePack) -> GeneratedAnswer:
        """Render a governed answer without inventing facts."""
        ...
