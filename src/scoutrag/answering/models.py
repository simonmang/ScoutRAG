"""Typed contracts for evidence-bound model output."""

from pydantic import Field, model_validator

from scoutrag.domain.base import ScoutRAGModel


class AllowedFact(ScoutRAGModel):
    """One immutable fact a text generator is allowed to verbalize."""

    fact_id: str = Field(min_length=1)
    player_id: str = Field(min_length=1)
    field_name: str = Field(min_length=1)
    display_name: str = Field(min_length=1)
    value: str = Field(min_length=1)
    source_reference: str = Field(min_length=1)


class EvidenceFactCatalog(ScoutRAGModel):
    """Auditable allowlist derived exclusively from an Evidence Pack."""

    facts: list[AllowedFact] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_unique_fact_ids(self) -> "EvidenceFactCatalog":
        fact_ids = [fact.fact_id for fact in self.facts]
        if len(fact_ids) != len(set(fact_ids)):
            raise ValueError("fact IDs must be unique")
        return self

    def by_id(self) -> dict[str, AllowedFact]:
        """Index facts without changing the serialized audit contract."""
        return {fact.fact_id: fact for fact in self.facts}


class GroundedClaim(ScoutRAGModel):
    """A short generated statement linked to explicit allowlisted facts."""

    player_id: str = Field(min_length=1)
    text: str = Field(min_length=1, max_length=500)
    fact_ids: list[str] = Field(min_length=1, max_length=8)


class GroundedAnswerDraft(ScoutRAGModel):
    """Structured model response before deterministic safety validation."""

    claims: list[GroundedClaim] = Field(min_length=1, max_length=20)
