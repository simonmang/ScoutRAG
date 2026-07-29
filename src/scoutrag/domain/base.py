"""Shared Pydantic configuration for ScoutRAG domain objects."""

from pydantic import BaseModel, ConfigDict


class ScoutRAGModel(BaseModel):
    """Strict base model to keep API and audit data predictable."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
        validate_assignment=True,
    )
