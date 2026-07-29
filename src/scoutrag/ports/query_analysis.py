"""Port for deterministic or optional model-assisted query analysis."""

from typing import Protocol

from scoutrag.domain.query import QueryProfile


class QueryAnalyzer(Protocol):
    """Convert raw input into a typed retrieval plan."""

    def analyze(self, query: str) -> QueryProfile:
        """Analyze and normalize a user query."""
        ...
