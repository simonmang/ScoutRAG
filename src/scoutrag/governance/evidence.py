"""Typed loading and candidate-safe selection of metric evidence."""

from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq  # type: ignore[import-untyped]

from scoutrag.domain.player import PlayerMetricEvidence
from scoutrag.domain.retrieval import RankedPlayerCandidate


def load_metric_evidence(path: Path) -> list[PlayerMetricEvidence]:
    """Load Phase 3 metric evidence without converting it into text chunks."""
    return [PlayerMetricEvidence.model_validate(row) for row in pq.read_table(path).to_pylist()]


class PlayerMetricEvidenceIndex:
    """Select evidence only for the ranked player IDs under governance."""

    def __init__(self, evidence: list[PlayerMetricEvidence]) -> None:
        grouped: defaultdict[str, list[PlayerMetricEvidence]] = defaultdict(list)
        for item in evidence:
            grouped[item.player_id].append(item)
        self._by_player = {
            player_id: tuple(
                sorted(
                    items,
                    key=lambda item: (
                        item.season_id,
                        item.metric_name,
                        item.source_reference,
                    ),
                )
            )
            for player_id, items in grouped.items()
        }

    def for_candidates(
        self,
        candidates: list[RankedPlayerCandidate],
    ) -> dict[str, list[PlayerMetricEvidence]]:
        """Return a copy so downstream code cannot mutate the index."""
        return {
            candidate.profile.player_id: list(self._by_player[candidate.profile.player_id])
            for candidate in candidates
            if candidate.profile.player_id in self._by_player
        }
