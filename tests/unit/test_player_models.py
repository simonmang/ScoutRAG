"""Typed retrieval unit validation."""

import pytest
from pydantic import ValidationError

from scoutrag.domain.player import PlayerMetricEvidence, PlayerSeasonProfile
from scoutrag.domain.retrieval import CandidateRetrievalTrace, PlayerCandidate


def test_percentiles_reject_values_outside_zero_to_one_hundred() -> None:
    with pytest.raises(ValidationError):
        PlayerSeasonProfile(
            player_id="p-1",
            player_name="Spieler Eins",
            team_name="Test FC",
            competition_name="Testliga",
            season_name="2025/2026",
            position_group="forward",
            minutes_played=900,
            percentiles={"shots": 101},
            profile_text="Forward profile.",
            data_quality=0.9,
        )


def test_metric_evidence_requires_provenance() -> None:
    with pytest.raises(ValidationError):
        PlayerMetricEvidence(
            player_id="p-1",
            season_id="season-1",
            metric_name="pressures_per_90",
            raw_value=13.2,
            comparison_group="central_midfield",
            source_reference="",
        )


def test_candidate_trace_must_belong_to_profile(
    player_profile: PlayerSeasonProfile,
) -> None:
    with pytest.raises(ValidationError):
        PlayerCandidate(
            profile=player_profile,
            retrieval_trace=CandidateRetrievalTrace(
                player_id="different-player",
                retrieved_by=["dense"],
                dense_score=0.8,
                fused_score=0.8,
            ),
        )
