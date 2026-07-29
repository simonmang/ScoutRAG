"""Reusable test fixtures for Phase 1 domain contracts."""

import pytest

from scoutrag.domain.player import PlayerSeasonProfile
from scoutrag.domain.retrieval import CandidateRetrievalTrace, PlayerCandidate


@pytest.fixture
def player_profile() -> PlayerSeasonProfile:
    return PlayerSeasonProfile(
        player_id="player-7",
        player_name="Ada Beispiel",
        team_name="FC Beispiel",
        competition_name="Bundesliga",
        season_name="2025/2026",
        position_group="central_midfield",
        minutes_played=1_420,
        structured_features={"ball_recoveries_per_90": 8.4},
        percentiles={"ball_recoveries_per_90": 91.0},
        profile_text=("Central midfielder with high ball-recovery volume in the 2025/2026 season."),
        data_quality=0.94,
    )


@pytest.fixture
def player_candidate(player_profile: PlayerSeasonProfile) -> PlayerCandidate:
    return PlayerCandidate(
        profile=player_profile,
        retrieval_trace=CandidateRetrievalTrace(
            player_id=player_profile.player_id,
            retrieved_by=["sparse", "structured"],
            sparse_score=0.82,
            structured_score=0.91,
            fused_score=0.87,
        ),
    )
