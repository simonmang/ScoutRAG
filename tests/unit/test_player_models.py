"""Typed retrieval unit validation."""

from datetime import date, timedelta

import pytest
from pydantic import ValidationError

from scoutrag.domain.player import PlayerMetricEvidence, PlayerSeasonProfile
from scoutrag.domain.retrieval import CandidateRetrievalTrace, PlayerCandidate


def _profile_with_metadata(**metadata: object) -> PlayerSeasonProfile:
    return PlayerSeasonProfile(
        player_id="p-1",
        player_name="Spieler Eins",
        team_name="FC Bayern München",
        competition_name="Bundesliga",
        season_name="2025/2026",
        position_group="central_midfield",
        minutes_played=900,
        profile_text="Central midfielder profile.",
        data_quality=0.9,
        **metadata,
    )


def test_player_metadata_is_optional_for_backward_compatibility() -> None:
    profile = _profile_with_metadata()

    assert profile.date_of_birth is None
    assert profile.birth_place is None
    assert profile.birth_country is None
    assert profile.nationality is None
    assert profile.height_cm is None
    assert profile.weight_kg is None
    assert profile.photo_url is None


def test_player_metadata_accepts_api_football_values() -> None:
    profile = _profile_with_metadata(
        date_of_birth=date(1995, 2, 8),
        birth_place="Kempten",
        birth_country="Germany",
        nationality="Germany",
        height_cm=177,
        weight_kg=75,
        photo_url="https://media.api-sports.io/football/players/521.png",
    )

    assert profile.date_of_birth == date(1995, 2, 8)
    assert profile.birth_place == "Kempten"
    assert profile.birth_country == "Germany"
    assert profile.nationality == "Germany"
    assert profile.height_cm == 177
    assert profile.weight_kg == 75


def test_blank_optional_player_text_is_normalized_to_none() -> None:
    profile = _profile_with_metadata(
        birth_place=" ",
        birth_country="\t",
        nationality="",
        photo_url="  ",
    )

    assert profile.birth_place is None
    assert profile.birth_country is None
    assert profile.nationality is None
    assert profile.photo_url is None


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("date_of_birth", date.today() + timedelta(days=1)),
        ("height_cm", 99),
        ("height_cm", 251),
        ("weight_kg", 29),
        ("weight_kg", 251),
        ("photo_url", "ftp://example.com/player.png"),
    ],
)
def test_player_metadata_rejects_implausible_values(
    field_name: str,
    invalid_value: object,
) -> None:
    with pytest.raises(ValidationError):
        _profile_with_metadata(**{field_name: invalid_value})


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
