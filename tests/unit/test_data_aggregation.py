"""Season aggregation tests, including explicit transfer handling."""

from scoutrag.data.aggregation import aggregate_player_seasons
from scoutrag.data.models import CompetitionSeason, PlayerMatchParticipation


def participation(
    *,
    match_id: int,
    team_id: int,
    team_name: str,
    minutes: float,
) -> PlayerMatchParticipation:
    return PlayerMatchParticipation(
        match_id=match_id,
        competition_id=9,
        season_id=281,
        player_id=7,
        player_name="Transfer Player",
        team_id=team_id,
        team_name=team_name,
        primary_position="Center Midfield",
        position_group="midfield",
        minutes_played=minutes,
        started=True,
        source_reference=f"statsbomb:matches/{match_id}/lineups/players/7",
    )


def test_transfers_remain_one_season_profile_with_team_provenance() -> None:
    competition = CompetitionSeason(
        competition_id=9,
        season_id=281,
        country_name="Germany",
        competition_name="1. Bundesliga",
        season_name="2023/2024",
        competition_gender="male",
        source_reference="statsbomb:competitions/9/seasons/281",
    )

    result = aggregate_player_seasons(
        competition,
        events=[],
        participations=[
            participation(match_id=1, team_id=10, team_name="Alpha FC", minutes=90),
            participation(match_id=2, team_id=20, team_name="Beta 04", minutes=30),
        ],
    )

    assert len(result.profiles) == 1
    profile = result.profiles[0]
    assert profile.team_name == "Alpha FC"
    assert profile.team_names == ["Alpha FC", "Beta 04"]
    assert profile.minutes_played == 120
    assert profile.structured_features["teams_count"] == 2
