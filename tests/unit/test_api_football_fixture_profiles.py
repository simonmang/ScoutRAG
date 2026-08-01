"""Fixture-level API-Football aggregation tests with no network dependency."""

from typing import Any

import pytest

from scoutrag.data.api_football_fixture_profiles import (
    build_api_football_fixture_profiles,
)


def _stats(
    *,
    minutes: int,
    position: str = "M",
    substitute: bool = False,
    captain: bool = False,
    rating: str | None = "7.0",
    passes: int | None = 50,
    completed_passes: str | None = "40",
    fouls_committed: int | None = 2,
) -> dict[str, Any]:
    return {
        "games": {
            "minutes": minutes,
            "number": 6,
            "position": position,
            "rating": rating,
            "captain": captain,
            "substitute": substitute,
        },
        "offsides": None,
        "shots": {"total": None, "on": None},
        "goals": {"total": None, "conceded": None, "assists": None, "saves": None},
        "passes": {"total": passes, "key": None, "accuracy": completed_passes},
        "tackles": {"total": 2, "blocks": None, "interceptions": 1},
        "duels": {"total": 5, "won": 3},
        "dribbles": {"attempts": None, "success": None, "past": None},
        "fouls": {"drawn": None, "committed": fouls_committed},
        "cards": {"yellow": 0, "yellowred": None, "red": 0},
        "penalty": {"won": None, "commited": None, "scored": 0, "missed": 0, "saved": 0},
    }


def _fixture(
    fixture_id: int,
    groups: list[tuple[int, str, list[tuple[int, str, dict[str, Any]]]]],
    *,
    league_id: int = 78,
    season: int = 2024,
    round_name: str = "Regular Season - 1",
) -> dict[str, Any]:
    teams = [{"id": team_id, "name": team_name} for team_id, team_name, _ in groups]
    return {
        "fixture": {"id": fixture_id, "date": "2024-08-24T13:30:00+00:00"},
        "league": {
            "id": league_id,
            "name": "Bundesliga",
            "season": season,
            "round": round_name,
        },
        "teams": {
            "home": teams[0],
            "away": teams[1] if len(teams) > 1 else {"id": 9999, "name": "Opponent"},
        },
        "players": [
            {
                "team": {"id": team_id, "name": team_name},
                "players": [
                    {
                        "player": {"id": player_id, "name": player_name},
                        "statistics": [statistics],
                    }
                    for player_id, player_name, statistics in players
                ],
            }
            for team_id, team_name, players in groups
        ],
    }


def _payloads() -> list[dict[str, Any]]:
    fixture_one = _fixture(
        100,
        [
            (
                168,
                "Bayer Leverkusen",
                [
                    (
                        1,
                        "J. Tah",
                        _stats(
                            minutes=90,
                            captain=True,
                            rating="8.0",
                            passes=50,
                            completed_passes="40",
                            fouls_committed=10,
                        ),
                    ),
                    (2, "Low Foul Peer", _stats(minutes=90, fouls_committed=1)),
                    (3, "Middle Foul Peer", _stats(minutes=90, fouls_committed=5)),
                    (99, "Unused Player", _stats(minutes=0)),
                ],
            )
        ],
    )
    fixture_one["lineups"] = [
        {
            "startXI": [
                {"player": {"id": 1, "name": "Wrong Lineup Identity"}},
            ],
            "substitutes": [],
        }
    ]
    fixture_two = _fixture(
        101,
        [
            (
                168,
                "Bayer Leverkusen",
                [
                    (
                        1,
                        "J. Tah",
                        _stats(
                            minutes=30,
                            substitute=True,
                            rating="6.0",
                            passes=20,
                            completed_passes="10",
                            fouls_committed=None,
                        ),
                    )
                ],
            ),
            (
                157,
                "FC Bayern München",
                [(4, "Transfer Player", _stats(minutes=30, position="D"))],
            ),
        ],
        round_name="Regular Season - 2",
    )
    fixture_three = _fixture(
        102,
        [
            (
                168,
                "Bayer Leverkusen",
                [(4, "Transfer Player", _stats(minutes=90, position="D"))],
            )
        ],
        round_name="Regular Season - 3",
    )
    wrong_season = _fixture(103, [(157, "FC Bayern München", [(5, "Wrong", _stats(minutes=90))])])
    wrong_season["league"]["season"] = 2023
    relegation = _fixture(
        104,
        [(999, "SV Elversberg", [(6, "Relegation Player", _stats(minutes=90))])],
        round_name="Relegation Round",
    )
    return [
        fixture_one,
        fixture_two,
        fixture_three,
        fixture_one,  # Exact duplicate must not double-count fixture 100.
        wrong_season,
        relegation,
    ]


def _identities() -> list[dict[str, Any]]:
    return [
        {
            "player": {
                "id": 1,
                "name": "J. Tah",
                "firstname": "Jonathan Glao",
                "birth": {"date": "1996-02-11", "place": "Hamburg", "country": "Germany"},
                "nationality": "Germany",
                "height": "195 cm",
                "weight": "94 kg",
                "photo": "https://example.test/tah.png",
            },
            # Deliberately malicious current-team contamination: ignored in full.
            "statistics": [
                {
                    "team": {"id": 157, "name": "FC Bayern München"},
                    "games": {"minutes": 9999, "position": "Defender"},
                }
            ],
        }
    ]


def test_aggregates_unique_regular_fixtures_without_current_team_contamination() -> None:
    result = build_api_football_fixture_profiles(
        _payloads(),
        league_id=78,
        season_start_year=2024,
        competition_name="Bundesliga",
        player_identity_payloads=_identities(),
        minimum_minutes=1,
        minimum_comparison_group_size=3,
    )

    tah = next(profile for profile in result.profiles if profile.player_id == "api-football:1")
    assert tah.player_name == "Jonathan Tah"
    assert tah.profile_id == "api-football:78:2024:1"
    assert tah.team_name == "Bayer Leverkusen"
    assert tah.team_names == ["Bayer Leverkusen"]
    assert tah.position_group == "midfielder"
    assert tah.minutes_played == 120
    assert tah.structured_features["appearances"] == 2
    assert tah.structured_features["starts"] == 1
    assert tah.structured_features["substitute_appearances"] == 1
    assert tah.structured_features["captain_appearances"] == 1
    assert tah.structured_features["captain_flag"] == 1
    assert tah.structured_features["passes"] == 70
    assert tah.structured_features["passes_completed"] == 50
    assert tah.structured_features["pass_completion_rate"] == pytest.approx(71.4286)
    assert tah.structured_features["average_rating"] == 7.5
    assert tah.structured_features["rating_minutes_coverage"] == 1
    assert tah.structured_features["fouls_committed"] == 10
    assert tah.structured_features["null_action_values_as_zero"] > 0
    assert tah.date_of_birth is not None
    assert tah.birth_place == "Hamburg"
    assert tah.height_cm == 195
    assert tah.percentiles["fouls_committed_per_90"] == 0

    transfer = next(profile for profile in result.profiles if profile.player_id == "api-football:4")
    assert transfer.team_name == "Bayer Leverkusen"
    assert transfer.team_names == ["Bayer Leverkusen", "FC Bayern München"]
    assert transfer.minutes_played == 120
    assert transfer.position_group == "defender"

    assert all(profile.player_id != "api-football:99" for profile in result.profiles)
    assert all(profile.player_name != "Wrong" for profile in result.profiles)
    assert all(profile.player_name != "Relegation Player" for profile in result.profiles)
    assert any("duplicate" in limitation for limitation in result.limitations)
    assert any("identity mismatch" in limitation for limitation in result.limitations)
    assert any("Regular Season" in limitation for limitation in result.limitations)
    assert all(
        item.profile_id == f"api-football:78:2024:{item.player_id.split(':')[-1]}"
        for item in result.evidence
    )


def test_calendar_season_and_same_league_postseason_are_explicit() -> None:
    regular = _fixture(
        201,
        [
            (
                10,
                "Home",
                [
                    (0, "Provider Placeholder", _stats(minutes=90)),
                    (1, "Player One", _stats(minutes=90)),
                ],
            ),
            (20, "Away", [(2, "Player Two", _stats(minutes=90))]),
        ],
    )
    postseason = _fixture(
        202,
        [
            (10, "Home", [(1, "Player One", _stats(minutes=90))]),
            (20, "Away", [(2, "Player Two", _stats(minutes=90))]),
        ],
        round_name="Championship Round - 1",
    )
    outsider_playoff = _fixture(
        203,
        [
            (10, "Home", [(1, "Player One", _stats(minutes=90))]),
            (30, "Outsider", [(3, "Outsider", _stats(minutes=90))]),
        ],
        round_name="Final",
    )

    result = build_api_football_fixture_profiles(
        [regular, postseason, outsider_playoff],
        league_id=78,
        season_start_year=2024,
        season_name="2024",
        competition_name="Calendar League",
        minimum_minutes=1,
        include_same_league_postseason=True,
    )

    one = next(item for item in result.profiles if item.player_id == "api-football:1")
    assert one.season_name == "2024"
    assert one.minutes_played == 180
    assert all(item.player_id != "api-football:0" for item in result.profiles)
    assert all(item.player_id != "api-football:3" for item in result.profiles)
    assert any("cross-league play-offs were excluded" in item for item in result.limitations)


def test_withholds_structurally_missing_metric_but_maps_present_null_action_to_zero() -> None:
    payloads = _payloads()
    # Removing a documented key is different from the provider's present null no-action value.
    del payloads[1]["players"][0]["players"][0]["statistics"][0]["tackles"]["total"]
    result = build_api_football_fixture_profiles(
        payloads,
        league_id=78,
        season_start_year=2024,
        competition_name="Bundesliga",
        player_identity_payloads=_identities(),
        minimum_minutes=1,
        minimum_comparison_group_size=3,
    )

    tah = next(profile for profile in result.profiles if profile.player_id == "api-football:1")
    assert "tackles" not in tah.structured_features
    assert "tackles_per_90" not in tah.structured_features
    assert tah.structured_features["goals"] == 0
    assert tah.structured_features["goals_per_90"] == 0
    assert tah.structured_features["fixture_stat_coverage"] < 1
    assert not any(
        item.player_id == tah.player_id and item.metric_name == "tackles_per_90"
        for item in result.evidence
    )
    assert {item.metric_name for item in result.evidence} <= {
        definition.metric_name for definition in result.definitions
    }


def test_can_include_non_regular_rounds_explicitly() -> None:
    result = build_api_football_fixture_profiles(
        _payloads(),
        league_id=78,
        season_start_year=2024,
        competition_name="Bundesliga",
        minimum_minutes=1,
        minimum_comparison_group_size=2,
        enable_percentiles=False,
        round_prefix=None,
    )

    assert any(profile.player_name == "Relegation Player" for profile in result.profiles)


def test_lineup_formation_and_grid_refine_position_group() -> None:
    fixtures = []
    for fixture_id in range(200, 206):
        fixture = _fixture(
            fixture_id,
            [(168, "Bayer Leverkusen", [(10, "D. Mid", _stats(minutes=90, position="M"))])],
        )
        fixture["lineups"] = [
            {
                "formation": "4-2-3-1",
                "startXI": [{"player": {"id": 10, "grid": "3:1"}}],
                "substitutes": [],
            }
        ]
        fixtures.append(fixture)

    result = build_api_football_fixture_profiles(
        fixtures,
        league_id=78,
        season_start_year=2024,
        competition_name="Bundesliga",
        minimum_minutes=1,
        minimum_comparison_group_size=2,
        enable_percentiles=False,
    )

    player = next(profile for profile in result.profiles if profile.player_id == "api-football:10")
    assert player.position_group == "defensive_midfield"
    assert player.structured_features["position_refined"] == 1.0
    assert player.structured_features["position_confidence"] == 1.0


def test_lineup_without_formation_or_grid_keeps_coarse_position() -> None:
    # Mirrors the identity-mismatch fixture shape: lineups present, but with no
    # formation/grid, which must never crash and must never fabricate a role.
    fixtures = []
    for fixture_id in range(300, 304):
        fixture = _fixture(
            fixture_id,
            [(168, "Bayer Leverkusen", [(11, "No Grid", _stats(minutes=90, position="M"))])],
        )
        fixture["lineups"] = [
            {"startXI": [{"player": {"id": 11, "name": "No Grid"}}], "substitutes": []}
        ]
        fixtures.append(fixture)

    result = build_api_football_fixture_profiles(
        fixtures,
        league_id=78,
        season_start_year=2024,
        competition_name="Bundesliga",
        minimum_minutes=1,
        minimum_comparison_group_size=2,
        enable_percentiles=False,
    )

    player = next(profile for profile in result.profiles if profile.player_id == "api-football:11")
    assert player.position_group == "midfielder"
    assert player.structured_features["position_refined"] == 0.0
    assert player.structured_features["position_confidence"] == 0.0
