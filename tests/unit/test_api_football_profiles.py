"""Provider-specific profile conversion tests using inline API fixtures."""

import json
from pathlib import Path
from typing import Any, cast

import pyarrow.parquet as pq  # type: ignore[import-untyped]

from scoutrag.data.api_football_profiles import (
    ApiFootballDatasetWriter,
    build_api_football_profiles,
)


def _statistics(
    team_id: int,
    team_name: str,
    *,
    minutes: int,
    passes: int,
    tackles: int,
    goals: int,
    assists: int | None = 1,
    position: str = "Midfielder",
    appearances: int = 10,
    rating: float | None = 7.0,
    pass_accuracy: int | None = 80,
    fouls_committed: int | None = 9,
) -> dict[str, Any]:
    return {
        "team": {"id": team_id, "name": team_name},
        "league": {"id": 78, "name": "Bundesliga", "season": 2023},
        "games": {
            "appearences": appearances,
            "lineups": 8,
            "minutes": minutes,
            "number": 6,
            "position": position,
            "rating": rating,
            "captain": False,
        },
        "substitutes": {"in": 2, "out": 3, "bench": 4},
        "shots": {"total": 10, "on": 4},
        "goals": {"total": goals, "assists": assists, "conceded": 0, "saves": 0},
        "passes": {"total": passes, "key": 8, "accuracy": pass_accuracy},
        "tackles": {"total": tackles, "blocks": 3, "interceptions": 6},
        "duels": {"total": 70, "won": 40},
        "dribbles": {"attempts": 12, "success": 7, "past": 3},
        "fouls": {"drawn": 10, "committed": fouls_committed},
        "cards": {"yellow": 2, "yellowred": 0, "red": 0},
        "penalty": {"won": 1, "commited": 0, "scored": 1, "missed": 0, "saved": 0},
    }


def _payload() -> list[dict[str, Any]]:
    transfer_first = _statistics(
        1,
        "FC Bayern München",
        minutes=600,
        passes=500,
        tackles=30,
        goals=2,
        assists=2,
    )
    transfer_second = _statistics(
        2,
        "VfB Stuttgart",
        minutes=300,
        passes=200,
        tackles=12,
        goals=1,
        assists=None,
    )
    wrong_season = _statistics(
        99,
        "Old Club",
        minutes=900,
        passes=999,
        tackles=99,
        goals=99,
    )
    wrong_season["league"]["season"] = 2022
    return [
        {
            "player": {
                "id": 10,
                "name": "Transfer Spieler",
                "birth": {"date": "1990-07-02", "place": "Berlin", "country": "Germany"},
                "nationality": "Germany",
                "height": "180 cm",
                "weight": "75 kg",
                "photo": "https://example.test/10.png",
            },
            "statistics": [transfer_first, transfer_second, wrong_season],
        },
        {
            "player": {
                "id": 11,
                "name": "J. Kimmich",
                "firstname": "Joshua Walter",
                "birth": {
                    "date": "1995-02-08",
                    "place": "Rottweil",
                    "country": "Germany",
                },
                "nationality": "Germany",
                "height": "177",
                "weight": "75",
                "photo": "https://example.test/11.png",
                "age": 99,
                "injured": True,
            },
            "statistics": [
                _statistics(
                    1,
                    "FC Bayern München",
                    minutes=900,
                    passes=900,
                    tackles=60,
                    goals=4,
                )
            ],
        },
        {
            "player": {"id": 12, "name": "Bayern Spieler Drei"},
            "statistics": [
                _statistics(
                    1,
                    "FC Bayern München",
                    minutes=900,
                    passes=400,
                    tackles=15,
                    goals=1,
                )
            ],
        },
    ]


def test_builds_transfer_safe_profiles_percentiles_and_evidence() -> None:
    result = build_api_football_profiles(
        _payload(),
        league_id=78,
        season_start_year=2023,
        competition_name="Bundesliga",
        minimum_minutes=450,
        full_sample_minutes=900,
        minimum_comparison_group_size=3,
    )

    transfer = next(
        profile for profile in result.profiles if profile.player_id == "api-football:10"
    )
    kimmich = next(profile for profile in result.profiles if profile.player_id == "api-football:11")
    assert kimmich.player_name == "Joshua Kimmich"
    assert str(kimmich.date_of_birth) == "1995-02-08"
    assert kimmich.birth_place == "Rottweil"
    assert kimmich.birth_country == "Germany"
    assert kimmich.nationality == "Germany"
    assert kimmich.height_cm == 177
    assert kimmich.weight_kg == 75
    assert kimmich.photo_url == "https://example.test/11.png"
    assert kimmich.structured_features["age_at_season_start"] == 28
    assert "age" not in kimmich.structured_features
    assert "injured" not in kimmich.structured_features
    assert transfer.minutes_played == 900
    assert transfer.team_name == "FC Bayern München"
    assert transfer.team_names == ["FC Bayern München", "VfB Stuttgart"]
    assert transfer.structured_features["teams_count"] == 2
    assert transfer.structured_features["passes"] == 700
    assert transfer.structured_features["passes_per_90"] == 70
    assert transfer.structured_features["goals"] == 3
    assert transfer.structured_features["appearances"] == 20
    assert transfer.structured_features["starts"] == 16
    assert transfer.structured_features["substitutions_in"] == 4
    assert transfer.structured_features["substitutions_out"] == 6
    assert transfer.structured_features["bench_appearances"] == 8
    assert transfer.structured_features["captain_flag"] == 0
    assert transfer.structured_features["shirt_number"] == 6
    assert transfer.structured_features["average_rating"] == 7
    assert transfer.structured_features["pass_accuracy"] == 80
    assert transfer.structured_features["shots_on_target_rate"] == 40
    assert transfer.structured_features["duel_win_rate"] == 57.1429
    assert transfer.structured_features["dribble_success_rate"] == 58.3333
    assert transfer.structured_features["dribbles_past"] == 6
    assert transfer.structured_features["penalties_scored"] == 2
    assert "assists" not in transfer.structured_features
    assert "assists_per_90" not in transfer.structured_features
    assert "pressures_per_90" not in transfer.structured_features
    assert "progressive_passes_per_90" not in transfer.structured_features
    assert transfer.percentiles["passes_per_90"] == 50
    assert transfer.percentiles["fouls_committed_per_90"] == 0
    assert transfer.data_quality < 1
    assert "API-Football aggregate" in transfer.profile_text
    assert "Fouls committed" not in transfer.profile_text
    assert "Yellow cards" not in transfer.profile_text

    passes = next(
        item
        for item in result.evidence
        if item.player_id == transfer.player_id and item.metric_name == "passes_per_90"
    )
    assert passes.raw_value == 700
    assert passes.normalized_value == 70
    assert passes.percentile == 50
    assert passes.season_id == "api-football:78:2023"
    assert passes.source_reference.endswith("#statistics.passes.total")
    assert "&id=10" in passes.source_reference
    assert not any(
        item.player_id == transfer.player_id and item.metric_name == "assists_per_90"
        for item in result.evidence
    )
    evidence_names = {
        item.metric_name for item in result.evidence if item.player_id == transfer.player_id
    }
    assert {
        "age_at_season_start",
        "appearances",
        "starts",
        "substitutions_in",
        "substitutions_out",
        "bench_appearances",
        "captain_flag",
        "shirt_number",
        "average_rating",
        "pass_accuracy",
        "shots_on_target_rate",
        "duel_win_rate",
        "dribble_success_rate",
        "dribbles_past_per_90",
        "yellow_red_cards_per_90",
        "penalties_won_per_90",
        "penalties_committed_per_90",
        "penalties_scored_per_90",
        "penalties_missed_per_90",
        "penalties_saved_per_90",
    } <= evidence_names
    assert {item.metric_name for item in result.evidence} <= {
        definition.metric_name for definition in result.definitions
    }
    goals_conceded_definition = next(
        item for item in result.definitions if item.metric_name == "goals_conceded_per_90"
    )
    assert any(
        "lower values rank higher" in limitation
        for limitation in goals_conceded_definition.limitations
    )


def test_keeps_unabbreviated_common_name_unchanged() -> None:
    payload = _payload()
    payload[2]["player"] = {
        "id": 12,
        "name": "Kim Min-Jae",
        "firstname": "Min-Jae",
    }

    result = build_api_football_profiles(
        payload,
        league_id=78,
        season_start_year=2023,
        competition_name="Bundesliga",
        minimum_comparison_group_size=3,
    )

    kim = next(profile for profile in result.profiles if profile.player_id == "api-football:12")
    assert kim.player_name == "Kim Min-Jae"


def test_withholds_incomplete_transfer_aggregates_and_excludes_non_participants() -> None:
    payload = _payload()
    first, second = payload[0]["statistics"][:2]
    second["games"]["rating"] = None
    second["passes"]["accuracy"] = None
    second["substitutes"]["out"] = None
    second["dribbles"]["past"] = None
    first["games"]["number"] = 6
    second["games"]["number"] = 9
    payload.append(
        {
            "player": {
                "id": 99,
                "name": "No Season Evidence",
                "birth": {"date": "2000-01-01"},
            },
            "statistics": [
                _statistics(
                    1,
                    "FC Bayern München",
                    minutes=0,
                    passes=0,
                    tackles=0,
                    goals=0,
                    appearances=0,
                    rating=None,
                    pass_accuracy=None,
                )
            ],
        }
    )

    result = build_api_football_profiles(
        payload,
        league_id=78,
        season_start_year=2023,
        competition_name="Bundesliga",
        minimum_comparison_group_size=3,
    )

    transfer = next(
        profile for profile in result.profiles if profile.player_id == "api-football:10"
    )
    assert "average_rating" not in transfer.structured_features
    assert "pass_accuracy" not in transfer.structured_features
    assert "substitutions_out" not in transfer.structured_features
    assert "dribbles_past" not in transfer.structured_features
    assert "shirt_number" not in transfer.structured_features
    assert all(profile.player_id != "api-football:99" for profile in result.profiles)
    assert all(item.player_id != "api-football:99" for item in result.evidence)


def test_weighted_provider_aggregates_use_appearances() -> None:
    payload = _payload()
    first, second = payload[0]["statistics"][:2]
    first["games"]["appearences"] = 10
    first["games"]["rating"] = "8.0"
    first["passes"]["accuracy"] = 90
    second["games"]["appearences"] = 5
    second["games"]["rating"] = 6
    second["passes"]["accuracy"] = 60

    result = build_api_football_profiles(
        payload,
        league_id=78,
        season_start_year=2023,
        competition_name="Bundesliga",
        minimum_comparison_group_size=3,
    )

    transfer = next(
        profile for profile in result.profiles if profile.player_id == "api-football:10"
    )
    assert transfer.structured_features["average_rating"] == 7.3333
    assert transfer.structured_features["pass_accuracy"] == 80


def test_writer_persists_only_canonical_atomic_artifacts(tmp_path: Path) -> None:
    result = build_api_football_profiles(
        _payload(),
        league_id=78,
        season_start_year=2023,
        competition_name="Bundesliga",
        minimum_comparison_group_size=3,
        comparison_scope="FC Bayern München team-filtered sample",
        enable_percentiles=False,
    )
    paths = ApiFootballDatasetWriter().write(
        tmp_path,
        result=result,
        league_id=78,
        season_start_year=2023,
        competition_name="Bundesliga",
    )

    assert [path.name for path in paths] == [
        "player_season_profiles.parquet",
        "player_metric_evidence.parquet",
        "metric_definitions.json",
        "manifest.json",
    ]
    assert not list(tmp_path.glob("*.tmp"))
    profile_table = pq.read_table(tmp_path / "player_season_profiles.parquet")
    assert profile_table.schema.metadata[b"data_source"] == b"API-Football"
    assert profile_table.num_rows == 3

    definitions = cast(
        list[dict[str, Any]],
        json.loads((tmp_path / "metric_definitions.json").read_text("utf-8")),
    )
    names = {item["metric_name"] for item in definitions}
    assert "tackles_per_90" in names
    assert "pressures_per_90" not in names
    assert "progressive_passes_per_90" not in names

    manifest = cast(
        dict[str, Any],
        json.loads((tmp_path / "manifest.json").read_text("utf-8")),
    )
    assert manifest["source"]["provider"] == "API-Football"
    assert manifest["source"]["raw_responses_embedded_in_artifacts"] is False
    assert manifest["source"]["raw_cache_git_tracked"] is False
    assert manifest["schema_version"] == "api-football-v2"
    assert "complete competition comparison group" in manifest["limitations"][0]
    assert set(manifest["artifacts"]) == {
        "player_season_profiles.parquet",
        "player_metric_evidence.parquet",
        "metric_definitions.json",
    }
    assert not any("raw" in path.name for path in tmp_path.iterdir())

    assert all(not profile.percentiles for profile in result.profiles)
    assert all("percentiles=disabled" in item.comparison_group for item in result.evidence)
