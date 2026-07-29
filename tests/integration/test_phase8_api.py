"""Phase 8 governed API and same-origin dashboard integration."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from scoutrag.config import Settings
from scoutrag.domain.player import PlayerMetricEvidence, PlayerSeasonProfile
from scoutrag.governance.factory import build_governed_pipeline
from scoutrag.main import create_app


@pytest.fixture
def client() -> Iterator[TestClient]:
    profiles = [
        profile(
            "5579",
            "Joshua Kimmich",
            "Bayern Munich",
            source_coverage=0.1,
            minutes=180,
            percentile=None,
        ),
        profile(
            "40724",
            "Florian Wirtz",
            "Bayer Leverkusen",
            source_coverage=1,
            minutes=2_500,
            percentile=95,
        ),
    ]
    evidence = [
        metric("5579", percentile=None, minutes=180),
        metric("40724", percentile=95, minutes=2_500),
    ]
    app = create_app(
        Settings(
            environment="test",
            max_result_count=5,
            enable_dense_retrieval=False,
        ),
        pipeline=build_governed_pipeline(profiles, evidence),
    )
    with TestClient(app) as test_client:
        yield test_client


def profile(
    player_id: str,
    name: str,
    team: str,
    *,
    source_coverage: float,
    minutes: float,
    percentile: float | None,
) -> PlayerSeasonProfile:
    return PlayerSeasonProfile(
        player_id=player_id,
        player_name=name,
        team_name=team,
        team_names=[team],
        competition_name="1. Bundesliga",
        season_name="2023/2024",
        position_group="defensive_midfield",
        minutes_played=minutes,
        structured_features={
            "pressures_per_90": 10,
            "source_coverage_ratio": source_coverage,
            "feature_coverage_ratio": 1,
            "comparison_group_size": 3,
        },
        percentiles=({"pressures_per_90": percentile} if percentile is not None else {}),
        profile_text=f"{name} | {team} | defensive midfield",
        data_quality=source_coverage,
    )


def metric(
    player_id: str,
    *,
    percentile: float | None,
    minutes: float,
) -> PlayerMetricEvidence:
    return PlayerMetricEvidence(
        player_id=player_id,
        season_id="281",
        metric_name="pressures_per_90",
        raw_value=20,
        normalized_value=10,
        percentile=percentile,
        comparison_group="Bundesliga defensive midfield n=3",
        sample_size=minutes,
        source_reference=f"test:{player_id}",
    )


def test_retrieve_returns_complete_governed_evidence_pack(client: TestClient) -> None:
    response = client.post(
        "/api/v1/retrieve",
        json={
            "query": "Zeige das Profil von Joshua Kimmich",
            "result_count": 3,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["query_profile"]["named_players"] == ["Joshua Kimmich"]
    assert payload["governance"]["verdict"] == "limited"
    assert payload["governance"]["factors"]["data_coverage"] == 0.1
    assert payload["candidates"][0]["profile"]["team_name"] == "Bayern Munich"
    assert payload["metric_evidence"]["5579"][0]["source_reference"] == "test:5579"
    assert payload["retrieval_trace"]["query_id"]
    assert payload["runtime_metrics"]["governance_ms"] >= 0


def test_search_is_a_compact_projection_of_the_same_pipeline(client: TestClient) -> None:
    response = client.post(
        "/api/v1/search",
        json={"query": "Zeige das Profil von Joshua Kimmich"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["verdict"] == "limited"
    assert payload["candidates"][0]["player_name"] == "Joshua Kimmich"
    assert payload["candidates"][0]["retrieved_by"]
    assert "metric_evidence" not in payload


def test_answer_renders_only_a_validated_evidence_pack(client: TestClient) -> None:
    pack = client.post(
        "/api/v1/retrieve",
        json={"query": "Zeige das Profil von Joshua Kimmich"},
    ).json()

    response = client.post("/api/v1/answer", json={"evidence_pack": pack})

    assert response.status_code == 200
    answer = response.json()
    assert answer["verdict"] == "limited"
    assert answer["cited_player_ids"] == ["5579"]
    assert "Evidence Quality Score" in answer["text"]
    assert "Einschränkungen" in answer["text"]


def test_out_of_scope_answer_abstains_without_player_citations(
    client: TestClient,
) -> None:
    pack = client.post(
        "/api/v1/retrieve",
        json={"query": "Wer gewinnt die nächste Weltmeisterschaft?"},
    ).json()

    answer = client.post("/api/v1/answer", json={"evidence_pack": pack}).json()

    assert answer["verdict"] == "out_of_scope"
    assert answer["cited_player_ids"] == []
    assert "keine Ergebnisvorhersagen" in answer["text"]


def test_api_limits_result_count_and_validates_query(client: TestClient) -> None:
    too_many = client.post(
        "/api/v1/retrieve",
        json={"query": "Bayern Spieler", "result_count": 6},
    )
    invalid_query = client.post("/api/v1/retrieve", json={"query": "x"})

    assert too_many.status_code == 422
    assert too_many.json()["detail"] == "result_count cannot exceed 5"
    assert invalid_query.status_code == 422


def test_missing_local_artifacts_return_actionable_503(tmp_path: Path) -> None:
    app = create_app(
        Settings(
            environment="test",
            profiles_path=tmp_path / "profiles.parquet",
            metric_evidence_path=tmp_path / "evidence.parquet",
            enable_dense_retrieval=False,
        )
    )

    with TestClient(app) as missing_client:
        response = missing_client.post(
            "/api/v1/retrieve",
            json={"query": "Zeige Joshua Kimmich"},
        )

    assert response.status_code == 503
    assert "scoutrag-data build" in response.json()["detail"]


def test_dashboard_assets_and_openapi_are_exposed(client: TestClient) -> None:
    dashboard = client.get("/")
    css = client.get("/assets/styles.css")
    javascript = client.get("/assets/app.js")
    openapi = client.get("/openapi.json").json()

    assert dashboard.status_code == 200
    assert "Evidence before eloquence" in dashboard.text
    assert "Recommendation Evidence Pack" in dashboard.text
    assert css.status_code == 200
    assert "--green: #b8ed73" in css.text
    assert javascript.status_code == 200
    assert 'const API_PREFIX = "/api/v1"' in javascript.text
    assert "/api/v1/retrieve" in openapi["paths"]
    assert "/api/v1/search" in openapi["paths"]
    assert "/api/v1/answer" in openapi["paths"]
