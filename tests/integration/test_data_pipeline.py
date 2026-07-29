"""End-to-end Phase 2 build from StatsBomb JSON to Parquet."""

import json
from pathlib import Path
from typing import Any, cast

import pyarrow.parquet as pq  # type: ignore[import-untyped]

from scoutrag.data.pipeline import Phase2DataPipeline

FIXTURE_ROOT = Path(__file__).parents[1] / "fixtures" / "statsbomb"


def test_phase2_pipeline_builds_valid_auditable_artifacts(tmp_path: Path) -> None:
    result = Phase2DataPipeline().run(
        FIXTURE_ROOT,
        tmp_path,
        competition_id=9,
        season_id=281,
    )

    assert result.validation.valid is True
    assert result.validation.match_count == 1
    assert result.validation.event_count == 8
    assert result.validation.profile_count == 4
    assert len(result.output_files) == 7

    profile_table = pq.read_table(tmp_path / "player_season_profiles.parquet")
    profiles = profile_table.to_pylist()
    ada = next(profile for profile in profiles if profile["player_id"] == "1")
    features = cast(
        dict[str, float],
        json.loads(ada["structured_features_json"]),
    )
    assert ada["minutes_played"] == 90
    assert ada["position_group"] == "midfield"
    assert ada["team_names"] == ["Alpha FC"]
    assert features["passes"] == 1
    assert features["pressures"] == 1
    assert "per_90" not in ada["structured_features_json"]

    evidence_table = pq.read_table(tmp_path / "player_metric_evidence.parquet")
    assert evidence_table.num_rows == result.validation.evidence_count

    report = cast(
        dict[str, Any],
        json.loads((tmp_path / "validation_report.json").read_text("utf-8")),
    )
    manifest = cast(
        dict[str, Any],
        json.loads((tmp_path / "manifest.json").read_text("utf-8")),
    )
    assert report["valid"] is True
    assert manifest["schema_version"] == "phase2-v1"
    assert manifest["source"]["provider"] == "StatsBomb Open Data"
    assert "events.parquet" in manifest["artifacts"]
