"""Generate a fully synthetic player dataset for Docker/CI smoke testing only.

Every name, statistic, and identifier below is invented. Nothing here is derived
from StatsBomb, API-Football, or any other licensed provider, so it carries none
of their usage restrictions and can be committed to a public repository and baked
into a public Docker image. It exists solely to let the container smoke test
(``.github/workflows/ci.yml``) and a cold ``docker run`` prove the retrieval and
governance pipeline actually works end to end, without any real football data or
API key. It is never presented as ScoutRAG's real dataset.

Run with: python scripts/build_synthetic_ci_fixture.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from scoutrag.domain.player import PlayerMetricEvidence, PlayerSeasonProfile

OUTPUT_ROOT = Path(__file__).resolve().parent.parent / "data" / "processed" / "synthetic-ci-fixture"

_PROFILES = [
    PlayerSeasonProfile(
        player_id="synthetic:1",
        profile_id="synthetic:demo:1",
        player_name="Alex Musterfeld",
        team_name="FC Beispielhausen",
        team_names=["FC Beispielhausen"],
        competition_name="Synthetic Demo League",
        season_name="2025/2026",
        position_group="defensive_midfield",
        minutes_played=1800.0,
        structured_features={"pressures_per_90": 18.4, "ball_recoveries_per_90": 9.1},
        percentiles={"pressures_per_90": 88.0, "ball_recoveries_per_90": 76.0},
        profile_text=(
            "Alex Musterfeld | FC Beispielhausen | Synthetic Demo League 2025/2026 | "
            "defensive_midfield | 1800.0 minutes | fully synthetic fixture data for "
            "container smoke tests only, not a real player."
        ),
        data_quality=0.9,
    ),
    PlayerSeasonProfile(
        player_id="synthetic:2",
        profile_id="synthetic:demo:2",
        player_name="Jamie Beispiel",
        team_name="SC Testheim",
        team_names=["SC Testheim"],
        competition_name="Synthetic Demo League",
        season_name="2025/2026",
        position_group="center_back",
        minutes_played=2100.0,
        structured_features={"duel_win_rate": 61.2, "interceptions_per_90": 2.4},
        percentiles={"duel_win_rate": 72.0, "interceptions_per_90": 65.0},
        profile_text=(
            "Jamie Beispiel | SC Testheim | Synthetic Demo League 2025/2026 | "
            "center_back | 2100.0 minutes | fully synthetic fixture data for "
            "container smoke tests only, not a real player."
        ),
        data_quality=0.9,
    ),
    PlayerSeasonProfile(
        player_id="synthetic:3",
        profile_id="synthetic:demo:3",
        player_name="Robin Platzhalter",
        team_name="FC Beispielhausen",
        team_names=["FC Beispielhausen"],
        competition_name="Synthetic Demo League",
        season_name="2025/2026",
        position_group="forward",
        minutes_played=1500.0,
        structured_features={"goals_per_90": 0.61, "shots_on_target_rate": 44.0},
        percentiles={"goals_per_90": 83.0, "shots_on_target_rate": 58.0},
        profile_text=(
            "Robin Platzhalter | FC Beispielhausen | Synthetic Demo League 2025/2026 | "
            "forward | 1500.0 minutes | fully synthetic fixture data for container "
            "smoke tests only, not a real player."
        ),
        data_quality=0.9,
    ),
]

_EVIDENCE = [
    PlayerMetricEvidence(
        player_id=profile.player_id,
        profile_id=profile.profile_id,
        season_id="synthetic:demo",
        metric_name=metric_name,
        raw_value=value,
        normalized_value=value,
        percentile=profile.percentiles.get(metric_name),
        comparison_group="Synthetic Demo League 2025/2026 fixture profiles",
        sample_size=3,
        source_reference="synthetic:fixture#no-external-source",
    )
    for profile in _PROFILES
    for metric_name, value in profile.structured_features.items()
]


def _profile_record(profile: PlayerSeasonProfile) -> dict[str, Any]:
    record = profile.model_dump(mode="json")
    record["structured_features_json"] = json.dumps(
        record.pop("structured_features"), sort_keys=True
    )
    record["percentiles_json"] = json.dumps(record.pop("percentiles"), sort_keys=True)
    return record


def _write_parquet(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(records)
    metadata = dict(table.schema.metadata or {})
    metadata[b"data_source"] = b"Synthetic fixture - no real provider data"
    table = table.replace_schema_metadata(metadata)
    pq.write_table(table, path, compression="zstd")


def main() -> None:
    _write_parquet(
        OUTPUT_ROOT / "player_season_profiles.parquet",
        [_profile_record(profile) for profile in _PROFILES],
    )
    _write_parquet(
        OUTPUT_ROOT / "player_metric_evidence.parquet",
        [item.model_dump(mode="json") for item in _EVIDENCE],
    )
    print(f"Wrote synthetic fixture to {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
