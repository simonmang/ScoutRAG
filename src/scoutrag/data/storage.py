"""Parquet persistence and reproducibility manifest creation."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from scoutrag.data.models import (
    CompetitionSeason,
    DataValidationReport,
    MatchRecord,
    NormalizedEvent,
    PlayerMatchParticipation,
)
from scoutrag.domain.player import PlayerMetricEvidence, PlayerSeasonProfile


def _write_parquet(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        raise ValueError(f"cannot write empty Parquet dataset: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(records)
    metadata = dict(table.schema.metadata or {})
    metadata[b"data_source"] = b"StatsBomb Open Data"
    metadata[b"scoutrag_phase"] = b"2"
    table = table.replace_schema_metadata(metadata)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    pq.write_table(table, temporary_path, compression="zstd")
    temporary_path.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ParquetDatasetWriter:
    """Write typed artifacts without allowing dynamic profile maps into schemas."""

    def write(
        self,
        output_root: Path,
        *,
        competition: CompetitionSeason,
        matches: list[MatchRecord],
        events: list[NormalizedEvent],
        participations: list[PlayerMatchParticipation],
        profiles: list[PlayerSeasonProfile],
        evidence: list[PlayerMetricEvidence],
        validation: DataValidationReport,
    ) -> list[Path]:
        output_root.mkdir(parents=True, exist_ok=True)
        files: dict[str, list[dict[str, Any]]] = {
            "matches.parquet": [item.model_dump(mode="json") for item in matches],
            "events.parquet": [item.model_dump(mode="json") for item in events],
            "player_match_minutes.parquet": [
                item.model_dump(mode="json") for item in participations
            ],
            "player_season_profiles.parquet": [self._profile_record(item) for item in profiles],
            "player_metric_evidence.parquet": [item.model_dump(mode="json") for item in evidence],
        }

        written_paths: list[Path] = []
        for name, records in files.items():
            path = output_root / name
            _write_parquet(path, records)
            written_paths.append(path)

        validation_path = output_root / "validation_report.json"
        validation_path.write_text(
            validation.model_dump_json(indent=2),
            encoding="utf-8",
        )
        written_paths.append(validation_path)

        manifest_path = output_root / "manifest.json"
        manifest = {
            "schema_version": "phase2-v1",
            "generated_at": datetime.now(UTC).isoformat(),
            "source": {
                "provider": "StatsBomb Open Data",
                "repository": "https://github.com/statsbomb/open-data",
                "competition_id": competition.competition_id,
                "season_id": competition.season_id,
            },
            "artifacts": {
                path.name: {
                    "sha256": _sha256(path),
                    "bytes": path.stat().st_size,
                }
                for path in written_paths
            },
        }
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        written_paths.append(manifest_path)
        return written_paths

    @staticmethod
    def _profile_record(profile: PlayerSeasonProfile) -> dict[str, Any]:
        record = profile.model_dump(mode="json")
        record["structured_features_json"] = json.dumps(
            record.pop("structured_features"),
            ensure_ascii=False,
            sort_keys=True,
        )
        record["percentiles_json"] = json.dumps(
            record.pop("percentiles"),
            ensure_ascii=False,
            sort_keys=True,
        )
        return record
