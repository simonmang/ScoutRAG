"""Orchestrate the complete, LLM-free Phase 2 data build."""

from pathlib import Path

from scoutrag.data.aggregation import aggregate_player_seasons
from scoutrag.data.minutes import calculate_match_participations
from scoutrag.data.models import MatchRecord, NormalizedEvent, PipelineResult
from scoutrag.data.normalization import normalize_events, normalize_match
from scoutrag.data.statsbomb import StatsBombOpenDataReader
from scoutrag.data.storage import ParquetDatasetWriter
from scoutrag.data.validation import validate_dataset


class Phase2DataPipeline:
    """Build one exact competition-season into auditable Parquet artifacts."""

    def __init__(self, writer: ParquetDatasetWriter | None = None) -> None:
        self.writer = writer or ParquetDatasetWriter()

    def run(
        self,
        input_root: Path,
        output_root: Path,
        *,
        competition_id: int,
        season_id: int,
    ) -> PipelineResult:
        reader = StatsBombOpenDataReader(input_root)
        competition = reader.competition(competition_id, season_id)
        raw_matches = reader.matches(competition_id, season_id)

        matches: list[MatchRecord] = []
        events: list[NormalizedEvent] = []
        participations = []
        for raw_match in raw_matches:
            match_id = int(raw_match["match_id"])
            match_events = normalize_events(
                reader.events(match_id),
                match_id=match_id,
                competition_id=competition_id,
                season_id=season_id,
            )
            match = normalize_match(raw_match, competition, match_events)
            match_participations = calculate_match_participations(
                reader.lineups(match_id),
                match,
                competition,
            )
            matches.append(match)
            events.extend(match_events)
            participations.extend(match_participations)

        aggregation = aggregate_player_seasons(
            competition,
            events,
            participations,
        )
        validation = validate_dataset(
            competition,
            matches,
            events,
            participations,
            aggregation.profiles,
            aggregation.evidence,
        )
        output_files = self.writer.write(
            output_root,
            competition=competition,
            matches=matches,
            events=events,
            participations=participations,
            profiles=aggregation.profiles,
            evidence=aggregation.evidence,
            validation=validation,
        )
        return PipelineResult(
            competition=competition,
            validation=validation,
            output_files=[str(path.resolve()) for path in output_files],
        )
