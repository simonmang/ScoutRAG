"""Orchestrate the complete, LLM-free Phase 3 data build."""

from pathlib import Path

from scoutrag.data.aggregation import aggregate_player_seasons
from scoutrag.data.feature_engineering import FeatureEngineeringConfig, engineer_player_features
from scoutrag.data.minutes import calculate_match_participations
from scoutrag.data.models import MatchRecord, NormalizedEvent, PipelineResult
from scoutrag.data.normalization import normalize_events, normalize_match
from scoutrag.data.statsbomb import StatsBombOpenDataReader
from scoutrag.data.storage import ParquetDatasetWriter
from scoutrag.data.validation import validate_dataset


class Phase3DataPipeline:
    """Build one competition-season into comparable, auditable feature artifacts."""

    def __init__(
        self,
        writer: ParquetDatasetWriter | None = None,
        feature_config: FeatureEngineeringConfig | None = None,
    ) -> None:
        self.writer = writer or ParquetDatasetWriter()
        self.feature_config = feature_config or FeatureEngineeringConfig()

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
        engineered = engineer_player_features(
            competition,
            matches,
            aggregation.profiles,
            aggregation.evidence,
            config=self.feature_config,
        )
        validation = validate_dataset(
            competition,
            matches,
            events,
            participations,
            engineered.profiles,
            engineered.evidence,
            engineered.definitions,
        )
        output_files = self.writer.write(
            output_root,
            competition=competition,
            matches=matches,
            events=events,
            participations=participations,
            profiles=engineered.profiles,
            evidence=engineered.evidence,
            definitions=engineered.definitions,
            validation=validation,
        )
        return PipelineResult(
            competition=competition,
            validation=validation,
            output_files=[str(path.resolve()) for path in output_files],
        )


# Kept as a source-compatible alias for callers created during Phase 2.
Phase2DataPipeline = Phase3DataPipeline
