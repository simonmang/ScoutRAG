"""Build a source-linked fact allowlist from a Recommendation Evidence Pack."""

import re

from scoutrag.answering.models import AllowedFact, EvidenceFactCatalog
from scoutrag.domain.evidence import RecommendationEvidencePack

_UNSAFE_ID = re.compile(r"[^a-z0-9_.-]+")


def build_fact_catalog(pack: RecommendationEvidencePack) -> EvidenceFactCatalog:
    """Expose only stored player and metric values—never derived statistics."""
    facts: list[AllowedFact] = []
    for candidate in pack.candidates:
        profile = candidate.profile
        player_key = _fact_key(profile.player_id)
        profile_source = (
            f"profile:{profile.player_id}:{profile.competition_name}:{profile.season_name}"
        )
        profile_values: tuple[tuple[str, str, str], ...] = (
            ("name", "Player", profile.player_name),
            ("team", "Team", profile.team_name),
            ("competition", "Competition", profile.competition_name),
            ("season", "Season", profile.season_name),
            ("position", "Position group", profile.position_group.replace("_", " ")),
            ("minutes", "Minutes played", _number(profile.minutes_played)),
            ("data_quality", "Data quality", _number(profile.data_quality)),
        )
        for field_name, display_name, value in profile_values:
            facts.append(
                AllowedFact(
                    fact_id=f"player:{player_key}:{field_name}",
                    player_id=profile.player_id,
                    field_name=field_name,
                    display_name=display_name,
                    value=value,
                    source_reference=profile_source,
                )
            )

        for metric_index, metric in enumerate(pack.metric_evidence.get(profile.player_id, [])):
            metric_key = _fact_key(metric.metric_name)
            prefix = f"metric:{player_key}:{metric_key}:{metric_index}"
            metric_values: tuple[tuple[str, str, str | None], ...] = (
                ("raw", f"{metric.metric_name} raw value", _optional_number(metric.raw_value)),
                (
                    "normalized",
                    f"{metric.metric_name} normalized value",
                    _optional_number(metric.normalized_value),
                ),
                (
                    "percentile",
                    f"{metric.metric_name} percentile",
                    _optional_number(metric.percentile),
                ),
                ("comparison_group", "Comparison group", metric.comparison_group),
                ("sample_size", "Sample size", _optional_number(metric.sample_size)),
            )
            for field_name, display_name, metric_value in metric_values:
                if metric_value is None:
                    continue
                facts.append(
                    AllowedFact(
                        fact_id=f"{prefix}:{field_name}",
                        player_id=profile.player_id,
                        field_name=f"{metric.metric_name}.{field_name}",
                        display_name=display_name.replace("_", " "),
                        value=metric_value,
                        source_reference=metric.source_reference,
                    )
                )
    return EvidenceFactCatalog(facts=facts)


def _fact_key(value: str) -> str:
    return _UNSAFE_ID.sub("-", value.casefold()).strip("-") or "unknown"


def _number(value: float) -> str:
    return f"{value:g}"


def _optional_number(value: float | None) -> str | None:
    return None if value is None else _number(value)
