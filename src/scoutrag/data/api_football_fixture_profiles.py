"""Aggregate API-Football fixture packages into canonical player-season evidence.

Unlike :mod:`scoutrag.data.api_football_profiles`, this module never uses the
season/team statistics blocks returned by ``/players``.  Performance data and
team membership come exclusively from the player groups embedded in historical
fixture packages.  Optional ``/players`` records contribute identity metadata
only.

API-Football's fixture-player schema uses a JSON ``null`` for many count fields
when the player recorded no such action.  For the explicitly enumerated action
count paths below, a present-but-null leaf is therefore aggregated as zero.  A
missing object or key is treated as unavailable and causes that season metric to
be withheld.  The number of these null-to-zero interpretations and the resulting
field coverage are retained as profile features and described in the result
limitations; ratings are never covered by this rule.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from typing import Any

from scoutrag.data.api_football_profiles import (
    API_FOOTBALL_METRICS,
    API_FOOTBALL_RATIOS,
    ApiFootballMetricSpec,
    ApiFootballProfileResult,
)
from scoutrag.data.position_inference import refine_position_group
from scoutrag.data.temporal import build_recent_form
from scoutrag.domain.player import (
    MetricDefinition,
    PlayerIdentity,
    PlayerMatchPerformance,
    PlayerMetricEvidence,
    PlayerSeasonProfile,
    PlayerTeamSeasonStint,
)

_PASSES_COMPLETED = ApiFootballMetricSpec(
    raw_metric="passes_completed",
    metric_name="passes_completed_per_90",
    display_name="Completed passes per 90",
    source_path=("passes", "accuracy"),
    description="Completed passes in fixture player statistics per 90 minutes.",
)
_FIXTURE_METRICS = (*API_FOOTBALL_METRICS, _PASSES_COMPLETED)
_COUNT_FEATURES: tuple[tuple[str, str, str], ...] = (
    ("appearances", "Appearances", "positive-minute fixture appearances"),
    ("starts", "Starts", "appearances where games.substitute is false"),
    (
        "substitute_appearances",
        "Substitute appearances",
        "appearances where games.substitute is true",
    ),
    (
        "captain_appearances",
        "Captain appearances",
        "appearances where games.captain is true",
    ),
)


@dataclass(frozen=True, slots=True)
class _Appearance:
    fixture_id: int
    match_date: date | None
    team_id: int | None
    team_name: str
    opponent_id: int | None
    opponent_name: str | None
    home_away: str | None
    minutes: float
    position: str
    substitute: bool | None
    captain: bool | None
    statistics: dict[str, Any]
    formation: str | None
    grid: str | None


@dataclass(frozen=True, slots=True)
class _FixturePlayerAggregate:
    player_id: int
    player_name: str
    player_record: dict[str, Any]
    appearances: tuple[_Appearance, ...]


def build_api_football_fixture_profiles(
    fixture_payloads: list[dict[str, Any]],
    *,
    league_id: int,
    season_start_year: int,
    competition_name: str,
    player_identity_payloads: list[dict[str, Any]] | None = None,
    minimum_minutes: float = 450,
    full_sample_minutes: float = 900,
    minimum_comparison_group_size: int = 3,
    comparison_scope: str | None = None,
    enable_percentiles: bool = True,
    round_prefix: str | None = "Regular Season",
    season_name: str | None = None,
    include_same_league_postseason: bool = False,
) -> ApiFootballProfileResult:
    """Build one exact league-season from rich ``/fixtures`` response objects.

    Duplicate fixture IDs are counted once.  A player is included only for
    fixture-player records reporting positive minutes.  Identity records may be
    raw ``/players`` response items, but their ``statistics`` values are ignored
    deliberately so a current team cannot contaminate historical membership.
    """

    _validate_settings(
        league_id,
        season_start_year,
        competition_name,
        minimum_minutes,
        full_sample_minutes,
        minimum_comparison_group_size,
    )
    fixtures, duplicate_count = _deduplicate_scoped_fixtures(
        fixture_payloads,
        league_id=league_id,
        season_start_year=season_start_year,
        round_prefix=round_prefix,
        include_same_league_postseason=include_same_league_postseason,
    )
    lineup_identity_mismatches = _lineup_identity_mismatch_count(fixtures)
    identities = _collect_identities(player_identity_payloads or [])
    aggregates = _collect_appearances(fixtures, identities)
    season_name = season_name or f"{season_start_year}/{season_start_year + 1}"
    season_id = f"api-football:{league_id}:{season_start_year}"

    prepared: list[PlayerSeasonProfile] = []
    completeness: dict[str, float] = {}
    null_zero_counts: dict[str, int] = {}
    fixture_ids_by_player: dict[str, tuple[int, ...]] = {}
    rating_minutes_by_player: dict[str, float] = {}
    for aggregate in aggregates:
        profile, metric_coverage, null_zero_count, rating_minutes = _prepare_profile(
            aggregate,
            competition_name=competition_name,
            season_name=season_name,
            season_start_year=season_start_year,
            profile_id=f"{season_id}:{aggregate.player_id}",
            full_sample_minutes=full_sample_minutes,
        )
        if profile is None:
            continue
        prepared.append(profile)
        completeness[profile.player_id] = metric_coverage
        null_zero_counts[profile.player_id] = null_zero_count
        fixture_ids_by_player[profile.player_id] = tuple(
            appearance.fixture_id for appearance in aggregate.appearances
        )
        rating_minutes_by_player[profile.player_id] = rating_minutes

    eligible_groups: defaultdict[str, list[PlayerSeasonProfile]] = defaultdict(list)
    for profile in prepared:
        if profile.minutes_played >= minimum_minutes:
            eligible_groups[profile.position_group].append(profile)

    scope_label = comparison_scope or f"{competition_name} {season_name}"
    percentile_specs = _percentile_specs()
    display_names = {name: label for name, label, _, _ in percentile_specs}
    profiles: list[PlayerSeasonProfile] = []
    evidence: list[PlayerMetricEvidence] = []
    for profile in prepared:
        peers = eligible_groups[profile.position_group]
        percentiles: dict[str, float] = {}
        if (
            enable_percentiles
            and profile.minutes_played >= minimum_minutes
            and len(peers) >= minimum_comparison_group_size
        ):
            for metric_name, _, higher_is_better, _ in percentile_specs:
                if metric_name not in profile.structured_features:
                    continue
                peer_values = [
                    peer.structured_features[metric_name]
                    for peer in peers
                    if metric_name in peer.structured_features
                ]
                if len(peer_values) >= minimum_comparison_group_size:
                    percentiles[metric_name] = _percentile(
                        profile.structured_features[metric_name],
                        peer_values,
                        higher_is_better=higher_is_better,
                    )

        minutes_score = min(profile.minutes_played / full_sample_minutes, 1)
        comparison_score = (
            min(len(peers) / minimum_comparison_group_size, 1) if enable_percentiles else 0
        )
        quality = round(
            (0.50 * minutes_score)
            + (0.35 * completeness[profile.player_id])
            + (0.15 * comparison_score),
            3,
        )
        summary_metrics = {
            name for name, _, _, summary_eligible in percentile_specs if summary_eligible
        }
        strongest = sorted(
            (
                (name, percentile)
                for name, percentile in percentiles.items()
                if name in summary_metrics
            ),
            key=lambda item: (-item[1], item[0]),
        )[:3]
        percentile_text = (
            " Highest position-group percentiles: "
            + ", ".join(
                f"{display_names[name]} P{percentile:.0f}" for name, percentile in strongest
            )
            + "."
            if strongest
            else (
                " No position percentile: the player or comparison group does not meet "
                "the configured evidence thresholds."
            )
        )
        enriched = profile.model_copy(
            update={
                "percentiles": percentiles,
                "data_quality": quality,
                "profile_text": (
                    f"{profile.player_name} | {' / '.join(profile.team_names)} | "
                    f"{competition_name} {season_name} | {profile.position_group} | "
                    f"{profile.minutes_played:.1f} minutes | "
                    f"API-Football fixture aggregation | Evidence Quality Score "
                    f"{quality:.3f}.{percentile_text}"
                ),
            }
        )
        profiles.append(enriched)
        comparison_group = (
            f"{scope_label} {profile.position_group} eligible fixture profiles "
            f"(n={len(peers)}, min_minutes={minimum_minutes:g}, "
            f"percentiles={'enabled' if enable_percentiles else 'disabled'})"
        )
        fixture_source = _fixture_source(fixture_ids_by_player[profile.player_id])
        evidence.extend(
            _profile_evidence(
                enriched,
                season_id=season_id,
                comparison_group=comparison_group,
                fixture_source=fixture_source,
                rating_minutes=rating_minutes_by_player[profile.player_id],
            )
        )

    identity_models = _build_player_identities(identities, aggregates)
    stints = _build_team_stints(
        aggregates,
        season_id=season_id,
        season_name=season_name,
        season_start_year=season_start_year,
        competition_name=competition_name,
        full_sample_minutes=full_sample_minutes,
    )
    match_performances = _build_match_performances(
        aggregates,
        season_id=season_id,
        season_name=season_name,
        competition_name=competition_name,
    )
    recent_forms = build_recent_form(match_performances)

    limitations = [
        (
            "Performance statistics and historical team membership were reconstructed "
            f"from {len(fixtures)} unique positive-scope fixture packages; optional "
            "/players statistics/team blocks were ignored."
        ),
        (
            "In fixture-player action-count fields, a present JSON null is interpreted "
            "as no recorded action (zero); a missing object/key withholds the season "
            "metric. Per-player null-to-zero counts and statistic coverage are retained "
            "as structured features."
        ),
        (
            "Average rating is weighted by reported match minutes and rating_minutes_coverage "
            "shows how much of the player's playing time supplied a rating."
        ),
    ]
    if duplicate_count:
        limitations.append(
            f"{duplicate_count} duplicate in-scope fixture package(s) were deduplicated by "
            "fixture ID before aggregation."
        )
    if lineup_identity_mismatches:
        limitations.append(
            f"{lineup_identity_mismatches} lineup/player-stat identity mismatch(es) were "
            "detected; fixtures.players remained the statistical identity authority."
        )
    if round_prefix is not None and include_same_league_postseason:
        limitations.append(
            f"Rounds beginning with {round_prefix!r} plus post-season fixtures between "
            "teams from that regular league phase were included; cross-league play-offs "
            "were excluded."
        )
    elif round_prefix is not None:
        limitations.append(f"Only fixture rounds beginning with {round_prefix!r} were included.")
    if not enable_percentiles:
        limitations.append(
            "Position-group percentiles were disabled because the supplied fixtures may not "
            "represent a complete competition comparison group."
        )
    elif comparison_scope is not None:
        limitations.append(f"Percentiles use only the declared comparison scope: {scope_label}.")
    return ApiFootballProfileResult(
        profiles=sorted(profiles, key=lambda item: (item.player_name.casefold(), item.player_id)),
        evidence=sorted(evidence, key=lambda item: (item.player_id, item.metric_name)),
        definitions=api_football_fixture_metric_definitions(),
        limitations=limitations,
        identities=identity_models,
        stints=stints,
        match_performances=match_performances,
        recent_forms=recent_forms,
    )


def api_football_fixture_metric_definitions() -> list[MetricDefinition]:
    """Document every evidence metric emitted by fixture aggregation."""

    definitions = [
        MetricDefinition(
            metric_name="minutes_played",
            display_name="Minutes played",
            description="Positive minutes summed across unique in-scope fixtures.",
            calculation_method="sum(fixtures.players.statistics.games.minutes)",
            required_event_types=["fixtures.players.statistics.games.minutes"],
            limitations=[
                "Only positive-minute appearances are included.",
                "Fixture IDs are deduplicated before aggregation.",
            ],
        ),
        MetricDefinition(
            metric_name="age_at_season_start",
            display_name="Age at season start",
            description="Player age on 1 July of the selected season start year.",
            calculation_method="completed years from identity birth date to 1 July",
            required_event_types=["players.player.birth.date"],
            limitations=[
                "Optional /players data is used only for identity metadata.",
                "This context value is not used for percentile ranking.",
            ],
        ),
        *[
            MetricDefinition(
                metric_name=name,
                display_name=display_name,
                description=f"Fixture-derived {display_name.lower()}.",
                calculation_method=calculation,
                required_event_types=["fixtures.players.statistics.games"],
                limitations=[
                    "Only positive-minute appearances are counted.",
                    "This context value is not used for percentile ranking.",
                ],
            )
            for name, display_name, calculation in _COUNT_FEATURES
        ],
        MetricDefinition(
            metric_name="captain_flag",
            display_name="Captain flag",
            description="Whether the player captained at least one included fixture.",
            calculation_method="1 when captain_appearances > 0, otherwise 0",
            required_event_types=["fixtures.players.statistics.games.captain"],
            limitations=["This context value is not used for percentile ranking."],
        ),
        MetricDefinition(
            metric_name="position_refined",
            display_name="Position refined",
            description=(
                "Whether position_group was narrowed from the provider's coarse "
                "goalkeeper/defender/midfielder/forward tag into a tactical sub-role "
                "(for example fullback_wingback or attacking_midfield) using lineup "
                "formation and grid slot data."
            ),
            calculation_method="1 when position_confidence > 0, otherwise 0",
            required_event_types=[
                "fixtures.lineups.formation",
                "fixtures.lineups.startXI.player.grid",
            ],
            limitations=[
                "Refinement is limited to back-four formations across the player's starts.",
                "This context value is not used for percentile ranking.",
            ],
        ),
        MetricDefinition(
            metric_name="position_confidence",
            display_name="Position confidence",
            description=(
                "Share of the player's grid-eligible starts agreeing with the majority "
                "refined tactical role; 0 when position_group stayed at the coarse tag."
            ),
            calculation_method=(
                "majority-role agreement across (formation, grid) observations from starts"
            ),
            required_event_types=[
                "fixtures.lineups.formation",
                "fixtures.lineups.startXI.player.grid",
            ],
            limitations=[
                "Requires at least five grid-eligible starts and 60% role agreement.",
                "This context value is not used for percentile ranking.",
            ],
        ),
        MetricDefinition(
            metric_name="average_rating",
            display_name="Average provider rating",
            description="API-Football match ratings weighted by reported match minutes.",
            calculation_method="sum(rating * rated_minutes) / sum(rated_minutes)",
            required_event_types=[
                "fixtures.players.statistics.games.rating",
                "fixtures.players.statistics.games.minutes",
            ],
            limitations=[
                "Matches without a rating are excluded from the weighted mean.",
                "Evidence sample_size reports rated minutes; rating_minutes_coverage "
                "is retained on the profile.",
                "Percentile desirability direction: higher values rank higher.",
            ],
        ),
    ]
    for spec in _FIXTURE_METRICS:
        definitions.append(
            MetricDefinition(
                metric_name=spec.metric_name,
                display_name=spec.display_name,
                description=spec.description,
                calculation_method=(
                    f"sum(fixtures.players.statistics.{'.'.join(spec.source_path)}) "
                    "/ minutes_played * 90"
                ),
                required_event_types=[
                    f"fixtures.players.statistics.{'.'.join(spec.source_path)}",
                    "fixtures.players.statistics.games.minutes",
                ],
                limitations=[
                    "A present null action count is interpreted as zero; a missing path "
                    "withholds this metric.",
                    "Per-90 volume does not adjust for team tactics, possession, or "
                    "opponent strength.",
                    (
                        "Percentile desirability direction: higher values rank higher."
                        if spec.higher_is_better
                        else "Percentile desirability direction: lower values rank higher."
                    ),
                ],
            )
        )
    for ratio_spec in API_FOOTBALL_RATIOS:
        definitions.append(
            MetricDefinition(
                metric_name=ratio_spec.metric_name,
                display_name=ratio_spec.display_name,
                description=ratio_spec.description,
                calculation_method=(
                    f"{ratio_spec.numerator_metric} / {ratio_spec.denominator_metric} * 100"
                ),
                required_event_types=[
                    ratio_spec.numerator_metric,
                    ratio_spec.denominator_metric,
                ],
                limitations=[
                    "The ratio is withheld when a required season count is unavailable "
                    "or its denominator is zero.",
                    "Percentile desirability direction: higher values rank higher.",
                ],
            )
        )
    definitions.append(
        MetricDefinition(
            metric_name="pass_completion_rate",
            display_name="Pass completion rate",
            description="Percentage of fixture passes recorded as completed.",
            calculation_method="passes_completed / passes * 100",
            required_event_types=["passes_completed", "passes"],
            limitations=[
                "In fixture-player payloads passes.accuracy is a completed-pass count, "
                "not a ready-made percentage.",
                "The rate is withheld when pass counts are unavailable or zero.",
                "Percentile desirability direction: higher values rank higher.",
            ],
        )
    )
    return definitions


def _deduplicate_scoped_fixtures(
    payloads: list[dict[str, Any]],
    *,
    league_id: int,
    season_start_year: int,
    round_prefix: str | None,
    include_same_league_postseason: bool,
) -> tuple[list[dict[str, Any]], int]:
    base_team_ids: set[int] = set()
    if round_prefix is not None and include_same_league_postseason:
        for payload in payloads:
            league = payload.get("league")
            if (
                not isinstance(league, dict)
                or _as_int(league.get("id")) != league_id
                or _as_int(league.get("season")) != season_start_year
                or not (_text_at(payload, ("league", "round")) or "").startswith(round_prefix)
            ):
                continue
            base_team_ids.update(_fixture_team_ids(payload))

    grouped: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    for payload in payloads:
        fixture = payload.get("fixture")
        league = payload.get("league")
        if not isinstance(fixture, dict) or not isinstance(league, dict):
            continue
        fixture_id = _as_int(fixture.get("id"))
        if (
            fixture_id is None
            or _as_int(league.get("id")) != league_id
            or _as_int(league.get("season")) != season_start_year
        ):
            continue
        round_matches = round_prefix is None or (
            _text_at(payload, ("league", "round")) or ""
        ).startswith(round_prefix)
        same_league_postseason = (
            include_same_league_postseason
            and len(base_team_ids) >= 2
            and (team_ids := _fixture_team_ids(payload))
            and team_ids.issubset(base_team_ids)
        )
        if not round_matches and not same_league_postseason:
            continue
        grouped[fixture_id].append(payload)

    selected: list[dict[str, Any]] = []
    duplicate_count = 0
    for fixture_id in sorted(grouped):
        variants = grouped[fixture_id]
        duplicate_count += len(variants) - 1
        selected.append(
            sorted(
                variants,
                key=lambda item: (
                    -_fixture_completeness(item),
                    json.dumps(item, ensure_ascii=False, sort_keys=True, default=str),
                ),
            )[0]
        )
    return selected, duplicate_count


def _fixture_team_ids(payload: dict[str, Any]) -> set[int]:
    teams = payload.get("teams")
    if not isinstance(teams, dict):
        return set()
    return {
        team_id
        for team in teams.values()
        if isinstance(team, dict) and (team_id := _as_int(team.get("id"))) is not None
    }


def _fixture_completeness(payload: dict[str, Any]) -> int:
    players = payload.get("players")
    if not isinstance(players, list):
        return 0
    return sum(
        len(group.get("players", []))
        for group in players
        if isinstance(group, dict) and isinstance(group.get("players"), list)
    )


def _lineup_identity_mismatch_count(fixtures: list[dict[str, Any]]) -> int:
    """Count roster and identity disagreements without altering player statistics."""

    mismatches = 0
    for payload in fixtures:
        statistical_names: dict[int, str] = {}
        groups = payload.get("players")
        if isinstance(groups, list):
            for group in groups:
                players = group.get("players") if isinstance(group, dict) else None
                if not isinstance(players, list):
                    continue
                for item in players:
                    player = item.get("player") if isinstance(item, dict) else None
                    if not isinstance(player, dict):
                        continue
                    player_id = _as_int(player.get("id"))
                    player_name = _display_player_name(player)
                    if player_id is not None and player_id > 0 and player_name is not None:
                        statistical_names[player_id] = player_name

        lineup_names: dict[int, str] = {}
        lineups = payload.get("lineups")
        if not isinstance(lineups, list):
            continue
        for lineup in lineups:
            if not isinstance(lineup, dict):
                continue
            for section in ("startXI", "substitutes"):
                lineup_players = lineup.get(section)
                if not isinstance(lineup_players, list):
                    continue
                for item in lineup_players:
                    player = item.get("player") if isinstance(item, dict) else None
                    if not isinstance(player, dict):
                        continue
                    player_id = _as_int(player.get("id"))
                    lineup_name = _display_player_name(player)
                    if player_id is not None and player_id > 0 and lineup_name is not None:
                        lineup_names[player_id] = lineup_name

        lineup_only = set(lineup_names).difference(statistical_names)
        statistics_only = set(statistical_names).difference(lineup_names)
        mismatches += max(len(lineup_only), len(statistics_only))
        for player_id in set(lineup_names).intersection(statistical_names):
            if not _names_compatible(
                lineup_names[player_id],
                statistical_names[player_id],
            ):
                mismatches += 1
    return mismatches


def _fixture_lineup_slots(payload: dict[str, Any]) -> dict[int, tuple[str, str]]:
    """Map starting player IDs to their (formation, grid) slot in this fixture.

    Only ``startXI`` entries carry a non-null grid; substitutes and unused
    squad members are deliberately excluded rather than guessed.
    """

    slots: dict[int, tuple[str, str]] = {}
    lineups = payload.get("lineups")
    if not isinstance(lineups, list):
        return slots
    for lineup in lineups:
        if not isinstance(lineup, dict):
            continue
        formation = _text_at(lineup, ("formation",))
        start_xi = lineup.get("startXI")
        if formation is None or not isinstance(start_xi, list):
            continue
        for item in start_xi:
            player = item.get("player") if isinstance(item, dict) else None
            if not isinstance(player, dict):
                continue
            player_id = _as_int(player.get("id"))
            grid = _text_at(player, ("grid",))
            if player_id is not None and player_id > 0 and grid is not None:
                slots[player_id] = (formation, grid)
    return slots


def _names_compatible(left: str, right: str) -> bool:
    """Accept common initial/full-name variants while surfacing wrong identities."""

    left_tokens = _name_tokens(left)
    right_tokens = _name_tokens(right)
    if left_tokens == right_tokens:
        return True
    if not left_tokens or not right_tokens:
        return False
    # Provider surfaces may reverse East-Asian names, omit a given name, or use
    # an initial.  A shared substantive token is sufficient because the numeric
    # player ID is already equal at this point.
    if {token for token in left_tokens if len(token) >= 3} & {
        token for token in right_tokens if len(token) >= 3
    }:
        return True
    return left_tokens[0][0] == right_tokens[0][0]


def _name_tokens(value: str) -> tuple[str, ...]:
    decomposed = unicodedata.normalize("NFKD", value.casefold())
    ascii_like = "".join(
        character for character in decomposed if not unicodedata.combining(character)
    )
    return tuple(re.findall(r"[^\W_]+", ascii_like, flags=re.UNICODE))


def _collect_identities(payloads: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    records: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in payloads:
        # Deliberately do not inspect item["statistics"].
        player = item.get("player")
        if not isinstance(player, dict):
            continue
        player_id = _as_int(player.get("id"))
        if player_id is not None and player_id > 0:
            records[player_id].append(player)
    return {
        player_id: sorted(
            candidates,
            key=lambda record: (
                -_identity_completeness(record),
                json.dumps(record, ensure_ascii=False, sort_keys=True, default=str),
            ),
        )[0]
        for player_id, candidates in records.items()
    }


def _identity_completeness(player: dict[str, Any]) -> int:
    return sum(
        value is not None
        for value in (
            _text_at(player, ("name",)),
            _text_at(player, ("firstname",)),
            _text_at(player, ("birth", "date")),
            _text_at(player, ("birth", "place")),
            _text_at(player, ("birth", "country")),
            _text_at(player, ("nationality",)),
            _text_at(player, ("height",)),
            _text_at(player, ("weight",)),
            _text_at(player, ("photo",)),
        )
    )


def _collect_appearances(
    fixtures: list[dict[str, Any]],
    identities: dict[int, dict[str, Any]],
) -> list[_FixturePlayerAggregate]:
    appearances: defaultdict[int, list[_Appearance]] = defaultdict(list)
    names: defaultdict[int, list[str]] = defaultdict(list)
    fixture_records: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    for payload in fixtures:
        fixture = payload["fixture"]
        fixture_id = _as_int(fixture.get("id"))
        if fixture_id is None:
            continue
        match_date = _parse_date(fixture.get("date"))
        lineup_slots = _fixture_lineup_slots(payload)
        player_groups = payload.get("players")
        if not isinstance(player_groups, list):
            continue
        seen_player_ids: set[int] = set()
        for group in player_groups:
            if not isinstance(group, dict):
                continue
            team = group.get("team")
            team_id = _as_int(team.get("id")) if isinstance(team, dict) else None
            team_name = _text_at(group, ("team", "name"))
            opponent_id, opponent_name, home_away = _fixture_opponent(
                payload,
                team_id=team_id,
                team_name=team_name,
            )
            group_players = group.get("players")
            if team_name is None or not isinstance(group_players, list):
                continue
            for item in group_players:
                if not isinstance(item, dict):
                    continue
                player = item.get("player")
                statistics = item.get("statistics")
                if not isinstance(player, dict) or not isinstance(statistics, list):
                    continue
                player_id = _as_int(player.get("id"))
                player_name = _display_player_name(player)
                if (
                    player_id is None
                    or player_id <= 0
                    or player_name is None
                    or player_id in seen_player_ids
                ):
                    continue
                valid_blocks = [
                    block
                    for block in statistics
                    if isinstance(block, dict)
                    and (_number_at(block, ("games", "minutes")) or 0) > 0
                ]
                if not valid_blocks:
                    continue
                block = sorted(
                    valid_blocks,
                    key=lambda value: (
                        -(_number_at(value, ("games", "minutes")) or 0),
                        json.dumps(value, ensure_ascii=False, sort_keys=True, default=str),
                    ),
                )[0]
                minutes = _number_at(block, ("games", "minutes"))
                if minutes is None or minutes <= 0:
                    continue
                seen_player_ids.add(player_id)
                names[player_id].append(player_name)
                fixture_records[player_id].append(player)
                formation, grid = lineup_slots.get(player_id, (None, None))
                appearances[player_id].append(
                    _Appearance(
                        fixture_id=fixture_id,
                        match_date=match_date,
                        team_id=team_id,
                        team_name=team_name,
                        opponent_id=opponent_id,
                        opponent_name=opponent_name,
                        home_away=home_away,
                        minutes=minutes,
                        position=_position_group(_text_at(block, ("games", "position"))),
                        substitute=_bool_at(block, ("games", "substitute")),
                        captain=_bool_at(block, ("games", "captain")),
                        statistics=block,
                        formation=formation,
                        grid=grid,
                    )
                )

    aggregates: list[_FixturePlayerAggregate] = []
    for player_id in sorted(appearances):
        identity = identities.get(player_id)
        records = fixture_records[player_id]
        player_record = identity or records[0]
        identity_name = _display_player_name(identity) if identity is not None else None
        if identity_name is not None:
            player_name = identity_name
        else:
            counts = Counter(names[player_id])
            player_name = sorted(
                counts,
                key=lambda value: (-counts[value], value.casefold(), value),
            )[0]
        aggregates.append(
            _FixturePlayerAggregate(
                player_id=player_id,
                player_name=player_name,
                player_record=player_record,
                appearances=tuple(sorted(appearances[player_id], key=lambda item: item.fixture_id)),
            )
        )
    return aggregates


def _fixture_opponent(
    payload: dict[str, Any],
    *,
    team_id: int | None,
    team_name: str | None,
) -> tuple[int | None, str | None, str | None]:
    teams = payload.get("teams")
    if not isinstance(teams, dict):
        return None, None, None
    home = teams.get("home")
    away = teams.get("away")
    if not isinstance(home, dict) or not isinstance(away, dict):
        return None, None, None
    home_id = _as_int(home.get("id"))
    away_id = _as_int(away.get("id"))
    home_name = _text_at(payload, ("teams", "home", "name"))
    away_name = _text_at(payload, ("teams", "away", "name"))
    if (team_id is not None and team_id == home_id) or (team_id is None and team_name == home_name):
        return away_id, away_name, "home"
    if (team_id is not None and team_id == away_id) or (team_id is None and team_name == away_name):
        return home_id, home_name, "away"
    return None, None, None


def _prepare_profile(
    aggregate: _FixturePlayerAggregate,
    *,
    competition_name: str,
    season_name: str,
    season_start_year: int,
    profile_id: str,
    full_sample_minutes: float,
) -> tuple[PlayerSeasonProfile | None, float, int, float]:
    if not aggregate.appearances:
        return None, 0, 0, 0
    minutes = round(sum(item.minutes for item in aggregate.appearances), 3)
    if minutes <= 0:
        return None, 0, 0, 0

    team_minutes: defaultdict[str, float] = defaultdict(float)
    position_minutes: defaultdict[str, float] = defaultdict(float)
    for appearance in aggregate.appearances:
        team_minutes[appearance.team_name] += appearance.minutes
        position_minutes[appearance.position] += appearance.minutes
    ordered_teams = sorted(
        team_minutes,
        key=lambda name: (-team_minutes[name], name.casefold(), name),
    )
    coarse_position_group = sorted(
        position_minutes,
        key=lambda position: (-position_minutes[position], position),
    )[0]
    position_observations = [
        (appearance.formation, appearance.grid)
        for appearance in aggregate.appearances
        if appearance.formation is not None and appearance.grid is not None
    ]
    position_group, position_confidence = refine_position_group(
        position_observations,
        coarse_group=coarse_position_group,
    )

    starts = sum(item.substitute is False for item in aggregate.appearances)
    substitute_appearances = sum(item.substitute is True for item in aggregate.appearances)
    captain_appearances = sum(item.captain is True for item in aggregate.appearances)
    features: dict[str, float] = {
        "teams_count": float(len(ordered_teams)),
        "appearances": float(len(aggregate.appearances)),
        "starts": float(starts),
        "substitute_appearances": float(substitute_appearances),
        "captain_appearances": float(captain_appearances),
        "captain_flag": float(captain_appearances > 0),
        "position_refined": float(position_confidence > 0),
        "position_confidence": position_confidence,
    }

    complete_metrics = 0
    null_zero_count = 0
    expected_observations = len(aggregate.appearances) * len(_FIXTURE_METRICS)
    available_observations = 0
    for spec in _FIXTURE_METRICS:
        values: list[float] = []
        metric_complete = True
        for appearance in aggregate.appearances:
            state, value = _action_count_at(appearance.statistics, spec.source_path)
            if state == "missing":
                metric_complete = False
                continue
            available_observations += 1
            if state == "null":
                null_zero_count += 1
                values.append(0)
            elif value is not None:
                values.append(value)
        if not metric_complete or len(values) != len(aggregate.appearances):
            continue
        complete_metrics += 1
        raw_value = round(sum(values), 4)
        features[spec.raw_metric] = raw_value
        features[spec.metric_name] = round((raw_value / minutes) * 90, 4)

    for ratio_spec in API_FOOTBALL_RATIOS:
        numerator = features.get(ratio_spec.numerator_metric)
        denominator = features.get(ratio_spec.denominator_metric)
        if numerator is not None and denominator is not None and denominator > 0:
            features[ratio_spec.metric_name] = round((numerator / denominator) * 100, 4)
    passes_completed = features.get("passes_completed")
    passes = features.get("passes")
    if passes_completed is not None and passes is not None and passes > 0:
        features["pass_completion_rate"] = round((passes_completed / passes) * 100, 4)

    rated = [
        (rating, appearance.minutes)
        for appearance in aggregate.appearances
        if (rating := _number_at(appearance.statistics, ("games", "rating"))) is not None
    ]
    rating_minutes = sum(item_minutes for _, item_minutes in rated)
    if rating_minutes > 0:
        features["average_rating"] = round(
            sum(rating * item_minutes for rating, item_minutes in rated) / rating_minutes,
            4,
        )
    features["rating_minutes_coverage"] = round(rating_minutes / minutes, 4)
    features["fixture_stat_coverage"] = round(
        available_observations / expected_observations if expected_observations else 0,
        4,
    )
    features["null_action_values_as_zero"] = float(null_zero_count)

    birth_date = _date_at(aggregate.player_record, ("birth", "date"))
    if birth_date is not None:
        features["age_at_season_start"] = float(_age_on(birth_date, date(season_start_year, 7, 1)))
    metric_coverage = (complete_metrics + features["rating_minutes_coverage"]) / (
        len(_FIXTURE_METRICS) + 1
    )
    provisional_quality = round(
        (0.6 * min(minutes / full_sample_minutes, 1)) + (0.4 * metric_coverage),
        3,
    )
    profile = PlayerSeasonProfile(
        player_id=f"api-football:{aggregate.player_id}",
        profile_id=profile_id,
        player_name=aggregate.player_name,
        date_of_birth=birth_date,
        birth_place=_text_at(aggregate.player_record, ("birth", "place")),
        birth_country=_text_at(aggregate.player_record, ("birth", "country")),
        nationality=_text_at(aggregate.player_record, ("nationality",)),
        height_cm=_measurement_at(aggregate.player_record, ("height",), unit="cm"),
        weight_kg=_measurement_at(aggregate.player_record, ("weight",), unit="kg"),
        photo_url=_text_at(aggregate.player_record, ("photo",)),
        team_name=ordered_teams[0],
        team_names=ordered_teams,
        competition_name=competition_name,
        season_name=season_name,
        position_group=position_group,
        minutes_played=minutes,
        structured_features=features,
        percentiles={},
        profile_text=(
            f"{aggregate.player_name} | {' / '.join(ordered_teams)} | "
            f"{competition_name} {season_name} | {position_group} | {minutes:.1f} minutes | "
            "API-Football fixture aggregation."
        ),
        data_quality=provisional_quality,
    )
    return profile, metric_coverage, null_zero_count, rating_minutes


def _build_player_identities(
    identity_records: dict[int, dict[str, Any]],
    aggregates: list[_FixturePlayerAggregate],
) -> list[PlayerIdentity]:
    records = dict(identity_records)
    for aggregate in aggregates:
        records.setdefault(aggregate.player_id, aggregate.player_record)
    identities: list[PlayerIdentity] = []
    for player_id, record in records.items():
        player_name = _display_player_name(record)
        if player_id <= 0 or player_name is None:
            continue
        identities.append(
            PlayerIdentity(
                player_id=f"api-football:{player_id}",
                player_name=player_name,
                date_of_birth=_date_at(record, ("birth", "date")),
                birth_place=_text_at(record, ("birth", "place")),
                birth_country=_text_at(record, ("birth", "country")),
                nationality=_text_at(record, ("nationality",)),
                height_cm=_measurement_at(record, ("height",), unit="cm"),
                weight_kg=_measurement_at(record, ("weight",), unit="kg"),
                photo_url=_text_at(record, ("photo",)),
                source_reference=(
                    "api-football:/players#player"
                    if player_id in identity_records
                    else "api-football:/fixtures?ids#players.player"
                ),
            )
        )
    return sorted(
        identities,
        key=lambda item: (item.player_name.casefold(), item.player_id),
    )


def _build_team_stints(
    aggregates: list[_FixturePlayerAggregate],
    *,
    season_id: str,
    season_name: str,
    season_start_year: int,
    competition_name: str,
    full_sample_minutes: float,
) -> list[PlayerTeamSeasonStint]:
    stints: list[PlayerTeamSeasonStint] = []
    for aggregate in aggregates:
        grouped: defaultdict[tuple[int | None, str], list[_Appearance]] = defaultdict(list)
        for appearance in aggregate.appearances:
            grouped[(appearance.team_id, appearance.team_name)].append(appearance)
        for (team_id, team_name), appearances in grouped.items():
            profile_id = f"{season_id}:{aggregate.player_id}"
            stint_profile, _, _, _ = _prepare_profile(
                _FixturePlayerAggregate(
                    player_id=aggregate.player_id,
                    player_name=aggregate.player_name,
                    player_record=aggregate.player_record,
                    appearances=tuple(appearances),
                ),
                competition_name=competition_name,
                season_name=season_name,
                season_start_year=season_start_year,
                profile_id=profile_id,
                full_sample_minutes=full_sample_minutes,
            )
            if stint_profile is None:
                continue
            team_key = str(team_id) if team_id is not None else _identifier_fragment(team_name)
            fixture_ids = tuple(item.fixture_id for item in appearances)
            stints.append(
                PlayerTeamSeasonStint(
                    stint_id=f"{profile_id}:team:{team_key}",
                    player_id=stint_profile.player_id,
                    profile_id=profile_id,
                    season_id=season_id,
                    season_name=season_name,
                    competition_name=competition_name,
                    team_id=team_id,
                    team_name=team_name,
                    position_group=stint_profile.position_group,
                    minutes_played=stint_profile.minutes_played,
                    appearances=len(appearances),
                    structured_features=stint_profile.structured_features,
                    data_quality=stint_profile.data_quality,
                    source_reference=_fixture_source(fixture_ids),
                )
            )
    return sorted(stints, key=lambda item: (item.stint_id, item.player_id))


def _build_match_performances(
    aggregates: list[_FixturePlayerAggregate],
    *,
    season_id: str,
    season_name: str,
    competition_name: str,
) -> list[PlayerMatchPerformance]:
    performances: list[PlayerMatchPerformance] = []
    for aggregate in aggregates:
        profile_id = f"{season_id}:{aggregate.player_id}"
        player_id = f"api-football:{aggregate.player_id}"
        for appearance in aggregate.appearances:
            features, coverage = _match_features(
                appearance.statistics,
                minutes=appearance.minutes,
            )
            performances.append(
                PlayerMatchPerformance(
                    performance_id=f"{profile_id}:fixture:{appearance.fixture_id}",
                    player_id=player_id,
                    profile_id=profile_id,
                    season_id=season_id,
                    season_name=season_name,
                    competition_name=competition_name,
                    fixture_id=appearance.fixture_id,
                    match_date=appearance.match_date,
                    team_id=appearance.team_id,
                    team_name=appearance.team_name,
                    opponent_id=appearance.opponent_id,
                    opponent_name=appearance.opponent_name,
                    home_away=appearance.home_away,
                    position_group=appearance.position,
                    minutes_played=appearance.minutes,
                    started=(
                        not appearance.substitute if appearance.substitute is not None else None
                    ),
                    substitute=appearance.substitute,
                    captain=appearance.captain,
                    structured_features=features,
                    data_quality=round(
                        (0.7 * min(appearance.minutes / 90, 1)) + (0.3 * coverage),
                        3,
                    ),
                    source_reference=(
                        f"api-football:/fixtures?ids={appearance.fixture_id}#players.statistics"
                    ),
                )
            )
    return sorted(
        performances,
        key=lambda item: (
            item.match_date or date.min,
            item.fixture_id,
            item.player_id,
        ),
    )


def _match_features(
    statistics: dict[str, Any],
    *,
    minutes: float,
) -> tuple[dict[str, float], float]:
    features: dict[str, float] = {}
    available = 0
    for spec in _FIXTURE_METRICS:
        state, value = _action_count_at(statistics, spec.source_path)
        if state == "missing":
            continue
        available += 1
        raw_value = 0 if state == "null" else value
        if raw_value is None:
            continue
        features[spec.raw_metric] = round(raw_value, 4)
        features[spec.metric_name] = round((raw_value / minutes) * 90, 4)
    rating = _number_at(statistics, ("games", "rating"))
    if rating is not None:
        features["rating"] = rating
    for ratio_spec in API_FOOTBALL_RATIOS:
        numerator = features.get(ratio_spec.numerator_metric)
        denominator = features.get(ratio_spec.denominator_metric)
        if numerator is not None and denominator is not None and denominator > 0:
            features[ratio_spec.metric_name] = round(
                (numerator / denominator) * 100,
                4,
            )
    completed = features.get("passes_completed")
    attempted = features.get("passes")
    if completed is not None and attempted is not None and attempted > 0:
        features["pass_completion_rate"] = round(
            (completed / attempted) * 100,
            4,
        )
    return features, available / len(_FIXTURE_METRICS)


def _identifier_fragment(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value.casefold())
    ascii_like = "".join(
        character for character in normalized if not unicodedata.combining(character)
    )
    return re.sub(r"[^a-z0-9]+", "-", ascii_like).strip("-") or "unknown"


def _profile_evidence(
    profile: PlayerSeasonProfile,
    *,
    season_id: str,
    comparison_group: str,
    fixture_source: str,
    rating_minutes: float,
) -> list[PlayerMetricEvidence]:
    evidence = [
        PlayerMetricEvidence(
            player_id=profile.player_id,
            season_id=season_id,
            metric_name="minutes_played",
            raw_value=profile.minutes_played,
            normalized_value=None,
            percentile=None,
            comparison_group=comparison_group,
            sample_size=profile.minutes_played,
            source_reference=f"{fixture_source}#players.statistics.games.minutes",
        )
    ]
    features = profile.structured_features
    for metric_name, _, _ in _COUNT_FEATURES:
        evidence.append(
            PlayerMetricEvidence(
                player_id=profile.player_id,
                season_id=season_id,
                metric_name=metric_name,
                raw_value=features[metric_name],
                normalized_value=None,
                percentile=None,
                comparison_group=comparison_group,
                sample_size=profile.minutes_played,
                source_reference=f"{fixture_source}#players.statistics.games",
            )
        )
    evidence.append(
        PlayerMetricEvidence(
            player_id=profile.player_id,
            season_id=season_id,
            metric_name="captain_flag",
            raw_value=features["captain_flag"],
            normalized_value=None,
            percentile=None,
            comparison_group=comparison_group,
            sample_size=features["appearances"],
            source_reference=f"{fixture_source}#players.statistics.games.captain",
        )
    )
    if "age_at_season_start" in features:
        evidence.append(
            PlayerMetricEvidence(
                player_id=profile.player_id,
                season_id=season_id,
                metric_name="age_at_season_start",
                raw_value=features["age_at_season_start"],
                normalized_value=None,
                percentile=None,
                comparison_group=comparison_group,
                sample_size=None,
                source_reference="api-football:/players#player.birth.date",
            )
        )
    if "average_rating" in features:
        evidence.append(
            PlayerMetricEvidence(
                player_id=profile.player_id,
                season_id=season_id,
                metric_name="average_rating",
                raw_value=features["average_rating"],
                normalized_value=features["average_rating"],
                percentile=profile.percentiles.get("average_rating"),
                comparison_group=comparison_group,
                sample_size=rating_minutes,
                source_reference=f"{fixture_source}#players.statistics.games.rating",
            )
        )
    for spec in _FIXTURE_METRICS:
        if spec.raw_metric not in features:
            continue
        evidence.append(
            PlayerMetricEvidence(
                player_id=profile.player_id,
                season_id=season_id,
                metric_name=spec.metric_name,
                raw_value=features[spec.raw_metric],
                normalized_value=features[spec.metric_name],
                percentile=profile.percentiles.get(spec.metric_name),
                comparison_group=comparison_group,
                sample_size=profile.minutes_played,
                source_reference=(
                    f"{fixture_source}#players.statistics.{'.'.join(spec.source_path)}"
                ),
            )
        )
    for ratio_spec in API_FOOTBALL_RATIOS:
        if ratio_spec.metric_name not in features:
            continue
        evidence.append(
            PlayerMetricEvidence(
                player_id=profile.player_id,
                season_id=season_id,
                metric_name=ratio_spec.metric_name,
                raw_value=None,
                normalized_value=features[ratio_spec.metric_name],
                percentile=profile.percentiles.get(ratio_spec.metric_name),
                comparison_group=comparison_group,
                sample_size=features.get(ratio_spec.denominator_metric),
                source_reference=(
                    f"{fixture_source}#derived."
                    f"{ratio_spec.numerator_metric}/{ratio_spec.denominator_metric}"
                ),
            )
        )
    if "pass_completion_rate" in features:
        evidence.append(
            PlayerMetricEvidence(
                player_id=profile.player_id,
                season_id=season_id,
                metric_name="pass_completion_rate",
                raw_value=None,
                normalized_value=features["pass_completion_rate"],
                percentile=profile.percentiles.get("pass_completion_rate"),
                comparison_group=comparison_group,
                sample_size=features.get("passes"),
                source_reference=f"{fixture_source}#derived.passes.accuracy/passes.total",
            )
        )
    return [item.model_copy(update={"profile_id": profile.profile_id}) for item in evidence]


def _percentile_specs() -> list[tuple[str, str, bool, bool]]:
    return [
        *[
            (spec.metric_name, spec.display_name, spec.higher_is_better, spec.summary_eligible)
            for spec in _FIXTURE_METRICS
        ],
        *[(spec.metric_name, spec.display_name, True, True) for spec in API_FOOTBALL_RATIOS],
        ("pass_completion_rate", "Pass completion rate", True, True),
        ("average_rating", "Average provider rating", True, True),
    ]


def _fixture_source(fixture_ids: tuple[int, ...]) -> str:
    return "api-football:/fixtures?ids=" + "-".join(str(value) for value in fixture_ids)


def _action_count_at(
    value: dict[str, Any],
    path: tuple[str, ...],
) -> tuple[str, float | None]:
    current: Any = value
    for part in path[:-1]:
        if not isinstance(current, dict) or part not in current:
            return "missing", None
        current = current[part]
    if not isinstance(current, dict) or path[-1] not in current:
        return "missing", None
    leaf = current[path[-1]]
    if leaf is None:
        return "null", 0
    number = _as_number(leaf)
    return ("value", number) if number is not None else ("missing", None)


def _value_at(value: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = value
    for part in path:
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _number_at(value: dict[str, Any], path: tuple[str, ...]) -> float | None:
    return _as_number(_value_at(value, path))


def _as_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        number = float(value)
    elif isinstance(value, str):
        try:
            number = float(value.strip().replace(",", "."))
        except ValueError:
            return None
    else:
        return None
    return number if math.isfinite(number) and number >= 0 else None


def _as_int(value: Any) -> int | None:
    number = _as_number(value)
    if number is None or not number.is_integer():
        return None
    return int(number)


def _text_at(value: dict[str, Any], path: tuple[str, ...]) -> str | None:
    current = _value_at(value, path)
    if not isinstance(current, str) or not current.strip():
        return None
    return current.strip()


def _bool_at(value: dict[str, Any], path: tuple[str, ...]) -> bool | None:
    current = _value_at(value, path)
    return current if isinstance(current, bool) else None


def _display_player_name(player: dict[str, Any] | None) -> str | None:
    if player is None:
        return None
    name = _text_at(player, ("name",))
    if name is None:
        return None
    abbreviated = re.fullmatch(r"[^\W\d_]\.\s+(.+)", name, flags=re.UNICODE)
    firstname = _text_at(player, ("firstname",))
    if abbreviated is None or firstname is None:
        return name
    return f"{firstname.split(maxsplit=1)[0]} {abbreviated.group(1).strip()}"


def _parse_date(value: Any) -> date | None:
    if not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def _date_at(value: dict[str, Any], path: tuple[str, ...]) -> date | None:
    return _parse_date(_value_at(value, path))


def _measurement_at(
    value: dict[str, Any],
    path: tuple[str, ...],
    *,
    unit: str,
) -> float | None:
    raw = _value_at(value, path)
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int | float):
        number = float(raw)
    elif isinstance(raw, str):
        match = re.fullmatch(
            rf"\s*(\d+(?:[.,]\d+)?)\s*(?:{re.escape(unit)})?\s*",
            raw,
            flags=re.IGNORECASE,
        )
        if match is None:
            return None
        number = float(match.group(1).replace(",", "."))
    else:
        return None
    return number if math.isfinite(number) and number > 0 else None


def _position_group(position: str | None) -> str:
    if position is None:
        return "unknown"
    normalized = position.casefold().strip()
    aliases = {
        "g": "goalkeeper",
        "goalkeeper": "goalkeeper",
        "d": "defender",
        "defender": "defender",
        "m": "midfielder",
        "midfielder": "midfielder",
        "midfield": "midfielder",
        "f": "forward",
        "attacker": "forward",
        "forward": "forward",
    }
    return aliases.get(normalized, normalized.replace(" ", "_"))


def _age_on(birth_date: date, reference_date: date) -> int:
    return (
        reference_date.year
        - birth_date.year
        - ((reference_date.month, reference_date.day) < (birth_date.month, birth_date.day))
    )


def _percentile(
    value: float,
    values: list[float],
    *,
    higher_is_better: bool,
) -> float:
    if len(values) == 1:
        return 50
    lower = sum(item < value for item in values)
    equal_others = sum(item == value for item in values) - 1
    raw = ((lower + (0.5 * equal_others)) / (len(values) - 1)) * 100
    return round(raw if higher_is_better else 100 - raw, 1)


def _validate_settings(
    league_id: int,
    season_start_year: int,
    competition_name: str,
    minimum_minutes: float,
    full_sample_minutes: float,
    minimum_comparison_group_size: int,
) -> None:
    if league_id <= 0:
        raise ValueError("league_id must be positive")
    if season_start_year < 1900:
        raise ValueError("season_start_year must be a four-digit year")
    if not competition_name.strip():
        raise ValueError("competition_name must not be empty")
    if minimum_minutes < 0:
        raise ValueError("minimum_minutes must be non-negative")
    if full_sample_minutes <= 0:
        raise ValueError("full_sample_minutes must be positive")
    if minimum_comparison_group_size < 2:
        raise ValueError("minimum_comparison_group_size must be at least 2")
