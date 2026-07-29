"""Rule-based, inspectable query analysis for the retrieval MVP."""

import re
import unicodedata

from scoutrag.domain.player import PlayerSeasonProfile
from scoutrag.domain.query import QueryIntent, QueryProfile

POSITION_ALIASES: dict[str, tuple[str, ...]] = {
    "goalkeeper": ("torwart", "goalkeeper", "keeper"),
    "center_back": ("innenverteidiger", "center back", "centre back"),
    "fullback_wingback": (
        "außenverteidiger",
        "aussenverteidiger",
        "wingback",
        "wing back",
        "fullback",
    ),
    "defensive_midfield": (
        "sechser",
        "6er",
        "defensives mittelfeld",
        "defensive midfield",
        "holding midfielder",
    ),
    "central_midfield": (
        "achter",
        "8er",
        "zentrales mittelfeld",
        "central midfield",
    ),
    "attacking_midfield": (
        "zehner",
        "10er",
        "offensives mittelfeld",
        "attacking midfield",
    ),
    "winger": ("flügelspieler", "fluegelspieler", "winger", "außenstürmer"),
    "forward": ("stürmer", "stuermer", "striker", "forward", "mittelstürmer"),
}

TRAIT_METRICS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "pressing": (
        ("pressing", "pressingstark", "pressure"),
        ("pressures_per_90",),
    ),
    "ball_winning": (
        ("ballgewinn", "ballgewinne", "ball recovery", "ball recoveries"),
        ("ball_recoveries_per_90", "interceptions_per_90", "tackles_per_90"),
    ),
    "progressive_passing": (
        ("progressiv", "progressive pass", "progressives passspiel"),
        ("progressive_passes_per_90",),
    ),
    "passing": (
        ("passspiel", "passing", "pass completion", "passquote"),
        ("passes_per_90", "pass_completion_rate"),
    ),
    "carrying": (
        ("balltragen", "carry", "carries", "progressiver lauf"),
        ("progressive_carries_per_90",),
    ),
    "dribbling": (
        ("dribbling", "dribbelstark", "dribble"),
        ("dribbles_completed_per_90",),
    ),
    "shooting": (
        ("abschluss", "torgefahr", "shooting", "shots", "expected goals", "xg"),
        ("shots_per_90", "expected_goals_per_90"),
    ),
    "defending": (
        ("verteidigung", "defending", "defensivstark"),
        ("interceptions_per_90", "tackles_per_90", "clearances_per_90"),
    ),
}

OUT_OF_SCOPE_PATTERNS = (
    "wer gewinnt",
    "who will win",
    "vorhersage",
    "prediction",
    "wetter",
)

TEAM_ALIASES = {
    "bayern munich": (
        "bayern munich",
        "bayern münchen",
        "fc bayern münchen",
        "fc bayern",
    ),
}


class RuleBasedQueryAnalyzer:
    """Parse supported German and English scouting language without an LLM."""

    def __init__(self, profiles: list[PlayerSeasonProfile]) -> None:
        self.profiles = tuple(profiles)

    def analyze(self, query: str) -> QueryProfile:
        normalized = _normalize(query)
        named_players = sorted(
            {
                profile.player_name
                for profile in self.profiles
                if _normalize(profile.player_name) in normalized
            }
        )
        positions = [
            position
            for position, aliases in POSITION_ALIASES.items()
            if any(alias in normalized for alias in aliases)
        ]
        requested_traits: list[str] = []
        requested_metrics: list[str] = []
        for trait, (aliases, metrics) in TRAIT_METRICS.items():
            if any(alias in normalized for alias in aliases):
                requested_traits.append(trait)
                requested_metrics.extend(metrics)

        competitions = sorted(
            {
                profile.competition_name
                for profile in self.profiles
                if _normalize(profile.competition_name) in normalized
            }
        )
        teams = sorted(
            {
                team_name
                for profile in self.profiles
                for team_name in profile.team_names
                if any(alias in normalized for alias in _team_aliases(team_name))
            }
        )
        seasons = sorted(set(re.findall(r"\b20\d{2}/20\d{2}\b", normalized)))
        minimum_minutes = _minimum_minutes(normalized)
        result_count = _result_count(normalized)
        intent = _intent(normalized, named_players)
        return QueryProfile(
            original_query=query,
            normalized_query=normalized,
            intent=intent,
            requested_positions=positions,
            requested_traits=requested_traits,
            requested_metrics=list(dict.fromkeys(requested_metrics)),
            named_players=named_players,
            team_filters=teams,
            competition_filters=competitions,
            season_filters=seasons,
            minimum_minutes=minimum_minutes,
            result_count=result_count,
            expected_evidence_types=list(dict.fromkeys(requested_metrics)),
        )


def _normalize(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold().strip()
    return " ".join(normalized.split())


def _team_aliases(team_name: str) -> tuple[str, ...]:
    normalized = _normalize(team_name)
    return TEAM_ALIASES.get(normalized, (normalized,))


def _intent(normalized: str, named_players: list[str]) -> QueryIntent:
    if any(pattern in normalized for pattern in OUT_OF_SCOPE_PATTERNS):
        return QueryIntent.OUT_OF_SCOPE
    if ("vergleiche" in normalized or "compare" in normalized) and len(named_players) >= 2:
        return QueryIntent.PLAYER_COMPARISON
    if (
        "wie " in normalized or "similar" in normalized or "ähnlich" in normalized
    ) and named_players:
        return QueryIntent.SIMILAR_PLAYER
    if named_players and any(term in normalized for term in ("zeige", "profil", "show", "lookup")):
        return QueryIntent.EXACT_PLAYER_LOOKUP
    if any(term in normalized for term in ("meisten", "höchsten", "top ", "highest", "most ")):
        return QueryIntent.AGGREGATION
    return QueryIntent.PLAYER_DISCOVERY


def _minimum_minutes(normalized: str) -> int | None:
    pattern = r"(?:mindestens|minimum|min\.?|at least)\s*(\d{2,5})\s*min"
    match = re.search(pattern, normalized)
    return int(match.group(1)) if match else None


def _result_count(normalized: str) -> int:
    match = re.search(r"\b(?:top|beste[n]?|besten)\s*(\d{1,3})\b", normalized)
    if match:
        return min(max(int(match.group(1)), 1), 100)
    words = {"fünf": 5, "zehn": 10, "zwanzig": 20, "five": 5, "ten": 10}
    return next((count for word, count in words.items() if word in normalized), 10)
