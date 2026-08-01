"""Rule-based, inspectable query analysis for the retrieval MVP."""

import re
import unicodedata

from scoutrag.domain.player import PlayerSeasonProfile
from scoutrag.domain.query import QueryIntent, QueryProfile, TemporalScope

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
        (
            "abschluss",
            "torgefahr",
            "torgefährlich",
            "shooting",
            "shots",
            "expected goals",
            "xg",
        ),
        ("shots_per_90", "expected_goals_per_90"),
    ),
    "creativity": (
        (
            "expected assists",
            "expected assist",
            "erwartete assists",
            "kreativ",
            "chance creation",
        ),
        ("expected_assists_per_90",),
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


# Keyed by _normalize(profile.team_name) exactly, since _team_aliases() looks up aliases by
# that normalized real name, not by any of the alias phrasings themselves. A key that doesn't
# match _normalize(team_name) verbatim (e.g. an English "munich" for the real "münchen") is
# silently unreachable - _team_aliases() just falls back to the single exact name instead.
TEAM_ALIASES = {
    "bayern münchen": (
        "bayern",
        "bayern munich",
        "bayern münchen",
        "bayern muenchen",
        "fc bayern münchen",
        "fc bayern muenchen",
        "fc bayern",
    ),
}

# Canonical keys match scoutrag.domain.player.PlayerSeasonProfile.competition_name exactly, as
# produced by the 24-competition scouting universe catalog (config/scouting_leagues.json) plus
# the optional StatsBomb pipeline's "1. Bundesliga". Aliases add common German/English phrasings
# and ASCII umlaut substitutes; _matches_any does a bidirectional substring match, so an alias
# only needs to disambiguate query wording, not restate the canonical name.
COMPETITION_ALIASES: dict[str, tuple[str, ...]] = {
    "1. Bundesliga": ("1. bundesliga",),
    "2. Bundesliga": ("2. bundesliga", "2 bundesliga", "zweite bundesliga"),
    "Premier League": ("premier league",),
    "Championship": ("championship",),
    "La Liga": ("la liga", "laliga", "primera division"),
    "Segunda Division": ("segunda division", "segunda división", "la liga 2"),
    "Serie A": ("serie a",),
    "Serie B": ("serie b",),
    "Ligue 1": ("ligue 1",),
    "Ligue 2": ("ligue 2",),
    "Eredivisie": ("eredivisie",),
    "Eerste Divisie": ("eerste divisie",),
    "Primeira Liga": ("primeira liga", "liga portugal"),
    "Jupiler Pro League": ("jupiler pro league", "belgian pro league"),
    "Challenger Pro League": ("challenger pro league",),
    "Super Lig": ("süper lig", "sueper lig", "super lig", "türkische liga", "tuerkische liga"),
    "Turkish 1. Lig": ("turkish 1. lig", "1. lig", "tff 1. lig"),
    "Swiss Super League": ("swiss super league", "schweizer super league"),
    "Austrian Bundesliga": (
        "austrian bundesliga",
        "österreichische bundesliga",
        "oesterreichische bundesliga",
    ),
    "Scottish Premiership": ("scottish premiership",),
    "Danish Superliga": ("danish superliga", "superliga"),
    "Allsvenskan": ("allsvenskan",),
    "Superettan": ("superettan",),
    "Eliteserien": ("eliteserien",),
}


class RuleBasedQueryAnalyzer:
    """Parse supported German and English scouting language without an LLM."""

    def __init__(self, profiles: list[PlayerSeasonProfile]) -> None:
        self.profiles = tuple(profiles)

    def analyze(self, query: str) -> QueryProfile:
        normalized = _normalize(query)
        name_search_text = _name_search_text(normalized)
        named_players = sorted(
            {
                profile.player_name
                for profile in self.profiles
                if _normalize(profile.player_name) in name_search_text
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

        competitions = {
            competition_name
            for competition_name, aliases in COMPETITION_ALIASES.items()
            if any(alias in normalized for alias in aliases)
        }
        competitions.update(
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
        seasons = _seasons(normalized)
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
            competition_filters=sorted(competitions),
            season_filters=seasons,
            minimum_minutes=minimum_minutes,
            result_count=result_count,
            expected_evidence_types=list(dict.fromkeys(requested_metrics)),
            temporal_scope=_temporal_scope(normalized, seasons, self.profiles),
        )


def _normalize(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold().strip()
    return " ".join(normalized.split())


# The exact lookup phrasing recognized by _intent() below ("zeige ... das
# profil von", "show (me) the profile of", "show me") coincidentally
# collides with at least one real, short player name ("Show", a real
# API-Football profile). Stripping only these specific multi-word command
# constructions - not the bare word alone - lets "Show the profile of X"
# resolve to X without also making a player genuinely named "Show"
# permanently unmatchable in every other phrasing (for example "Compare
# Show and X"). _intent() itself still inspects the unstripped, original
# normalized text.
_NAME_MATCH_STOPPHRASE_PATTERN = re.compile(
    r"\bzeige(?:\s+mir)?\s+das\s+profil\s+von\b"
    r"|\bshow\s+(?:me\s+)?the\s+profile\s+of\b"
    r"|\bshow\s+me\b"
)


def _name_search_text(normalized: str) -> str:
    return _NAME_MATCH_STOPPHRASE_PATTERN.sub(" ", normalized)


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


def _seasons(normalized: str) -> list[str]:
    seasons = set(re.findall(r"\b20\d{2}/20\d{2}\b", normalized))
    for start, short_end in re.findall(r"\b(20\d{2})/(\d{2})\b", normalized):
        century = int(start[:2]) * 100
        seasons.add(f"{start}/{century + int(short_end)}")
    return sorted(seasons)


def _temporal_scope(
    normalized: str,
    seasons: list[str],
    profiles: tuple[PlayerSeasonProfile, ...],
) -> TemporalScope:
    if any(term in normalized for term in ("form", "letzten spiele", "recent", "last matches")):
        return TemporalScope.RECENT_FORM
    if any(
        term in normalized
        for term in (
            "entwicklung",
            "entwickelt",
            "trend",
            "konstant",
            "verbessert",
            "verschlechtert",
        )
    ):
        return TemporalScope.TREND
    latest_start = max((int(profile.season_name[:4]) for profile in profiles), default=0)
    if any(int(season[:4]) < latest_start for season in seasons) or any(
        term in normalized
        for term in ("letzte saison", "vorherige saison", "historisch", "history")
    ):
        return TemporalScope.HISTORY
    return TemporalScope.CURRENT


def _result_count(normalized: str) -> int:
    match = re.search(r"\b(?:top|beste[n]?|besten)\s*(\d{1,3})\b", normalized)
    if match:
        return min(max(int(match.group(1)), 1), 100)
    words = {"fünf": 5, "zehn": 10, "zwanzig": 20, "five": 5, "ten": 10}
    return next((count for word, count in words.items() if word in normalized), 10)
