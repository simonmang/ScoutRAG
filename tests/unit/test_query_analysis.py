"""Rule-based query analysis edge cases with no network dependency."""

from scoutrag.domain.player import PlayerSeasonProfile
from scoutrag.retrieval.query_analysis import RuleBasedQueryAnalyzer


def _profile(player_id: str, name: str) -> PlayerSeasonProfile:
    return PlayerSeasonProfile(
        player_id=player_id,
        player_name=name,
        team_name="Some Club",
        team_names=["Some Club"],
        competition_name="Super Lig",
        season_name="2025/2026",
        position_group="midfielder",
        minutes_played=1000,
        structured_features={},
        percentiles={},
        profile_text=f"{name} | Some Club | Super Lig 2025/2026 | midfielder | 1000.0 minutes.",
        data_quality=0.9,
    )


def test_command_verb_is_not_mistaken_for_a_same_named_player() -> None:
    # A real API-Football player is literally named "Show". The English
    # command phrasing "Show the profile of ..." must not also match him.
    profiles = [_profile("1", "Joshua Kimmich"), _profile("2", "Show")]

    analyzed = RuleBasedQueryAnalyzer(profiles).analyze("Show the profile of Joshua Kimmich")

    assert analyzed.named_players == ["Joshua Kimmich"]
    assert analyzed.intent.value == "exact_player_lookup"


def test_command_verb_still_yields_exact_lookup_intent_without_collision() -> None:
    profiles = [_profile("1", "Joshua Kimmich")]

    analyzed = RuleBasedQueryAnalyzer(profiles).analyze("Show the profile of Joshua Kimmich")

    assert analyzed.named_players == ["Joshua Kimmich"]
    assert analyzed.intent.value == "exact_player_lookup"


def test_a_player_actually_named_show_is_still_found_when_directly_requested() -> None:
    profiles = [_profile("1", "Joshua Kimmich"), _profile("2", "Show")]

    analyzed = RuleBasedQueryAnalyzer(profiles).analyze("Compare Show and Joshua Kimmich")

    assert analyzed.named_players == ["Joshua Kimmich", "Show"]


def test_german_command_verb_is_not_mistaken_for_a_same_named_player() -> None:
    profiles = [_profile("1", "Joshua Kimmich"), _profile("2", "Zeige")]

    analyzed = RuleBasedQueryAnalyzer(profiles).analyze("Zeige das Profil von Joshua Kimmich")

    assert analyzed.named_players == ["Joshua Kimmich"]
