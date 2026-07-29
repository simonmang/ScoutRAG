"""Query profile invariants."""

import pytest
from pydantic import ValidationError

from scoutrag.domain.query import QueryIntent, QueryProfile


def test_discovery_profile_has_safe_defaults() -> None:
    profile = QueryProfile(
        original_query="Ich suche einen pressingstarken Sechser",
        normalized_query="pressingstarker sechser",
        intent=QueryIntent.PLAYER_DISCOVERY,
        requested_positions=["defensive_midfield"],
        requested_traits=["pressing"],
    )

    assert profile.result_count == 10
    assert profile.minimum_minutes is None
    assert profile.named_players == []


@pytest.mark.parametrize(
    ("intent", "named_players"),
    [
        (QueryIntent.EXACT_PLAYER_LOOKUP, []),
        (QueryIntent.SIMILAR_PLAYER, []),
        (QueryIntent.PLAYER_COMPARISON, ["Spieler A"]),
    ],
)
def test_named_player_intents_require_reference_players(
    intent: QueryIntent,
    named_players: list[str],
) -> None:
    with pytest.raises(ValidationError):
        QueryProfile(
            original_query="Spieleranfrage",
            normalized_query="spieleranfrage",
            intent=intent,
            named_players=named_players,
        )


def test_result_count_is_bounded() -> None:
    with pytest.raises(ValidationError):
        QueryProfile(
            original_query="Top Spieler",
            normalized_query="top spieler",
            intent=QueryIntent.PLAYER_DISCOVERY,
            result_count=101,
        )
