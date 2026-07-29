"""StatsBomb event and lineup normalization tests."""

import json
from pathlib import Path
from typing import Any, cast

import pytest

from scoutrag.data.minutes import parse_elapsed_seconds, position_group
from scoutrag.data.normalization import normalize_event

FIXTURE_ROOT = Path(__file__).parents[1] / "fixtures" / "statsbomb"


def load_event(index: int) -> dict[str, Any]:
    events = cast(
        list[dict[str, Any]],
        json.loads((FIXTURE_ROOT / "events" / "1001.json").read_text("utf-8")),
    )
    return events[index]


def test_event_normalization_preserves_retrieval_relevant_fields() -> None:
    event = normalize_event(
        load_event(1),
        match_id=1001,
        competition_id=9,
        season_id=281,
    )

    assert event.event_type == "Pass"
    assert event.outcome_name == "Incomplete"
    assert event.player_id == 1
    assert event.location_x == 45.2
    assert event.end_location_x == 52.0
    assert event.pass_length == 6.9
    assert event.under_pressure is True
    assert event.source_reference.endswith("/events/event-2")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("68:47", 4_127.0),
        ("01:05:12", 3_912.0),
    ],
)
def test_elapsed_timestamp_parsing(value: str, expected: float) -> None:
    assert parse_elapsed_seconds(value) == expected


@pytest.mark.parametrize(
    ("position", "expected"),
    [
        ("Goalkeeper", "goalkeeper"),
        ("Left Center Back", "center_back"),
        ("Left Wing Back", "fullback_wingback"),
        ("Right Defensive Midfield", "defensive_midfield"),
        ("Right Center Midfield", "central_midfield"),
        ("Center Attacking Midfield", "attacking_midfield"),
        ("Left Wing", "winger"),
        ("Center Forward", "forward"),
        ("Unknown", "other"),
    ],
)
def test_position_group_mapping(position: str, expected: str) -> None:
    assert position_group(position) == expected


def test_invalid_elapsed_timestamp_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported elapsed timestamp"):
        parse_elapsed_seconds("invalid")
