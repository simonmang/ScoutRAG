"""Formation/grid-derived tactical position refinement, no network dependency."""

from scoutrag.data.position_inference import (
    infer_slot_role,
    refine_position_group,
)


def test_back_four_defensive_line_splits_fullback_from_centre_back() -> None:
    assert infer_slot_role("4-2-3-1", "2:1") == "fullback_wingback"
    assert infer_slot_role("4-2-3-1", "2:4") == "fullback_wingback"
    assert infer_slot_role("4-2-3-1", "2:2") == "center_back"
    assert infer_slot_role("4-2-3-1", "2:3") == "center_back"


def test_double_pivot_is_defensive_midfield() -> None:
    assert infer_slot_role("4-2-3-1", "3:1") == "defensive_midfield"
    assert infer_slot_role("4-2-3-1", "3:2") == "defensive_midfield"


def test_advanced_three_splits_winger_from_attacking_midfield() -> None:
    assert infer_slot_role("4-2-3-1", "4:1") == "winger"
    assert infer_slot_role("4-2-3-1", "4:2") == "attacking_midfield"
    assert infer_slot_role("4-2-3-1", "4:3") == "winger"


def test_lone_striker_is_forward() -> None:
    assert infer_slot_role("4-2-3-1", "5:1") == "forward"


def test_goalkeeper_row_is_not_refined() -> None:
    assert infer_slot_role("4-2-3-1", "1:1") is None


def test_flat_four_two_midfield_splits_winger_from_central() -> None:
    # 4-4-2: single midfield line of four, edges are wide, centre two are central.
    assert infer_slot_role("4-4-2", "3:1") == "winger"
    assert infer_slot_role("4-4-2", "3:2") == "central_midfield"
    assert infer_slot_role("4-4-2", "3:3") == "central_midfield"
    assert infer_slot_role("4-4-2", "3:4") == "winger"
    assert infer_slot_role("4-4-2", "4:1") == "forward"
    assert infer_slot_role("4-4-2", "4:2") == "forward"


def test_central_midfield_trio_is_not_split_into_wingers() -> None:
    # 4-3-3: the middle three is a central triangle, not wide players.
    assert infer_slot_role("4-3-3", "3:1") == "central_midfield"
    assert infer_slot_role("4-3-3", "3:2") == "central_midfield"
    assert infer_slot_role("4-3-3", "3:3") == "central_midfield"
    # The front three, in contrast, genuinely has wide attacking slots.
    assert infer_slot_role("4-3-3", "4:1") == "winger"
    assert infer_slot_role("4-3-3", "4:2") == "forward"
    assert infer_slot_role("4-3-3", "4:3") == "winger"


def test_back_three_formations_are_not_refined() -> None:
    assert infer_slot_role("3-5-2", "2:1") is None
    assert infer_slot_role("3-4-3", "2:2") is None
    assert infer_slot_role("5-3-2", "2:1") is None


def test_unrecognized_or_malformed_input_is_not_refined() -> None:
    assert infer_slot_role("not-a-formation", "2:1") is None
    assert infer_slot_role("4-2-3-1", "not-a-grid") is None
    assert infer_slot_role("4-2-3-1", "2:9") is None  # Column outside the line.
    assert infer_slot_role("4-4-4", "2:1") is None  # Does not sum to ten outfield players.


def test_refine_position_group_requires_enough_observations() -> None:
    observations = [("4-2-3-1", "3:1")] * 4  # Below the default minimum of five.
    position, confidence = refine_position_group(observations, coarse_group="midfielder")
    assert position == "midfielder"
    assert confidence == 0.0


def test_refine_position_group_accepts_a_clear_majority() -> None:
    observations = [("4-2-3-1", "3:1")] * 8 + [("4-2-3-1", "4:2")] * 2
    position, confidence = refine_position_group(observations, coarse_group="midfielder")
    assert position == "defensive_midfield"
    assert confidence == 0.8


def test_refine_position_group_rejects_low_agreement() -> None:
    observations = [("4-2-3-1", "3:1")] * 3 + [("4-2-3-1", "4:2")] * 3
    position, confidence = refine_position_group(observations, coarse_group="midfielder")
    assert position == "midfielder"
    assert confidence == 0.0


def test_refine_position_group_rejects_coarse_mismatch() -> None:
    # Majority role resolves to a defender sub-role, but the season-long coarse
    # tag says midfielder - a handful of out-of-position starts must not
    # relabel the player's whole season.
    observations = [("4-2-3-1", "2:1")] * 8
    position, confidence = refine_position_group(observations, coarse_group="midfielder")
    assert position == "midfielder"
    assert confidence == 0.0


def test_refine_position_group_ignores_unrefinable_observations() -> None:
    observations = [("3-5-2", "2:1")] * 8  # Back-three: none of these refine.
    position, confidence = refine_position_group(observations, coarse_group="defender")
    assert position == "defender"
    assert confidence == 0.0
