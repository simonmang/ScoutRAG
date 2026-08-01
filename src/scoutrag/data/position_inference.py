"""Derive a refined tactical position group from lineup formation grid data.

API-Football's per-appearance ``games.position`` field is coarse
(``goalkeeper``/``defender``/``midfielder``/``forward``). Its fixture
``lineups`` block additionally reports, for starting players only, the
team's ``formation`` (for example ``"4-2-3-1"``) and each player's
``grid`` slot (``"row:column"``) within it. Combined across a player's
starts in one season, this is enough to distinguish tactical roles the
coarse field cannot — a fullback from a centre-back, a holding
midfielder from an attacking one, a winger from a central forward —
without any second data source or player-identity matching.

Refinement only ever narrows a coarse group into one of its own
sub-roles; it never contradicts the provider's own coarse tag. It is
also deliberately limited to back-four formations, where the
fullback/centre-back and wide/central splits are unambiguous. Back-three
formations mix wing-back and midfield conventions in ways a grid slot
alone cannot resolve safely, so they are left at the coarse group
instead of guessed.
"""

from __future__ import annotations

from collections import Counter

REFINED_TO_COARSE: dict[str, str] = {
    "goalkeeper": "goalkeeper",
    "center_back": "defender",
    "fullback_wingback": "defender",
    "defensive_midfield": "midfielder",
    "central_midfield": "midfielder",
    "attacking_midfield": "midfielder",
    "winger": "midfielder",
    "forward": "forward",
}

DEFAULT_MINIMUM_OBSERVATIONS = 5
DEFAULT_MINIMUM_AGREEMENT = 0.6


def infer_slot_role(formation: str, grid: str) -> str | None:
    """Return a refined role for one lineup slot, or ``None`` if unclear.

    ``formation`` is the provider's back-to-front string (for example
    ``"4-2-3-1"``); ``grid`` is the player's ``"row:column"`` slot (row 1
    is always the goalkeeper). Returns ``None`` for goalkeepers (already
    unambiguous at the coarse level), back-three/back-five formations,
    formations with more than two midfield lines, or malformed input.
    """

    rows = _parse_formation(formation)
    if rows is None:
        return None
    row_index, column = _parse_grid(grid)
    if row_index is None or column is None:
        return None
    if row_index == 1:
        return None  # Goalkeeper: already unambiguous, nothing to refine.
    if rows[0] != 4:
        return None  # Back-three/back-five: wing-back convention is ambiguous.

    line_index = row_index - 2  # 0 = defence, increasing towards attack.
    if line_index < 0 or line_index >= len(rows):
        return None
    count = rows[line_index]
    if column < 1 or column > count:
        return None
    is_edge = column in (1, count)

    if line_index == 0:  # Defensive line.
        return "fullback_wingback" if is_edge and count == 4 else "center_back"
    if line_index == len(rows) - 1:  # Forward line.
        if count <= 2:
            return "forward"
        return "winger" if is_edge else "forward"

    midfield_lines = rows[1:-1]
    if len(midfield_lines) > 2:
        return None  # Too many midfield lines to attribute confidently.
    is_deepest_midfield_line = line_index == 1
    if len(midfield_lines) == 1:
        if count <= 2:
            return "defensive_midfield"
        if count == 3:
            return "central_midfield"  # A central trio, not wide slots.
        return "winger" if is_edge else "central_midfield"
    if is_deepest_midfield_line:
        return "central_midfield" if count == 3 else "defensive_midfield"
    if count <= 2:
        return "attacking_midfield"
    return "winger" if is_edge else "attacking_midfield"


def refine_position_group(
    observations: list[tuple[str, str]],
    *,
    coarse_group: str,
    minimum_observations: int = DEFAULT_MINIMUM_OBSERVATIONS,
    minimum_agreement: float = DEFAULT_MINIMUM_AGREEMENT,
) -> tuple[str, float]:
    """Return ``(position_group, confidence)`` for one player-season.

    ``observations`` is every ``(formation, grid)`` pair from the
    player's starts that season. Falls back to ``coarse_group`` with
    confidence ``0.0`` unless there are enough observations, a clear
    majority role, and that role's own coarse bucket agrees with
    ``coarse_group`` — a player used out of position in a handful of
    matches must not relabel their whole season.
    """

    roles = [
        role
        for formation, grid in observations
        if (role := infer_slot_role(formation, grid)) is not None
    ]
    if len(roles) < minimum_observations:
        return coarse_group, 0.0
    role, count = Counter(roles).most_common(1)[0]
    agreement = count / len(roles)
    if agreement < minimum_agreement:
        return coarse_group, 0.0
    if REFINED_TO_COARSE[role] != coarse_group:
        return coarse_group, 0.0
    return role, round(agreement, 3)


def _parse_formation(formation: str) -> list[int] | None:
    parts = formation.strip().split("-")
    if len(parts) < 2:
        return None
    counts: list[int] = []
    for part in parts:
        if not part.isdigit():
            return None
        value = int(part)
        if value <= 0:
            return None
        counts.append(value)
    if sum(counts) + 1 != 11:
        return None  # Must describe exactly ten outfield players plus the keeper.
    return counts


def _parse_grid(grid: str) -> tuple[int | None, int | None]:
    parts = grid.strip().split(":")
    if len(parts) != 2 or not all(part.isdigit() for part in parts):
        return None, None
    row, column = int(parts[0]), int(parts[1])
    if row < 1 or column < 1:
        return None, None
    return row, column
