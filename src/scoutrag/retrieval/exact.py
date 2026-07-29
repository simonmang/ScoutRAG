"""Exact retrieval for names and typed football metadata."""

from scoutrag.domain.player import PlayerSeasonProfile
from scoutrag.domain.query import QueryIntent, QueryProfile
from scoutrag.domain.retrieval import CandidateRetrievalTrace, PlayerCandidate
from scoutrag.retrieval.common import matches_hard_filters


class ExactPlayerRetriever:
    """Retrieve explicit player, team, competition, season, and position matches."""

    strategy_name = "exact"

    def __init__(self, profiles: list[PlayerSeasonProfile]) -> None:
        self.profiles = tuple(profiles)

    def retrieve(self, query_profile: QueryProfile, *, limit: int) -> list[PlayerCandidate]:
        if query_profile.intent is QueryIntent.OUT_OF_SCOPE:
            return []
        query = query_profile.normalized_query.casefold()
        scored: list[tuple[float, PlayerSeasonProfile]] = []
        named = {name.casefold() for name in query_profile.named_players}
        team_filters = {team.casefold() for team in query_profile.team_filters}
        for profile in self.profiles:
            if not matches_hard_filters(profile, query_profile):
                continue
            signals: list[float] = []
            player_name = profile.player_name.casefold()
            if player_name in named or player_name in query:
                signals.append(1.0)
            if any(team.casefold() in team_filters for team in profile.team_names) or any(
                team.casefold() in query for team in profile.team_names
            ):
                signals.append(0.9)
            if profile.competition_name.casefold() in query:
                signals.append(0.8)
            if profile.season_name.casefold() in query:
                signals.append(0.8)
            if (
                query_profile.requested_positions
                and profile.position_group in query_profile.requested_positions
            ):
                signals.append(0.7)
            if signals:
                scored.append((max(signals), profile))

        scored.sort(key=lambda item: (-item[0], item[1].player_name, item[1].season_name))
        return [
            PlayerCandidate(
                profile=profile,
                retrieval_trace=CandidateRetrievalTrace(
                    player_id=profile.player_id,
                    retrieved_by=[self.strategy_name],
                    exact_score=round(score, 6),
                    fused_score=round(score, 6),
                ),
            )
            for score, profile in scored[:limit]
        ]
