"""Dependency-free BM25 baseline over deterministic player documents."""

import math
from collections import Counter
from dataclasses import dataclass

from scoutrag.domain.player import PlayerSeasonProfile
from scoutrag.domain.query import QueryIntent, QueryProfile
from scoutrag.domain.retrieval import CandidateRetrievalTrace, PlayerCandidate
from scoutrag.retrieval.common import (
    matches_hard_filters,
    profile_search_text,
    query_search_text,
    tokenize,
)


@dataclass(frozen=True, slots=True)
class BM25Config:
    """Robertson BM25 parameters."""

    k1: float = 1.5
    b: float = 0.75

    def __post_init__(self) -> None:
        if self.k1 <= 0:
            raise ValueError("k1 must be positive")
        if not 0 <= self.b <= 1:
            raise ValueError("b must be between 0 and 1")


class BM25PlayerRetriever:
    """Lexical baseline that preserves rare names and football terminology."""

    strategy_name = "sparse"

    def __init__(
        self,
        profiles: list[PlayerSeasonProfile],
        config: BM25Config | None = None,
    ) -> None:
        self.profiles = tuple(profiles)
        self.config = config or BM25Config()
        self.documents = [tokenize(profile_search_text(profile)) for profile in self.profiles]
        self.term_frequencies = [Counter(document) for document in self.documents]
        self.document_frequencies = Counter(
            token for document in self.documents for token in set(document)
        )
        self.average_document_length = (
            sum(len(document) for document in self.documents) / len(self.documents)
            if self.documents
            else 0
        )

    def retrieve(self, query_profile: QueryProfile, *, limit: int) -> list[PlayerCandidate]:
        if query_profile.intent is QueryIntent.OUT_OF_SCOPE or not self.profiles:
            return []
        query_terms = Counter(tokenize(query_search_text(query_profile)))
        scored: list[tuple[float, PlayerSeasonProfile]] = []
        for index, profile in enumerate(self.profiles):
            if not matches_hard_filters(profile, query_profile):
                continue
            score = self._score(index, query_terms)
            if score > 0:
                scored.append((score, profile))
        scored.sort(key=lambda item: (-item[0], item[1].player_name, item[1].season_name))
        return [
            PlayerCandidate(
                profile=profile,
                retrieval_trace=CandidateRetrievalTrace(
                    player_id=profile.player_id,
                    retrieved_by=[self.strategy_name],
                    sparse_score=round(score, 6),
                    fused_score=round(score, 6),
                ),
            )
            for score, profile in scored[:limit]
        ]

    def _score(self, document_index: int, query_terms: Counter[str]) -> float:
        frequencies = self.term_frequencies[document_index]
        document_length = len(self.documents[document_index])
        score = 0.0
        for term, query_frequency in query_terms.items():
            term_frequency = frequencies.get(term, 0)
            if not term_frequency:
                continue
            document_frequency = self.document_frequencies[term]
            inverse_document_frequency = math.log(
                1 + ((len(self.documents) - document_frequency + 0.5) / (document_frequency + 0.5))
            )
            length_normalization = 1 - self.config.b
            if self.average_document_length:
                length_normalization += (
                    self.config.b * document_length / self.average_document_length
                )
            denominator = term_frequency + (self.config.k1 * length_normalization)
            score += (
                query_frequency
                * inverse_document_frequency
                * ((term_frequency * (self.config.k1 + 1)) / denominator)
            )
        return score
