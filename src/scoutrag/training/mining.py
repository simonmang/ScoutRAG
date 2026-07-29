"""Domain-constrained hard-negative mining over typed player profiles."""

from collections.abc import Sequence

from scoutrag.domain.player import PlayerSeasonProfile
from scoutrag.retrieval.common import profile_search_text, query_search_text
from scoutrag.retrieval.dense import TextEmbeddingModel, cosine_similarity
from scoutrag.retrieval.query_analysis import RuleBasedQueryAnalyzer
from scoutrag.training.models import (
    BiEncoderTrainingDataset,
    MinedTrainingDataset,
    MinedTrainingExample,
    RetrievalTrainingQuery,
)


class FootballHardNegativeMiner:
    """Mine similar but fachlich invalid negatives with explicit constraints."""

    def __init__(
        self,
        profiles: list[PlayerSeasonProfile],
        embedding_model: TextEmbeddingModel,
    ) -> None:
        if not profiles:
            raise ValueError("hard-negative mining requires player profiles")
        self.profiles = tuple(profiles)
        self.embedding_model = embedding_model
        self.analyzer = RuleBasedQueryAnalyzer(profiles)
        self.by_id = {profile.player_id: profile for profile in profiles}
        if len(self.by_id) != len(profiles):
            raise ValueError("training profiles must have unique player IDs")
        self.profile_texts = {
            profile.player_id: profile_search_text(profile) for profile in profiles
        }

    def mine(self, dataset: BiEncoderTrainingDataset) -> MinedTrainingDataset:
        """Resolve every query into one positive, hard negative, and easy negative."""
        resolved_queries = [
            query_search_text(self.analyzer.analyze(query.query)) for query in dataset.queries
        ]
        query_vectors = self.embedding_model.encode_queries(resolved_queries)
        document_vectors = self.embedding_model.encode_documents(
            [self.profile_texts[profile.player_id] for profile in self.profiles]
        )
        if len(query_vectors) != len(dataset.queries):
            raise ValueError("embedding model returned the wrong number of query vectors")
        if len(document_vectors) != len(self.profiles):
            raise ValueError("embedding model returned the wrong number of document vectors")
        vectors_by_id = {
            profile.player_id: vector
            for profile, vector in zip(self.profiles, document_vectors, strict=True)
        }

        examples = [
            self._mine_one(query, query_text, query_vector, vectors_by_id)
            for query, query_text, query_vector in zip(
                dataset.queries,
                resolved_queries,
                query_vectors,
                strict=True,
            )
        ]
        return MinedTrainingDataset(
            source_dataset_version=dataset.schema_version,
            embedding_model=self.embedding_model.model_name,
            examples=examples,
        )

    def _mine_one(
        self,
        query: RetrievalTrainingQuery,
        query_text: str,
        query_vector: Sequence[float],
        vectors_by_id: dict[str, list[float]],
    ) -> MinedTrainingExample:
        try:
            positive = self.by_id[query.positive_player_id]
        except KeyError as error:
            raise ValueError(
                f"{query.query_id}: unknown positive player {query.positive_player_id}"
            ) from error

        hard_pool = self._hard_pool(query, positive)
        if not hard_pool:
            raise ValueError(f"{query.query_id}: no profile satisfies the hard-negative rule")
        easy_pool = [
            profile
            for profile in self.profiles
            if profile.player_id != positive.player_id
            and profile.position_group != positive.position_group
        ]
        if not easy_pool:
            raise ValueError(f"{query.query_id}: no different-position easy negative exists")

        scored_hard = self._score_profiles(query_vector, hard_pool, vectors_by_id)
        scored_easy = self._score_profiles(query_vector, easy_pool, vectors_by_id)
        hard_score, hard = max(scored_hard, key=lambda item: (item[0], item[1].player_id))
        easy_score, easy = min(scored_easy, key=lambda item: (item[0], item[1].player_id))
        positive_score = cosine_similarity(query_vector, vectors_by_id[positive.player_id])
        return MinedTrainingExample(
            query_id=query.query_id,
            concept_id=query.concept_id,
            split=query.split,
            language=query.language,
            original_query=query.query,
            query_text=query_text,
            positive_player_id=positive.player_id,
            positive_player_name=positive.player_name,
            positive_text=self.profile_texts[positive.player_id],
            hard_negative_player_id=hard.player_id,
            hard_negative_player_name=hard.player_name,
            hard_negative_text=self.profile_texts[hard.player_id],
            easy_negative_player_id=easy.player_id,
            easy_negative_player_name=easy.player_name,
            easy_negative_text=self.profile_texts[easy.player_id],
            positive_score=round(positive_score, 6),
            hard_negative_score=round(hard_score, 6),
            easy_negative_score=round(easy_score, 6),
            negative_constraint=query.negative_constraint,
            rationale=query.rationale,
        )

    def _hard_pool(
        self,
        query: RetrievalTrainingQuery,
        positive: PlayerSeasonProfile,
    ) -> list[PlayerSeasonProfile]:
        same_position = [
            profile
            for profile in self.profiles
            if profile.player_id != positive.player_id
            and profile.position_group == positive.position_group
        ]
        if query.negative_constraint == "wrong_player":
            return same_position
        if query.negative_constraint == "wrong_team":
            required = (query.required_team or "").casefold()
            return [
                profile
                for profile in same_position
                if all(required not in team.casefold() for team in profile.team_names)
            ]

        metric = query.target_metric or ""
        positive_percentile = positive.percentiles.get(metric)
        if positive_percentile is None:
            raise ValueError(
                f"{query.query_id}: positive player has no percentile for {query.target_metric}"
            )
        threshold = positive_percentile - query.minimum_percentile_gap
        return [
            profile
            for profile in same_position
            if (percentile := profile.percentiles.get(metric)) is not None
            and percentile <= threshold
        ]

    @staticmethod
    def _score_profiles(
        query_vector: Sequence[float],
        profiles: list[PlayerSeasonProfile],
        vectors_by_id: dict[str, list[float]],
    ) -> list[tuple[float, PlayerSeasonProfile]]:
        return [
            (cosine_similarity(query_vector, vectors_by_id[profile.player_id]), profile)
            for profile in profiles
        ]
