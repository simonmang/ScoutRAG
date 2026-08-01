"""Phase 4 query analysis and independent hybrid retrieval tests."""

from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

import pytest

from scoutrag.domain.player import PlayerSeasonProfile
from scoutrag.retrieval.dense import (
    DensePlayerRetriever,
    SentenceTransformerEmbeddingModel,
)
from scoutrag.retrieval.exact import ExactPlayerRetriever
from scoutrag.retrieval.fusion import FusionWeights, WeightedRetrievalFusion
from scoutrag.retrieval.pipeline import HybridRetrievalPipeline
from scoutrag.retrieval.query_analysis import RuleBasedQueryAnalyzer
from scoutrag.retrieval.sparse import BM25PlayerRetriever
from scoutrag.retrieval.structured import StructuredFeaturePlayerRetriever


def profile(
    player_id: str,
    name: str,
    team: str,
    position: str,
    *,
    minutes: float,
    pressures: float,
    pressure_percentile: float | None,
) -> PlayerSeasonProfile:
    percentiles = (
        {"pressures_per_90": pressure_percentile} if pressure_percentile is not None else {}
    )
    return PlayerSeasonProfile(
        player_id=player_id,
        player_name=name,
        team_name=team,
        team_names=[team],
        competition_name="1. Bundesliga",
        season_name="2023/2024",
        position_group=position,
        minutes_played=minutes,
        structured_features={
            "pressures_per_90": pressures,
            "progressive_passes_per_90": pressures / 2,
        },
        percentiles=percentiles,
        profile_text=(
            f"{name} | {team} | 1. Bundesliga 2023/2024 | {position} | {minutes:.1f} minutes."
        ),
        data_quality=0.9,
    )


@pytest.fixture
def profiles() -> list[PlayerSeasonProfile]:
    return [
        profile(
            "1",
            "Joshua Kimmich",
            "Bayern Munich",
            "defensive_midfield",
            minutes=2_000,
            pressures=17,
            pressure_percentile=95,
        ),
        profile(
            "2",
            "Aleksandar Pavlović",
            "Bayern Munich",
            "defensive_midfield",
            minutes=1_100,
            pressures=11,
            pressure_percentile=65,
        ),
        profile(
            "3",
            "Florian Wirtz",
            "Bayer Leverkusen",
            "attacking_midfield",
            minutes=2_500,
            pressures=13,
            pressure_percentile=88,
        ),
        profile(
            "4",
            "Manuel Neuer",
            "Bayern Munich",
            "goalkeeper",
            minutes=2_200,
            pressures=0.2,
            pressure_percentile=None,
        ),
    ]


class FakeEmbeddingModel:
    """Small deterministic backend proving bi-encoder role separation."""

    model_name = "fake-multilingual-bi-encoder"

    def encode_queries(self, texts: Sequence[str]) -> list[list[float]]:
        return [[1.0, 0.0] if "press" in text else [0.0, 1.0] for text in texts]

    def encode_documents(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            if "Joshua Kimmich" in text:
                vectors.append([1.0, 0.0])
            elif "Aleksandar" in text:
                vectors.append([0.8, 0.2])
            else:
                vectors.append([0.0, 1.0])
        return vectors


class FailingDocumentEmbeddingModel(FakeEmbeddingModel):
    def encode_documents(self, texts: Sequence[str]) -> list[list[float]]:
        raise AssertionError("a valid persisted index should avoid document re-encoding")


def test_rule_based_query_analysis_extracts_football_constraints(
    profiles: list[PlayerSeasonProfile],
) -> None:
    analyzed = RuleBasedQueryAnalyzer(profiles).analyze(
        "Top 5 pressingstarke Sechser mit mindestens 900 Minuten"
    )

    assert analyzed.intent.value == "aggregation"
    assert analyzed.requested_positions == ["defensive_midfield"]
    assert analyzed.requested_traits == ["pressing"]
    assert analyzed.requested_metrics == ["pressures_per_90"]
    assert analyzed.minimum_minutes == 900
    assert analyzed.result_count == 5


def test_exact_retrieval_finds_bayern_player_without_embeddings(
    profiles: list[PlayerSeasonProfile],
) -> None:
    query = RuleBasedQueryAnalyzer(profiles).analyze("Zeige das Profil von Joshua Kimmich")

    candidates = ExactPlayerRetriever(profiles).retrieve(query, limit=10)

    assert candidates[0].profile.player_name == "Joshua Kimmich"
    assert candidates[0].retrieval_trace.exact_score == 1
    assert candidates[0].retrieval_trace.retrieved_by == ["exact"]

    sparse_candidates = BM25PlayerRetriever(profiles).retrieve(query, limit=10)
    assert [candidate.profile.player_name for candidate in sparse_candidates] == ["Joshua Kimmich"]


def test_team_name_becomes_a_hard_filter(profiles: list[PlayerSeasonProfile]) -> None:
    query = RuleBasedQueryAnalyzer(profiles).analyze(
        "pressingstarker Sechser von Bayern München mit mindestens 900 Minuten"
    )

    candidates = StructuredFeaturePlayerRetriever(profiles).retrieve(query, limit=10)
    exact_candidates = ExactPlayerRetriever(profiles).retrieve(query, limit=10)

    assert query.team_filters == ["Bayern Munich"]
    assert {candidate.profile.team_name for candidate in candidates} == {"Bayern Munich"}
    assert {candidate.profile.team_name for candidate in exact_candidates} == {"Bayern Munich"}


def test_short_bayern_alias_becomes_a_hard_filter(
    profiles: list[PlayerSeasonProfile],
) -> None:
    query = RuleBasedQueryAnalyzer(profiles).analyze("Bayern-Sechser mit mindestens 900 Minuten")

    assert query.team_filters == ["Bayern Munich"]
    assert query.requested_positions == ["defensive_midfield"]


def test_query_analysis_preserves_unknown_competition_and_missing_metric_requests(
    profiles: list[PlayerSeasonProfile],
) -> None:
    analyzed = RuleBasedQueryAnalyzer(profiles).analyze(
        "Kreativer Zehner mit Expected Assists in der Premier League"
    )

    assert analyzed.competition_filters == ["Premier League"]
    assert analyzed.requested_traits == ["creativity"]
    assert analyzed.requested_metrics == ["expected_assists_per_90"]


@pytest.mark.parametrize(
    ("query", "expected_intent"),
    [
        ("Finde Spieler wie Joshua Kimmich", "similar_player"),
        ("Vergleiche Joshua Kimmich und Florian Wirtz", "player_comparison"),
        ("Wer gewinnt die nächste Weltmeisterschaft?", "out_of_scope"),
    ],
)
def test_query_intents_cover_supported_and_out_of_scope_requests(
    profiles: list[PlayerSeasonProfile],
    query: str,
    expected_intent: str,
) -> None:
    assert RuleBasedQueryAnalyzer(profiles).analyze(query).intent.value == expected_intent


@pytest.mark.parametrize(
    ("query", "expected_scope", "expected_seasons"),
    [
        ("Zeige die aktuelle Form von Joshua Kimmich", "recent_form", []),
        ("Zeige die Entwicklung von Joshua Kimmich", "trend", []),
        ("Joshua Kimmich in der Saison 2022/23", "history", ["2022/2023"]),
    ],
)
def test_query_analysis_keeps_time_perspectives_explicit(
    profiles: list[PlayerSeasonProfile],
    query: str,
    expected_scope: str,
    expected_seasons: list[str],
) -> None:
    analyzed = RuleBasedQueryAnalyzer(profiles).analyze(query)

    assert analyzed.temporal_scope.value == expected_scope
    assert analyzed.season_filters == expected_seasons


def test_structured_retrieval_applies_minutes_position_and_metric(
    profiles: list[PlayerSeasonProfile],
) -> None:
    query = RuleBasedQueryAnalyzer(profiles).analyze(
        "pressingstarker Sechser mit mindestens 900 Minuten"
    )

    candidates = StructuredFeaturePlayerRetriever(profiles).retrieve(query, limit=10)

    assert [candidate.profile.player_name for candidate in candidates] == [
        "Joshua Kimmich",
        "Aleksandar Pavlović",
    ]
    assert candidates[0].retrieval_trace.structured_score == 0.95


def test_bm25_preserves_rare_player_names(profiles: list[PlayerSeasonProfile]) -> None:
    query = RuleBasedQueryAnalyzer(profiles).analyze("Joshua Kimmich")

    candidates = BM25PlayerRetriever(profiles).retrieve(query, limit=4)

    assert candidates[0].profile.player_name == "Joshua Kimmich"
    assert candidates[0].retrieval_trace.sparse_score is not None
    assert candidates[0].retrieval_trace.sparse_score > 0


def test_dense_retrieval_uses_separate_query_and_document_embeddings(
    profiles: list[PlayerSeasonProfile],
) -> None:
    query = RuleBasedQueryAnalyzer(profiles).analyze("pressingstarker Sechser")
    retriever = DensePlayerRetriever(profiles, FakeEmbeddingModel())

    candidates = retriever.retrieve(query, limit=4)

    assert candidates[0].profile.player_name == "Joshua Kimmich"
    assert candidates[0].retrieval_trace.dense_score == 1


def test_dense_index_can_be_reused_without_document_encoding(
    profiles: list[PlayerSeasonProfile],
    tmp_path: Path,
) -> None:
    index_path = tmp_path / "dense-index.json"
    DensePlayerRetriever(
        profiles,
        FakeEmbeddingModel(),
        index_path=index_path,
    )

    reused = DensePlayerRetriever(
        profiles,
        FailingDocumentEmbeddingModel(),
        index_path=index_path,
    )

    assert index_path.exists()
    assert len(reused.document_embeddings) == len(profiles)


def test_local_sentence_transformer_resolves_cached_snapshot_before_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}

    class FakeSentenceTransformer:
        def __init__(self, model_path: str, *, local_files_only: bool) -> None:
            calls["model_path"] = model_path
            calls["local_files_only"] = local_files_only

        def encode_query(self, texts: list[str], **kwargs: object) -> list[list[float]]:
            calls["query"] = (texts, kwargs)
            return [[1.0, 0.0]]

    def import_module(name: str) -> object:
        if name == "huggingface_hub":
            return SimpleNamespace(
                snapshot_download=lambda **kwargs: (
                    calls.update({"snapshot": kwargs}) or "C:/cache/model"
                )
            )
        return SimpleNamespace(SentenceTransformer=FakeSentenceTransformer)

    monkeypatch.setattr(
        "scoutrag.retrieval.dense.importlib.import_module",
        import_module,
    )
    model = SentenceTransformerEmbeddingModel("test/model", local_files_only=True)

    assert model.encode_queries(["Bayern midfielder"]) == [[1.0, 0.0]]
    assert calls["snapshot"] == {
        "repo_id": "test/model",
        "local_files_only": True,
    }
    assert calls["model_path"] == "C:/cache/model"


def test_fusion_normalizes_scales_and_preserves_strategy_provenance(
    profiles: list[PlayerSeasonProfile],
) -> None:
    query = RuleBasedQueryAnalyzer(profiles).analyze("pressingstarker Sechser Joshua Kimmich")
    retrievers = (
        ExactPlayerRetriever(profiles),
        StructuredFeaturePlayerRetriever(profiles),
        BM25PlayerRetriever(profiles),
        DensePlayerRetriever(profiles, FakeEmbeddingModel()),
    )
    by_strategy = {
        retriever.strategy_name: retriever.retrieve(query, limit=10) for retriever in retrievers
    }

    fused = WeightedRetrievalFusion().fuse(query, by_strategy, limit=10)

    assert fused[0].profile.player_name == "Joshua Kimmich"
    assert fused[0].retrieval_trace.retrieved_by == [
        "dense",
        "sparse",
        "structured",
        "exact",
    ]
    assert 0 <= fused[0].retrieval_trace.fused_score <= 1


def test_pipeline_returns_broad_recall_trace_and_ranked_results(
    profiles: list[PlayerSeasonProfile],
) -> None:
    analyzer = RuleBasedQueryAnalyzer(profiles)
    pipeline = HybridRetrievalPipeline(
        analyzer,
        (
            ExactPlayerRetriever(profiles),
            StructuredFeaturePlayerRetriever(profiles),
            BM25PlayerRetriever(profiles),
            DensePlayerRetriever(profiles, FakeEmbeddingModel()),
        ),
        WeightedRetrievalFusion(),
        candidate_pool_size=3,
    )

    result = pipeline.search("Top 2 pressingstarke Sechser mit mindestens 900 Minuten")

    assert len(result.candidates) == 2
    assert result.candidates[0].profile.player_name == "Joshua Kimmich"
    assert result.retrieval_trace.candidates_before_reranking >= 2
    assert result.retrieval_trace.candidates_after_reranking == 2
    assert result.retrieval_trace.strategies_used == [
        "exact",
        "structured",
        "sparse",
        "dense",
    ]
    assert result.retrieval_trace.filters_applied["minimum_minutes"] == 900
    assert result.retrieval_trace.stage_timings_ms["fusion"] >= 0


def test_fusion_weights_must_sum_to_one() -> None:
    with pytest.raises(ValueError, match=r"sum to 1\.0"):
        FusionWeights(dense=1, sparse=1, structured=1, exact=1)
