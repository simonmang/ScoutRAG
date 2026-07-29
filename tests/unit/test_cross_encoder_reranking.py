"""Cross-encoder reranking remains injectable, stable, and score-safe."""

from collections.abc import Sequence
from types import SimpleNamespace

import pytest

from scoutrag.domain.player import PlayerSeasonProfile
from scoutrag.domain.query import QueryIntent, QueryProfile
from scoutrag.domain.retrieval import CandidateRetrievalTrace, PlayerCandidate
from scoutrag.reranking.cross_encoder import (
    CrossEncoderPlayerReranker,
    SentenceTransformerCrossEncoderModel,
)


def profile(player_id: str, name: str) -> PlayerSeasonProfile:
    return PlayerSeasonProfile(
        player_id=player_id,
        player_name=name,
        team_name="Bayern Munich",
        team_names=["Bayern Munich"],
        competition_name="1. Bundesliga",
        season_name="2023/2024",
        position_group="defensive_midfield",
        minutes_played=1_000,
        structured_features={"pressures_per_90": 10},
        percentiles={"pressures_per_90": 80},
        profile_text=f"{name} is a Bayern Munich midfielder.",
        data_quality=0.8,
    )


def candidate(player_id: str, name: str, fused_score: float) -> PlayerCandidate:
    return PlayerCandidate(
        profile=profile(player_id, name),
        retrieval_trace=CandidateRetrievalTrace(
            player_id=player_id,
            retrieved_by=["sparse", "dense"],
            sparse_score=fused_score,
            dense_score=fused_score,
            fused_score=fused_score,
        ),
    )


def query() -> QueryProfile:
    return QueryProfile(
        original_query="pressingstarker Sechser von Bayern München",
        normalized_query="pressingstarker sechser von bayern münchen",
        intent=QueryIntent.PLAYER_DISCOVERY,
        requested_positions=["defensive_midfield"],
        requested_traits=["pressing"],
        requested_metrics=["pressures_per_90"],
        team_filters=["Bayern Munich"],
        result_count=2,
    )


class FakePairModel:
    model_name = "fake-football-cross-encoder"

    def __init__(self, scores: list[float]) -> None:
        self.scores = scores
        self.pairs: Sequence[tuple[str, str]] = []

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        self.pairs = pairs
        return self.scores


def test_cross_encoder_reorders_same_candidates_and_preserves_trace() -> None:
    model = FakePairModel([0.1, 0.9])
    original = [
        candidate("1", "Joshua Kimmich", 0.9),
        candidate("2", "Aleksandar Pavlović", 0.7),
    ]

    ranked = CrossEncoderPlayerReranker(model).rerank(query(), original)

    assert [item.profile.player_id for item in ranked] == ["2", "1"]
    assert [item.rank for item in ranked] == [1, 2]
    assert ranked[0].reranker_score == 0.9
    assert ranked[1].retrieval_trace is original[0].retrieval_trace
    assert "not a calibrated probability" in ranked[0].ranking_reasons[0]
    assert "pressures per 90" in model.pairs[0][0]
    assert "Joshua Kimmich" in model.pairs[0][1]


def test_cross_encoder_handles_empty_pool_and_rejects_score_count_mismatch() -> None:
    reranker = CrossEncoderPlayerReranker(FakePairModel([]))
    assert reranker.rerank(query(), []) == []

    with pytest.raises(ValueError, match="one score per candidate"):
        reranker.rerank(query(), [candidate("1", "Joshua Kimmich", 1)])


def test_sentence_transformer_adapter_is_lazy_and_forwards_inference_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}

    class FakeCrossEncoder:
        def __init__(self, model_name: str, **kwargs: object) -> None:
            calls["model_name"] = model_name
            calls["constructor"] = kwargs

        def predict(self, pairs: list[tuple[str, str]], **kwargs: object) -> list[float]:
            calls["pairs"] = pairs
            calls["predict"] = kwargs
            return [0.25] * len(pairs)

    monkeypatch.setattr(
        "scoutrag.reranking.cross_encoder.importlib.import_module",
        lambda name: SimpleNamespace(CrossEncoder=FakeCrossEncoder),
    )
    model = SentenceTransformerCrossEncoderModel(
        "test/model",
        batch_size=4,
        local_files_only=True,
        backend="onnx",
        device="cpu",
    )

    assert model.model_name == "test/model"
    assert model.score_pairs([]) == []
    assert model.score_pairs([("query", "profile")]) == [0.25]
    assert calls["model_name"] == "test/model"
    assert calls["constructor"] == {
        "local_files_only": True,
        "backend": "onnx",
        "device": "cpu",
        "model_kwargs": {"file_name": "onnx/model.onnx"},
    }
    assert calls["predict"] == {
        "batch_size": 4,
        "show_progress_bar": False,
        "convert_to_numpy": True,
    }


def test_sentence_transformer_adapter_validates_batch_size() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        SentenceTransformerCrossEncoderModel(batch_size=0)
