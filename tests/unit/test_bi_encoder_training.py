"""Phase 9 dataset, mining, pairwise evaluation, and configuration tests."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from pydantic import ValidationError

from scoutrag.domain.player import PlayerSeasonProfile
from scoutrag.evaluation.models import GoldenDataset, GoldenJudgment, GoldenQuery
from scoutrag.retrieval.dense import TextEmbeddingModel
from scoutrag.training import cli as training_cli
from scoutrag.training import trainer as trainer_module
from scoutrag.training.dataset import load_training_dataset
from scoutrag.training.evaluation import BiEncoderEvaluator
from scoutrag.training.mining import FootballHardNegativeMiner
from scoutrag.training.models import (
    BiEncoderTrainingDataset,
    MinedTrainingDataset,
    RetrievalTrainingQuery,
)
from scoutrag.training.trainer import (
    BiEncoderTrainingConfig,
    SentenceTransformerBiEncoderTrainer,
)


class MarkerEmbeddingModel:
    """Deterministic two-dimensional embedding model for domain tests."""

    def __init__(self, *, positive_wins: bool = True) -> None:
        self.positive_wins = positive_wins

    @property
    def model_name(self) -> str:
        return "marker-encoder"

    def encode_queries(self, texts: object) -> list[list[float]]:
        return [[1.0, 0.0] for _ in cast(list[str], texts)]

    def encode_documents(self, texts: object) -> list[list[float]]:
        vectors = []
        for text in cast(list[str], texts):
            if "POSITIVE_MARKER" in text:
                vectors.append([0.95, 0.10] if self.positive_wins else [0.20, 1.0])
            elif "HARD_MARKER" in text:
                vectors.append([0.80, 0.20])
            else:
                vectors.append([-1.0, 0.0])
        return vectors


def _profile(
    player_id: str,
    name: str,
    position: str,
    marker: str,
    *,
    pressure_percentile: float | None = None,
    team: str = "Test FC",
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
        minutes_played=1000,
        structured_features={},
        percentiles=percentiles,
        profile_text=marker,
        data_quality=1,
    )


def _query(query_id: str, split: str) -> RetrievalTrainingQuery:
    return RetrievalTrainingQuery(
        query_id=query_id,
        concept_id="pressing-six",
        split=split,
        query="Pressingstarker Sechser",
        language="de" if query_id.endswith("de") else "en",
        positive_player_id="positive",
        negative_constraint="lower_target_metric",
        target_metric="pressures_per_90",
        minimum_percentile_gap=25,
        rationale="Typed percentile label.",
    )


def _specs() -> BiEncoderTrainingDataset:
    return BiEncoderTrainingDataset(
        schema_version="test-v1",
        name="test",
        competition_id=9,
        season_id=281,
        source_reference="test",
        labeling_method="test",
        queries=[_query("train-de", "train"), _query("validation-en", "validation")],
    )


def _profiles() -> list[PlayerSeasonProfile]:
    return [
        _profile(
            "positive",
            "Positive",
            "defensive_midfield",
            "POSITIVE_MARKER",
            pressure_percentile=100,
        ),
        _profile(
            "hard",
            "Hard",
            "defensive_midfield",
            "HARD_MARKER",
            pressure_percentile=50,
        ),
        _profile("easy", "Easy", "forward", "EASY_MARKER"),
    ]


def test_committed_training_seed_has_explicit_splits_and_languages() -> None:
    dataset = load_training_dataset(Path("evaluation/bi_encoder_training_queries.json"))

    assert len(dataset.queries) == 32
    assert sum(query.split == "train" for query in dataset.queries) == 20
    assert sum(query.split == "validation" for query in dataset.queries) == 12
    assert {query.language for query in dataset.queries} == {"de", "en"}


def test_constraint_specific_fields_are_required() -> None:
    with pytest.raises(ValidationError, match="target_metric"):
        RetrievalTrainingQuery(
            query_id="invalid",
            concept_id="invalid",
            split="train",
            query="query",
            language="de",
            positive_player_id="1",
            negative_constraint="lower_target_metric",
            rationale="missing metric",
        )


def test_miner_selects_similar_failing_profile_and_different_position() -> None:
    mined = FootballHardNegativeMiner(
        _profiles(),
        cast(TextEmbeddingModel, MarkerEmbeddingModel()),
    ).mine(_specs())

    example = mined.examples[0]
    assert example.positive_player_id == "positive"
    assert example.hard_negative_player_id == "hard"
    assert example.easy_negative_player_id == "easy"
    assert example.positive_score > example.hard_negative_score > example.easy_negative_score


def test_pairwise_metrics_detect_language_stability() -> None:
    mined = FootballHardNegativeMiner(
        _profiles(),
        cast(TextEmbeddingModel, MarkerEmbeddingModel()),
    ).mine(_specs())
    validation = mined.examples[1].model_copy(update={"concept_id": "pair", "language": "en"})
    german = validation.model_copy(update={"query_id": "validation-de", "language": "de"})
    paired = MinedTrainingDataset(
        source_dataset_version=mined.source_dataset_version,
        embedding_model=mined.embedding_model,
        examples=[mined.examples[0], german, validation],
    )
    evaluator = BiEncoderEvaluator(
        _profiles(),
        cast(GoldenDataset, object()),
        paired,
    )

    metrics = evaluator.evaluate_pairwise(cast(TextEmbeddingModel, MarkerEmbeddingModel()))

    assert metrics.hard_negative_accuracy == 1
    assert metrics.easy_negative_accuracy == 1
    assert metrics.language_accuracy == {"de": 1.0, "en": 1.0}
    assert metrics.bilingual_pair_stability == 1


def test_full_comparison_reuses_dense_retrieval_and_reports_delta() -> None:
    mined = FootballHardNegativeMiner(
        _profiles(),
        cast(TextEmbeddingModel, MarkerEmbeddingModel()),
    ).mine(_specs())
    golden = GoldenDataset(
        schema_version="golden-test",
        name="test",
        competition_id=9,
        season_id=281,
        source_reference="test",
        labeling_method="test",
        queries=[
            GoldenQuery(
                query_id="pressing-six",
                query="Pressingstarker Sechser",
                language="de",
                category="discovery",
                judgments=[
                    GoldenJudgment(
                        player_id="positive",
                        relevance=3,
                        rationale="typed positive",
                    )
                ],
            )
        ],
    )
    evaluator = BiEncoderEvaluator(_profiles(), golden, mined)

    report = evaluator.compare(
        cast(TextEmbeddingModel, MarkerEmbeddingModel(positive_wins=False)),
        cast(TextEmbeddingModel, MarkerEmbeddingModel()),
    )

    assert report.baseline.golden_retrieval.aggregate.mean_reciprocal_rank == 0.5
    assert report.fine_tuned.golden_retrieval.aggregate.mean_reciprocal_rank == 1
    assert report.delta.mean_reciprocal_rank == 0.5


def test_training_config_rejects_non_training_batch_size() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        BiEncoderTrainingConfig(base_model_name="test", batch_size=1)


@pytest.mark.parametrize("command", ["mine", "train", "evaluate", "all"])
def test_training_cli_exposes_each_reproducible_stage(command: str) -> None:
    parser = training_cli.build_parser()

    args = parser.parse_args([command])

    assert args.command == command


def test_trainer_writes_reproducible_metadata_with_lazy_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    mined = FootballHardNegativeMiner(
        _profiles(),
        cast(TextEmbeddingModel, MarkerEmbeddingModel()),
    ).mine(_specs())
    observed: dict[str, Any] = {}

    class FakeModel:
        def __init__(self, model_path: str, **kwargs: object) -> None:
            observed["model_path"] = model_path
            observed["model_kwargs"] = kwargs

        def save_pretrained(self, path: str, **kwargs: object) -> None:
            observed["saved_path"] = path
            observed["save_kwargs"] = kwargs
            Path(path, "tokenizer_config.json").write_text("{}", encoding="utf-8")

    class FakeDataset:
        @staticmethod
        def from_dict(payload: dict[str, list[str]]) -> dict[str, list[str]]:
            observed["dataset"] = payload
            return payload

    class FakeLoss:
        def __init__(self, model: object) -> None:
            observed["loss_model"] = model

    class FakeTrainer:
        def __init__(self, **kwargs: object) -> None:
            observed["trainer_kwargs"] = kwargs

        def train(self) -> None:
            observed["trained"] = True

    fake_sentence_transformers = SimpleNamespace(
        SentenceTransformer=FakeModel,
        SentenceTransformerTrainingArguments=lambda **kwargs: kwargs,
        SentenceTransformerTrainer=FakeTrainer,
    )
    fake_losses = SimpleNamespace(MultipleNegativesRankingLoss=FakeLoss)
    fake_datasets = SimpleNamespace(Dataset=FakeDataset)

    def fake_optional_module(name: str, _: str) -> object:
        return {
            "sentence_transformers": fake_sentence_transformers,
            "sentence_transformers.losses": fake_losses,
            "datasets": fake_datasets,
        }[name]

    monkeypatch.setattr(trainer_module, "_optional_module", fake_optional_module)
    monkeypatch.setattr(
        trainer_module,
        "_resolve_model_path",
        lambda model_name, **_: model_name,
    )
    output = tmp_path / "model"

    summary = SentenceTransformerBiEncoderTrainer().train(
        mined,
        output,
        BiEncoderTrainingConfig(
            base_model_name="baseline",
            epochs=1,
            batch_size=2,
            max_steps=1,
        ),
    )

    metadata = json.loads((output / "scoutrag_training.json").read_text(encoding="utf-8"))
    assert observed["trained"] is True
    assert len(cast(dict[str, list[str]], observed["dataset"])["query"]) == 1
    assert metadata["dataset_fingerprint"] == summary.dataset_fingerprint
    tokenizer_config = json.loads((output / "tokenizer_config.json").read_text(encoding="utf-8"))
    assert tokenizer_config["fix_mistral_regex"] is True
    assert "MultipleNegativesRankingLoss" in (output / "README.md").read_text(encoding="utf-8")
