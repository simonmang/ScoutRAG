"""End-to-end Phase 5 ablation with deterministic embeddings."""

import json
from collections.abc import Sequence
from pathlib import Path

from scoutrag.domain.player import PlayerSeasonProfile
from scoutrag.evaluation.dataset import load_golden_dataset
from scoutrag.evaluation.models import GoldenDataset, GoldenJudgment, GoldenQuery
from scoutrag.evaluation.runner import AblationRunner
from scoutrag.retrieval.dense import DensePlayerRetriever

PROJECT_ROOT = Path(__file__).parents[2]


class EvaluationEmbeddingModel:
    model_name = "deterministic-evaluation-model"

    def encode_queries(self, texts: Sequence[str]) -> list[list[float]]:
        return [[1.0, 0.0] if "kimmich" in text.casefold() else [0.0, 1.0] for text in texts]

    def encode_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [[1.0, 0.0] if "Joshua Kimmich" in text else [0.0, 1.0] for text in texts]


def player(player_id: str, name: str, team: str) -> PlayerSeasonProfile:
    return PlayerSeasonProfile(
        player_id=player_id,
        player_name=name,
        team_name=team,
        team_names=[team],
        competition_name="1. Bundesliga",
        season_name="2023/2024",
        position_group="defensive_midfield",
        minutes_played=900,
        structured_features={"pressures_per_90": 10},
        percentiles={"pressures_per_90": 50},
        profile_text=f"{name} | {team} | defensive midfielder.",
        data_quality=0.8,
    )


def test_committed_golden_dataset_is_valid_and_versioned() -> None:
    dataset = load_golden_dataset(PROJECT_ROOT / "evaluation" / "golden_queries.json")

    assert dataset.schema_version == "phase5-golden-v2"
    assert len(dataset.queries) == 8
    assert {query.language for query in dataset.queries} == {"de", "en"}
    assert any("Bayern" in query.query for query in dataset.queries)

    baseline = json.loads(
        (PROJECT_ROOT / "evaluation" / "baseline_summary.json").read_text("utf-8")
    )
    assert baseline["dataset_version"] == dataset.schema_version
    assert baseline["query_count"] == len(dataset.queries)
    assert baseline["metrics"]["H_full_phase4_hybrid"]["ndcg_at_5"] == 0.807192


def test_ablation_runner_compares_all_phase5_variants(tmp_path: Path) -> None:
    profiles = [
        player("5579", "Joshua Kimmich", "Bayern Munich"),
        player("40724", "Florian Wirtz", "Bayer Leverkusen"),
    ]
    dataset = GoldenDataset(
        schema_version="test-golden-v1",
        name="Test",
        competition_id=9,
        season_id=281,
        source_reference="test",
        labeling_method="deterministic integration fixture",
        queries=[
            GoldenQuery(
                query_id="kimmich",
                query="Zeige das Profil von Joshua Kimmich",
                language="de",
                category="exact",
                judgments=[
                    GoldenJudgment(
                        player_id="5579",
                        relevance=3,
                        rationale="Exact identity.",
                    )
                ],
            )
        ],
    )
    dense = DensePlayerRetriever(
        profiles,
        EvaluationEmbeddingModel(),
        index_path=tmp_path / "dense-index.json",
    )

    report = AblationRunner(profiles, dense).run(dataset)

    assert [item.variant_name for item in report.reports] == [
        "A_bm25",
        "B_pretrained_bi_encoder",
        "C_bm25_plus_bi_encoder",
        "D_bm25_bi_encoder_structured",
        "H_full_phase4_hybrid",
    ]
    assert all(item.aggregate.candidate_recall == 1 for item in report.reports)
    assert all(item.aggregate.mean_reciprocal_rank == 1 for item in report.reports)
