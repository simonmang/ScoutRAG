"""Mathematical correctness tests for Phase 5 retrieval metrics."""

import math

import pytest
from pydantic import ValidationError

from scoutrag.evaluation.metrics import evaluate_ranking, mean_metrics
from scoutrag.evaluation.models import GoldenDataset


def test_ranking_metrics_separate_broad_recall_from_final_order() -> None:
    metrics = evaluate_ranking(
        ranked_player_ids=["a", "x", "b"],
        broad_candidate_ids=["a", "b", "c", "x"],
        relevance={"a": 3, "b": 2, "c": 1},
        k_values=(1, 3),
    )

    ideal_dcg_at_3 = 7 + (3 / math.log2(3)) + (1 / math.log2(4))
    actual_dcg_at_3 = 7 + (3 / math.log2(4))
    assert metrics.candidate_recall == 1
    assert metrics.mean_reciprocal_rank == 1
    assert metrics.at_k[1].precision == 1
    assert metrics.at_k[1].recall == pytest.approx(1 / 3, abs=1e-6)
    assert metrics.at_k[1].ndcg == 1
    assert metrics.at_k[3].precision == pytest.approx(2 / 3, abs=1e-6)
    assert metrics.at_k[3].recall == pytest.approx(2 / 3, abs=1e-6)
    assert metrics.at_k[3].ndcg == pytest.approx(
        actual_dcg_at_3 / ideal_dcg_at_3,
        abs=1e-6,
    )


def test_reciprocal_rank_and_candidate_recall_handle_misses() -> None:
    metrics = evaluate_ranking(
        ranked_player_ids=["x", "a"],
        broad_candidate_ids=["a"],
        relevance={"a": 3, "b": 2},
        k_values=(2,),
    )

    assert metrics.candidate_recall == 0.5
    assert metrics.mean_reciprocal_rank == 0.5
    assert metrics.at_k[2].precision == 0.5
    assert metrics.at_k[2].recall == 0.5


def test_macro_average_weights_queries_equally() -> None:
    first = evaluate_ranking(["a"], ["a"], {"a": 3}, k_values=(1,))
    second = evaluate_ranking(["x"], [], {"b": 3}, k_values=(1,))

    aggregate = mean_metrics([first, second], k_values=(1,))

    assert aggregate.candidate_recall == 0.5
    assert aggregate.mean_reciprocal_rank == 0.5
    assert aggregate.at_k[1].precision == 0.5
    assert aggregate.at_k[1].recall == 0.5
    assert aggregate.at_k[1].ndcg == 0.5


def test_metrics_reject_empty_judgments_and_invalid_cutoffs() -> None:
    with pytest.raises(ValueError, match="relevance judgment"):
        evaluate_ranking([], [], {}, k_values=(1,))
    with pytest.raises(ValueError, match="positive integers"):
        evaluate_ranking([], [], {"a": 1}, k_values=(0,))


def test_golden_dataset_rejects_duplicate_query_ids() -> None:
    query = {
        "query_id": "duplicate",
        "query": "Joshua Kimmich",
        "language": "de",
        "category": "exact",
        "judgments": [
            {
                "player_id": "5579",
                "relevance": 3,
                "rationale": "Exact identity.",
            }
        ],
    }
    with pytest.raises(ValidationError, match="query IDs"):
        GoldenDataset.model_validate(
            {
                "schema_version": "test-v1",
                "name": "Test",
                "competition_id": 9,
                "season_id": 281,
                "source_reference": "test",
                "labeling_method": "test labels",
                "queries": [query, query],
            }
        )
