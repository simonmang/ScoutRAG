"""Dependency-free information-retrieval metrics with graded relevance."""

import math
from collections.abc import Mapping, Sequence

from scoutrag.evaluation.models import KMetrics, RetrievalMetrics


def evaluate_ranking(
    ranked_player_ids: Sequence[str],
    broad_candidate_ids: Sequence[str],
    relevance: Mapping[str, int],
    *,
    k_values: Sequence[int] = (1, 5, 10),
) -> RetrievalMetrics:
    """Calculate candidate recall, Precision/Recall/nDCG@K, and reciprocal rank."""
    if not relevance:
        raise ValueError("at least one relevance judgment is required")
    cutoffs = tuple(sorted(set(k_values)))
    if not cutoffs or any(k < 1 for k in cutoffs):
        raise ValueError("k_values must contain positive integers")

    relevant_ids = set(relevance)
    broad_relevant = relevant_ids.intersection(broad_candidate_ids)
    candidate_recall = len(broad_relevant) / len(relevant_ids)
    first_relevant_rank = next(
        (
            rank
            for rank, player_id in enumerate(ranked_player_ids, start=1)
            if player_id in relevant_ids
        ),
        None,
    )
    reciprocal_rank = 1 / first_relevant_rank if first_relevant_rank is not None else 0

    metrics_at_k: dict[int, KMetrics] = {}
    ideal_relevances = sorted(relevance.values(), reverse=True)
    for k in cutoffs:
        top_k = ranked_player_ids[:k]
        retrieved_relevant = sum(player_id in relevant_ids for player_id in top_k)
        dcg = _discounted_cumulative_gain([relevance.get(player_id, 0) for player_id in top_k])
        ideal_dcg = _discounted_cumulative_gain(ideal_relevances[:k])
        metrics_at_k[k] = KMetrics(
            precision=_rounded(retrieved_relevant / k),
            recall=_rounded(retrieved_relevant / len(relevant_ids)),
            ndcg=_rounded(dcg / ideal_dcg if ideal_dcg else 0),
            hit_rate=float(retrieved_relevant > 0),
        )

    return RetrievalMetrics(
        candidate_recall=_rounded(candidate_recall),
        mean_reciprocal_rank=_rounded(reciprocal_rank),
        at_k=metrics_at_k,
    )


def mean_metrics(
    metrics: Sequence[RetrievalMetrics],
    *,
    k_values: Sequence[int],
) -> RetrievalMetrics:
    """Macro-average query metrics so large judgment sets cannot dominate."""
    if not metrics:
        raise ValueError("cannot aggregate an empty metric collection")
    return RetrievalMetrics(
        candidate_recall=_rounded(sum(item.candidate_recall for item in metrics) / len(metrics)),
        mean_reciprocal_rank=_rounded(
            sum(item.mean_reciprocal_rank for item in metrics) / len(metrics)
        ),
        at_k={
            k: KMetrics(
                precision=_rounded(sum(item.at_k[k].precision for item in metrics) / len(metrics)),
                recall=_rounded(sum(item.at_k[k].recall for item in metrics) / len(metrics)),
                ndcg=_rounded(sum(item.at_k[k].ndcg for item in metrics) / len(metrics)),
                hit_rate=_rounded(sum(item.at_k[k].hit_rate for item in metrics) / len(metrics)),
            )
            for k in k_values
        },
    )


def _discounted_cumulative_gain(relevances: Sequence[int]) -> float:
    total = 0.0
    for rank, relevance in enumerate(relevances, start=1):
        gain = float((2**relevance) - 1)
        total += gain / math.log2(rank + 1)
    return total


def _rounded(value: float) -> float:
    return round(value, 6)
