"""Ranking metrics with binary relevance.

Every function takes a ranked list of item ids and the set of relevant ids, and
returns a per-query value. Aggregation and uncertainty are handled in ``stats.py``.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Sequence


def recall_at_k(ranked: Sequence[str], relevant: Collection[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(ranked[:k]) & set(relevant)) / len(set(relevant))


def precision_at_k(ranked: Sequence[str], relevant: Collection[str], k: int) -> float:
    """Precision with a fixed denominator ``k``: a list shorter than ``k`` is not
    rewarded for being short."""
    return len(set(ranked[:k]) & set(relevant)) / k


def hit_at_k(ranked: Sequence[str], relevant: Collection[str], k: int) -> float:
    return float(bool(set(ranked[:k]) & set(relevant)))


def reciprocal_rank(ranked: Sequence[str], relevant: Collection[str], k: int) -> float:
    rel = set(relevant)
    for i, item in enumerate(ranked[:k], start=1):
        if item in rel:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: Collection[str], k: int) -> float:
    rel = set(relevant)
    if not rel:
        return 0.0
    dcg = sum(1.0 / math.log2(i + 1) for i, item in enumerate(ranked[:k], start=1) if item in rel)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(rel), k) + 1))
    return dcg / ideal


CASE_METRICS = {
    "recall@1": lambda r, g: recall_at_k(r, g, 1),
    "recall@5": lambda r, g: recall_at_k(r, g, 5),
    "recall@10": lambda r, g: recall_at_k(r, g, 10),
    "recall@20": lambda r, g: recall_at_k(r, g, 20),
    "recall@100": lambda r, g: recall_at_k(r, g, 100),
    "mrr@10": lambda r, g: reciprocal_rank(r, g, 10),
    "ndcg@10": lambda r, g: ndcg_at_k(r, g, 10),
    "precision@1": lambda r, g: precision_at_k(r, g, 1),
    "precision@5": lambda r, g: precision_at_k(r, g, 5),
    "precision@10": lambda r, g: precision_at_k(r, g, 10),
}

# Paragraph level: a retrieved chunk counts when it contains the quoted paragraph.
PASSAGE_METRICS = {
    "passage_hit@1": lambda r, g: hit_at_k(r, g, 1),
    "passage_hit@5": lambda r, g: hit_at_k(r, g, 5),
    "passage_hit@10": lambda r, g: hit_at_k(r, g, 10),
    "passage_hit@20": lambda r, g: hit_at_k(r, g, 20),
    "passage_mrr@10": lambda r, g: reciprocal_rank(r, g, 10),
}


def per_query(ranked: Sequence[str], relevant: Collection[str], metrics=None) -> dict[str, float]:
    return {name: fn(ranked, relevant) for name, fn in (metrics or CASE_METRICS).items()}
