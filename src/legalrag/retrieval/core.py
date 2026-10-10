"""Ranked lists, filtering and the two fusion methods.

A ``Ranking`` is the chunk-level result for one query: chunk indices in rank order
with their scores. Case-level results are derived from it by keeping each case's
best chunk, which is how a researcher would read the list.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from legalrag.index.store import ChunkStore


@dataclass
class Ranking:
    idx: np.ndarray  # chunk indices, best first
    scores: np.ndarray

    def __len__(self) -> int:
        return len(self.idx)

    def head(self, k: int) -> Ranking:
        return Ranking(self.idx[:k], self.scores[:k])


def top_k(scores: np.ndarray, k: int, mask: np.ndarray | None = None) -> Ranking:
    """Top ``k`` entries of a score vector, restricted to ``mask`` if given."""
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)
    k = min(k, scores.size)
    part = np.argpartition(-scores, k - 1)[:k]
    order = part[np.argsort(-scores[part], kind="stable")]
    order = order[np.isfinite(scores[order])]
    return Ranking(order.astype(np.int64), scores[order].astype(np.float32))


def rrf(rankings: list[Ranking], k: int = 60, depth: int | None = None) -> Ranking:
    """Reciprocal rank fusion: score(d) = sum over lists of 1 / (k + rank(d))."""
    fused: dict[int, float] = {}
    for r in rankings:
        for rank, i in enumerate(r.idx[:depth] if depth else r.idx, start=1):
            fused[int(i)] = fused.get(int(i), 0.0) + 1.0 / (k + rank)
    return _from_dict(fused)


def score_fusion(first: Ranking, second: Ranking, alpha: float) -> Ranking:
    """Convex combination of min-max normalised scores.

    ``alpha`` weights ``first``. Scores are normalised per query over each list's
    own candidates; a chunk missing from a list contributes zero from it. BM25 and
    cosine scores live on different scales, so the raw values cannot be added.
    """
    fused: dict[int, float] = {}
    for r, w in ((first, alpha), (second, 1.0 - alpha)):
        if len(r) == 0:
            continue
        lo, hi = float(r.scores.min()), float(r.scores.max())
        span = hi - lo
        norm = (r.scores - lo) / span if span > 0 else np.ones_like(r.scores)
        for i, s in zip(r.idx, norm, strict=True):
            fused[int(i)] = fused.get(int(i), 0.0) + w * float(s)
    return _from_dict(fused)


def _from_dict(fused: dict[int, float]) -> Ranking:
    if not fused:
        return Ranking(np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32))
    idx = np.fromiter(fused.keys(), dtype=np.int64, count=len(fused))
    scores = np.fromiter(fused.values(), dtype=np.float32, count=len(fused))
    order = np.lexsort((idx, -scores))  # ties broken by chunk index, deterministically
    return Ranking(idx[order], scores[order])


def case_ranking(ranking: Ranking, store: ChunkStore) -> list[str]:
    """Case ids in order of each case's best-ranked chunk."""
    seen: dict[int, None] = {}
    for c in store.chunk_case[ranking.idx]:
        seen.setdefault(int(c))
    return [store.case_ids[c] for c in seen]


def chunk_hits(ranking: Ranking, store: ChunkStore, gold_paras: set[str]) -> list[str]:
    """Per rank, ``"hit"`` if the chunk contains a gold paragraph, else its chunk id.

    Returning a constant for hits lets the generic metrics treat "any chunk that
    contains the passage" as one relevant item.
    """
    out = []
    for i in ranking.idx:
        chunk = store.chunks[int(i)]
        out.append("hit" if gold_paras.intersection(chunk.para_ids) else chunk.chunk_id)
    return out


def words_at_k(ranking: Ranking, store: ChunkStore, k: int) -> int:
    """Total words in the top ``k`` chunks: the reading cost of that cutoff."""
    return sum(len(store.chunks[int(i)].text.split()) for i in ranking.idx[:k])
