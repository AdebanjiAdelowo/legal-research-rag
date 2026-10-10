"""Cross-encoder reranking of first-stage candidates.

A cross-encoder reads the query and a candidate chunk together and outputs one
relevance score. It is far more expensive per pair than a bi-encoder, so it only
rescoring the top ``top_n`` candidates; anything below that keeps its first-stage
order beneath the reranked block.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from legalrag.index.dense import pick_device
from legalrag.index.store import ChunkStore
from legalrag.retrieval.core import Ranking


@dataclass(frozen=True)
class RerankerSpec:
    key: str
    model: str
    revision: str
    max_tokens: int = 512
    note: str = ""


RERANKERS: dict[str, RerankerSpec] = {
    spec.key: spec
    for spec in [
        RerankerSpec(
            "minilm-l6",
            "cross-encoder/ms-marco-MiniLM-L6-v2",
            "main",
            note="22M parameters, trained on MS MARCO web passages. Runs on a laptop.",
        ),
        RerankerSpec(
            "bge-reranker-base",
            "BAAI/bge-reranker-base",
            "2cfc18c9415c912f9d8155881c133215df768a70",
            note="278M parameters. GPU configuration.",
        ),
        RerankerSpec(
            "bge-reranker-v2-m3",
            "BAAI/bge-reranker-v2-m3",
            "main",
            note="568M parameters. GPU configuration.",
        ),
    ]
}


class Reranker:
    def __init__(self, spec: RerankerSpec | str, device: str | None = None, batch_size: int = 64):
        from sentence_transformers import CrossEncoder

        self.spec = RERANKERS[spec] if isinstance(spec, str) else spec
        self.device = pick_device(device)
        self.batch_size = batch_size
        self.model = CrossEncoder(
            self.spec.model,
            revision=self.spec.revision,
            device=self.device,
            max_length=self.spec.max_tokens,
        )

    def score(self, query: str, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty(0, dtype=np.float32)
        scores = self.model.predict(
            [(query, t) for t in texts], batch_size=self.batch_size, show_progress_bar=False
        )
        return np.asarray(scores, dtype=np.float32).reshape(-1)

    def rerank(self, query: str, ranking: Ranking, store: ChunkStore, top_n: int = 50) -> Ranking:
        return apply_scores(ranking, self.score(query, head_texts(ranking, store, top_n)))


def head_texts(ranking: Ranking, store: ChunkStore, top_n: int) -> list[str]:
    return [store.chunks[int(i)].text for i in ranking.idx[:top_n]]


def apply_scores(ranking: Ranking, head_scores: np.ndarray) -> Ranking:
    """Reorder the first ``len(head_scores)`` entries by the new scores.

    The tail keeps its order and is given scores below the reranked block, so the
    combined list stays sorted.
    """
    n = len(head_scores)
    if n == 0:
        return ranking
    order = np.argsort(-head_scores, kind="stable")
    idx = np.concatenate([ranking.idx[:n][order], ranking.idx[n:]])
    floor = float(head_scores.min()) - 1.0
    tail = floor - np.arange(1, len(ranking) - n + 1, dtype=np.float32)
    return Ranking(idx, np.concatenate([head_scores[order], tail]).astype(np.float32))
