"""Shared machinery for experiments: loading, first-stage retrieval with caching,
per-query evaluation and paired comparison.

Protocol used throughout:

* A query is dated by the opinion it comes from. By default only cases decided
  before that date are eligible ("as-of" retrieval), because a later case could not
  have been cited and returning it is a form of leakage.
* Hyperparameters are chosen on the ``dev`` split and reported on ``test``.
* Each system is evaluated on the same queries, so differences are paired.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from legalrag.data.build import load_queries
from legalrag.data.schema import Query
from legalrag.eval import metrics as M
from legalrag.eval import stats
from legalrag.index import store as S
from legalrag.index.bm25 import BM25Index
from legalrag.index.dense import DenseIndex
from legalrag.paths import Paths
from legalrag.retrieval.core import Ranking, case_ranking, chunk_hits, top_k

DEPTH = 1000  # chunks kept per query from each first-stage retriever


@dataclass
class Run:
    """Chunk rankings for a list of queries, with wall-clock cost."""

    name: str
    rankings: list[Ranking]
    seconds_per_query: float = float("nan")
    extra: dict = field(default_factory=dict)


@dataclass
class Bench:
    paths: Paths
    store: S.ChunkStore
    _bm25: BM25Index | None = None
    _dense: dict[str, DenseIndex] = field(default_factory=dict)

    @classmethod
    def load(cls, scheme: str = "section220", paths: Paths | None = None) -> Bench:
        paths = paths or Paths.default()
        return cls(paths, S.ChunkStore.load(paths, scheme))

    def queries(self, split: str, limit: int | None = None) -> list[Query]:
        items = load_queries(self.paths, split)
        return items[:limit] if limit else items

    @property
    def bm25(self) -> BM25Index:
        if self._bm25 is None:
            self._bm25 = S.bm25_index(self.paths, self.store)
        return self._bm25

    def dense(self, key: str) -> DenseIndex:
        if key not in self._dense:
            self._dense[key] = S.dense_index(self.paths, self.store, key)
        return self._dense[key]

    def has_dense(self, key: str) -> bool:
        return (S.index_dir(self.paths, self.store, f"dense-{key}") / "meta.json").exists()

    # -- first stage ---------------------------------------------------------

    def masks(self, queries: list[Query], as_of: bool = True, **filters) -> list:
        """Per-query chunk masks. ``as_of`` restricts to cases decided before the
        query date; ``filters`` are passed to ``ChunkStore.mask`` for every query."""
        out = []
        for q in queries:
            kw = dict(filters)
            if as_of:
                kw["before"] = q.date
            out.append(self.store.mask(**kw))
        return out

    def _cache_file(self, name: str, texts: list[str], masks_key: str) -> Path:
        h = hashlib.sha256()
        for t in texts:
            h.update(t.encode())
            h.update(b"\0")
        h.update(masks_key.encode())
        directory = self.paths.indexes / self.store.corpus_version / self.store.scheme / "runs"
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f"{name}-{h.hexdigest()[:16]}.npz"

    def first_stage(
        self,
        name: str,
        texts: list[str],
        masks: list | None = None,
        masks_key: str = "as_of",
        depth: int = DEPTH,
        cache: bool = True,
    ) -> Run:
        """Rank chunks for each text with ``"bm25"`` or ``"dense-<encoder>"``.

        ``masks_key`` must describe the masks, since it is part of the cache key.
        """
        masks = masks or [None] * len(texts)
        tag = name if name != "bm25" else f"bm25-k{self.bm25.k1}-b{self.bm25.b}"
        path = self._cache_file(tag, texts, f"{masks_key}|{depth}")
        if cache and path.exists():
            z = np.load(path)
            lengths = z["lengths"]
            offsets = np.concatenate([[0], np.cumsum(lengths)])
            rankings = [
                Ranking(z["idx"][a:b], z["scores"][a:b]) for a, b in itertools.pairwise(offsets)
            ]
            return Run(name, rankings, float(z["seconds_per_query"]))

        start = time.perf_counter()
        rankings: list[Ranking] = []
        if name == "bm25":
            for text, mask in zip(texts, masks, strict=True):
                rankings.append(top_k(self.bm25.score(text), depth, mask))
        else:
            index = self.dense(name.removeprefix("dense-"))
            for lo in range(0, len(texts), 64):
                scores = index.score_batch(texts[lo : lo + 64])
                for row, mask in zip(scores, masks[lo : lo + 64], strict=True):
                    rankings.append(top_k(row, depth, mask))
        seconds = (time.perf_counter() - start) / max(1, len(texts))
        if cache:
            np.savez(
                path,
                idx=np.concatenate([r.idx for r in rankings]) if rankings else np.empty(0),
                scores=np.concatenate([r.scores for r in rankings]) if rankings else np.empty(0),
                lengths=np.array([len(r) for r in rankings]),
                seconds_per_query=seconds,
            )
        return Run(name, rankings, seconds)

    # -- evaluation ----------------------------------------------------------

    def evaluate(self, run: Run, queries: list[Query]) -> dict[str, list[float]]:
        """Per-query case-level and paragraph-level metrics.

        Paragraph-level values are NaN for queries whose quoted passage could not
        be located in the source opinion; ``aggregate`` skips them.
        """
        out: dict[str, list[float]] = {name: [] for name in (*M.CASE_METRICS, *M.PASSAGE_METRICS)}
        for q, ranking in zip(queries, run.rankings, strict=True):
            cases = case_ranking(ranking, self.store)
            for name, fn in M.CASE_METRICS.items():
                out[name].append(fn(cases, q.gold_cases))
            if q.gold_paras:
                hits = chunk_hits(ranking.head(20), self.store, set(q.gold_paras))
                for name, fn in M.PASSAGE_METRICS.items():
                    out[name].append(fn(hits, {"hit"}))
            else:
                for name in M.PASSAGE_METRICS:
                    out[name].append(float("nan"))
        return out


def aggregate(per_query: dict[str, list[float]], seed: int = 0) -> dict[str, dict[str, float]]:
    clean = {k: [v for v in vals if v == v] for k, vals in per_query.items()}
    out = stats.summarise(clean, seed)
    for name, vals in clean.items():
        out[name]["n"] = len(vals)
    return out


def compare(
    a: dict[str, list[float]], b: dict[str, list[float]], metric: str = "ndcg@10", seed: int = 0
) -> dict[str, float]:
    """Paired difference ``a - b`` on one metric with interval and p-value."""
    pairs = [(x, y) for x, y in zip(a[metric], b[metric], strict=True) if x == x and y == y]
    xs, ys = [p[0] for p in pairs], [p[1] for p in pairs]
    diff, low, high = stats.paired_bootstrap(xs, ys, seed=seed)
    return {
        "metric": metric,
        "diff": round(diff, 4),
        "ci_low": round(low, 4),
        "ci_high": round(high, 4),
        "p": round(stats.paired_permutation(xs, ys, seed=seed), 5),
        "n": len(xs),
    }


def adjust(comparisons: dict[str, dict]) -> dict[str, dict]:
    """Add Holm-adjusted p-values across a family of comparisons."""
    names = list(comparisons)
    adjusted = stats.holm([comparisons[n]["p"] for n in names])
    for name, p in zip(names, adjusted, strict=True):
        comparisons[name]["p_holm"] = round(p, 5)
    return comparisons


def mean(values: list[float]) -> float:
    clean = [v for v in values if v == v]
    return float(np.mean(clean)) if clean else float("nan")


def dump(obj) -> str:
    return json.dumps(obj, indent=2)
