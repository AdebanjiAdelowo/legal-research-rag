"""RQ3: how much does cross-encoder reranking improve legal relevance, and at what
cost?

The first stage is fixed (BM25, or a hybrid when a dense index exists). The
cross-encoder rescoring depth ``top_n`` is chosen on ``dev`` from a small grid and
then applied on ``test``. Reported: ranking quality with and without reranking, the
paired difference, and the added latency per query.

A reranker can only reorder what the first stage returned, so its ceiling is the
first stage's recall at ``top_n``; that ceiling is reported alongside.
"""

from __future__ import annotations

import time

from legalrag.eval.metrics import recall_at_k
from legalrag.experiments.common import Bench, Run, adjust, aggregate, compare, mean
from legalrag.provenance import run_record, write_result
from legalrag.retrieval.core import case_ranking, rrf, score_fusion
from legalrag.retrieval.rerank import Reranker, apply_scores, head_texts

FUSION_DEPTH = 200


def _first_stage(bench: Bench, queries, encoder: str | None, fusion: str, alpha: float, k: int):
    texts, masks = [q.text for q in queries], bench.masks(queries)
    lexical = bench.first_stage("bm25", texts, masks)
    if not encoder:
        return lexical
    dense = bench.first_stage(f"dense-{encoder}", texts, masks)
    fused = []
    for a, b in zip(lexical.rankings, dense.rankings, strict=True):
        a, b = a.head(FUSION_DEPTH), b.head(FUSION_DEPTH)
        fused.append(score_fusion(a, b, alpha) if fusion == "score" else rrf([a, b], k))
    return Run(f"hybrid-{fusion}-{encoder}", fused, lexical.seconds_per_query)


def _rerank(bench: Bench, run: Run, queries, model: Reranker, top_n: int) -> Run:
    start = time.perf_counter()
    out = [
        apply_scores(r, model.score(q.text, head_texts(r, bench.store, top_n)))
        for r, q in zip(run.rankings, queries, strict=True)
    ]
    return Run(f"{run.name}+{model.spec.key}@{top_n}", out,
               (time.perf_counter() - start) / max(1, len(queries)))  # fmt: skip


def run(
    reranker: str = "minilm-l6",
    encoder: str | None = None,
    fusion: str = "rrf",
    alpha: float = 0.5,
    rrf_k: int = 60,
    limit: int | None = None,
    dev_limit: int = 300,
    k1: float | None = None,
    b: float | None = None,
):
    bench = Bench.load()
    if k1 is not None and b is not None:
        bench.bm25.set_params(float(k1), float(b))
    model = Reranker(reranker)
    dev, test = bench.queries("dev", int(dev_limit)), bench.queries("test", limit)

    dev_run = _first_stage(bench, dev, encoder, fusion, float(alpha), int(rrf_k))
    grid = {}
    for top_n in (20, 50, 100):
        reranked = _rerank(bench, dev_run, dev, model, top_n)
        grid[top_n] = {
            "ndcg@10": round(mean(bench.evaluate(reranked, dev)["ndcg@10"]), 4),
            "seconds_per_query": round(reranked.seconds_per_query, 4),
        }
    best_n = max(grid, key=lambda n: grid[n]["ndcg@10"])

    base = _first_stage(bench, test, encoder, fusion, float(alpha), int(rrf_k))
    reranked = _rerank(bench, base, test, model, best_n)
    pq_base, pq_rr = bench.evaluate(base, test), bench.evaluate(reranked, test)
    ceiling = mean(
        [
            recall_at_k(case_ranking(r.head(best_n), bench.store), q.gold_cases, best_n)
            for r, q in zip(base.rankings, test, strict=True)
        ]
    )
    result = run_record(
        {
            "experiment": "rerank",
            "first_stage": base.name,
            "reranker": model.spec.model,
            "device": model.device,
            "bm25": {"k1": bench.bm25.k1, "b": bench.bm25.b},
            "n_dev": len(dev),
            "n_test": len(test),
        },
        bench.store.corpus_version,
        dev_grid={str(k): v for k, v in grid.items()},
        top_n=best_n,
        first_stage=aggregate(pq_base),
        reranked={
            **aggregate(pq_rr),
            "rerank_seconds_per_query": round(reranked.seconds_per_query, 4),
        },
        case_recall_ceiling_in_reranked_chunks=round(ceiling, 4),
        comparisons=adjust(
            {
                f"reranked vs first stage [{m}]": compare(pq_rr, pq_base, m)
                for m in ("ndcg@10", "recall@10", "mrr@10", "passage_hit@10")
            }
        ),
    )
    write_result(bench.paths.results / "rerank" / f"rerank_{base.name}_{reranker}.json", result)
    return result
