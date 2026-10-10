"""RQ1 and RQ2: BM25, dense and hybrid retrieval; general against legal encoders.

Tuned on ``dev``: BM25 ``k1`` and ``b``; the fusion weight for score fusion; the
constant of reciprocal rank fusion. Everything is then frozen and run on ``test``.

Also reports the leakage control: the same systems on unmasked queries, in which
the citing opinion's own citation to the answer is still present.
"""

from __future__ import annotations

import itertools

from legalrag.experiments.common import Bench, Run, adjust, aggregate, compare, mean
from legalrag.index.dense import ENCODERS
from legalrag.provenance import run_record, write_result
from legalrag.retrieval.core import rrf, score_fusion

BM25_GRID = list(itertools.product((0.6, 0.9, 1.2, 1.6, 2.0), (0.3, 0.5, 0.75, 0.9)))
ALPHAS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
RRF_K = (10, 30, 60, 100)
FUSION_DEPTH = 200
TARGET = "ndcg@10"


def tune_bm25(bench: Bench, dev) -> tuple[float, float, list[dict]]:
    texts = [q.text for q in dev]
    masks = bench.masks(dev)
    grid = []
    for k1, b in BM25_GRID:
        bench.bm25.set_params(k1, b)
        run = bench.first_stage("bm25", texts, masks, depth=200)
        grid.append({"k1": k1, "b": b, TARGET: round(mean(bench.evaluate(run, dev)[TARGET]), 4)})
    best = max(grid, key=lambda g: g[TARGET])
    bench.bm25.set_params(best["k1"], best["b"])
    return best["k1"], best["b"], grid


def fuse_rrf(a: Run, b: Run, k: int) -> Run:
    return Run(
        f"rrf(k={k})",
        [rrf([x.head(FUSION_DEPTH), y.head(FUSION_DEPTH)], k) for x, y in zip(
            a.rankings, b.rankings, strict=True)],
        a.seconds_per_query + b.seconds_per_query,
    )  # fmt: skip


def fuse_scores(a: Run, b: Run, alpha: float) -> Run:
    return Run(
        f"score-fusion(alpha={alpha})",
        [score_fusion(x.head(FUSION_DEPTH), y.head(FUSION_DEPTH), alpha) for x, y in zip(
            a.rankings, b.rankings, strict=True)],
        a.seconds_per_query + b.seconds_per_query,
    )  # fmt: skip


def run(scheme: str = "section220", encoders: list[str] | None = None, limit: int | None = None):
    bench = Bench.load(scheme)
    dev, test = bench.queries("dev", limit), bench.queries("test", limit)
    encoders = [e for e in (encoders or list(ENCODERS)) if bench.has_dense(e)]

    k1, b, bm25_grid = tune_bm25(bench, dev)
    dev_texts, test_texts = [q.text for q in dev], [q.text for q in test]
    dev_masks, test_masks = bench.masks(dev), bench.masks(test)

    dev_runs = {"bm25": bench.first_stage("bm25", dev_texts, dev_masks)}
    test_runs = {"bm25": bench.first_stage("bm25", test_texts, test_masks)}
    for key in encoders:
        dev_runs[f"dense-{key}"] = bench.first_stage(f"dense-{key}", dev_texts, dev_masks)
        test_runs[f"dense-{key}"] = bench.first_stage(f"dense-{key}", test_texts, test_masks)

    dev_scores = {n: mean(bench.evaluate(r, dev)[TARGET]) for n, r in dev_runs.items()}
    tuning: dict = {"bm25": {"k1": k1, "b": b, "grid": bm25_grid}, "dev": dev_scores}

    # Hybrids pair BM25 with each dense encoder; weights are chosen per encoder on dev.
    for key in encoders:
        name = f"dense-{key}"
        alpha_grid = {
            a: mean(bench.evaluate(fuse_scores(dev_runs["bm25"], dev_runs[name], a), dev)[TARGET])
            for a in ALPHAS
        }
        k_grid = {
            k: mean(bench.evaluate(fuse_rrf(dev_runs["bm25"], dev_runs[name], k), dev)[TARGET])
            for k in RRF_K
        }
        alpha = max(alpha_grid, key=alpha_grid.get)
        k = max(k_grid, key=k_grid.get)
        tuning[f"hybrid-{key}"] = {
            "alpha_bm25": alpha,
            "alpha_grid": {str(a): round(v, 4) for a, v in alpha_grid.items()},
            "rrf_k": k,
            "rrf_grid": {str(a): round(v, 4) for a, v in k_grid.items()},
        }
        test_runs[f"hybrid-rrf-{key}"] = fuse_rrf(test_runs["bm25"], test_runs[name], k)
        test_runs[f"hybrid-score-{key}"] = fuse_scores(test_runs["bm25"], test_runs[name], alpha)

    per_query = {n: bench.evaluate(r, test) for n, r in test_runs.items()}
    systems = {
        n: {
            **aggregate(per_query[n]),
            "seconds_per_query": round(test_runs[n].seconds_per_query, 4),
        }
        for n in test_runs
    }

    comparisons: dict[str, dict] = {}
    for key in encoders:
        comparisons[f"dense-{key} vs bm25"] = compare(per_query[f"dense-{key}"], per_query["bm25"])
        best_single = max(("bm25", f"dense-{key}"), key=lambda n: systems[n][TARGET]["mean"])
        for kind in ("rrf", "score"):
            comparisons[f"hybrid-{kind}-{key} vs {best_single}"] = compare(
                per_query[f"hybrid-{kind}-{key}"], per_query[best_single]
            )
        comparisons[f"hybrid-score-{key} vs hybrid-rrf-{key}"] = compare(
            per_query[f"hybrid-score-{key}"], per_query[f"hybrid-rrf-{key}"]
        )
    if "modernbert-legal" in encoders and "modernbert-general" in encoders:
        for metric in (TARGET, "recall@10", "passage_hit@10"):
            comparisons[f"legal vs general encoder [{metric}]"] = compare(
                per_query["dense-modernbert-legal"], per_query["dense-modernbert-general"], metric
            )
    adjust(comparisons)

    # Leakage control: unmasked queries, in which citations and case names remain.
    raw_texts = [q.raw_text for q in test]
    leakage = {}
    for name in ["bm25"] + [f"dense-{k}" for k in encoders]:
        raw_run = bench.first_stage(name, raw_texts, test_masks)
        raw_pq = bench.evaluate(raw_run, test)
        leakage[name] = {
            "masked": systems[name][TARGET]["mean"],
            "unmasked": round(mean(raw_pq[TARGET]), 4),
            "difference": compare(raw_pq, per_query[name]),
        }

    result = run_record(
        {
            "experiment": "retrieval",
            "scheme": scheme,
            "encoders": encoders,
            "depth": 1000,
            "fusion_depth": FUSION_DEPTH,
            "as_of_filter": True,
            "tuned_on": "dev",
            "reported_on": "test",
            "n_dev": len(dev),
            "n_test": len(test),
            "chunks": len(bench.store),
        },
        bench.store.corpus_version,
        tuning=tuning,
        systems=systems,
        comparisons=comparisons,
        leakage_control=leakage,
    )
    name = "retrieval" if not limit else f"retrieval_limit{limit}"
    write_result(bench.paths.results / "retrieval" / f"{name}_{scheme}.json", result)
    return result
