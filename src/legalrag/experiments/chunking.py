"""Fixed-size against section-aware chunking.

Both schemes index the same opinions and are queried with the same texts. BM25
parameters are tuned separately for each on ``dev``, since chunk length interacts
with length normalisation. Three things are compared on ``test``:

* case-level ranking (does the right opinion surface),
* paragraph-level hits (does a returned chunk contain the quoted paragraph),
* reading cost: words in the top ``k`` chunks.

The schemes differ in mean chunk length, so a paragraph hit is easier for the scheme
with longer chunks. Hits per thousand words read is reported to make that visible.
"""

from __future__ import annotations

import gc

from legalrag.experiments.common import Bench, adjust, aggregate, compare, mean
from legalrag.provenance import run_record, write_result
from legalrag.retrieval.core import words_at_k

SCHEMES = ("section220", "fixed220")
GRID = [(0.9, 0.5), (0.9, 0.75), (1.2, 0.5), (1.2, 0.75), (1.6, 0.75), (2.0, 0.75), (1.6, 0.9)]


def _one(scheme: str, limit: int | None):
    bench = Bench.load(scheme)
    dev, test = bench.queries("dev", limit), bench.queries("test", limit)
    best = (-1.0, None)
    for k1, b in GRID:
        bench.bm25.set_params(k1, b)
        run = bench.first_stage("bm25", [q.text for q in dev], bench.masks(dev), depth=200)
        score = mean(bench.evaluate(run, dev)["ndcg@10"])
        if score > best[0]:
            best = (score, (k1, b))
    bench.bm25.set_params(*best[1])
    run = bench.first_stage("bm25", [q.text for q in test], bench.masks(test))
    per_query = bench.evaluate(run, test)
    words = {k: [float(words_at_k(r, bench.store, k)) for r in run.rankings] for k in (5, 10)}
    per_query["words@5"], per_query["words@10"] = words[5], words[10]
    info = {
        "chunks": len(bench.store),
        "bm25": {"k1": best[1][0], "b": best[1][1], "dev_ndcg@10": round(best[0], 4)},
        "n_test": len(test),
        "corpus_version": bench.store.corpus_version,
        "results_dir": bench.paths.results,
    }
    del bench
    gc.collect()
    return per_query, info


def run(limit: int | None = None):
    per_query, info = {}, {}
    for scheme in SCHEMES:
        per_query[scheme], info[scheme] = _one(scheme, limit)

    systems = {}
    for scheme in SCHEMES:
        agg = aggregate(per_query[scheme])
        for k in (5, 10):
            hit = agg[f"passage_hit@{k}"]["mean"]
            agg[f"passage_hits_per_1000_words@{k}"] = round(
                1000 * hit / agg[f"words@{k}"]["mean"], 4
            )
        systems[scheme] = agg

    a, b = per_query["section220"], per_query["fixed220"]
    comparisons = adjust(
        {
            f"section vs fixed [{m}]": compare(a, b, m)
            for m in ("ndcg@10", "recall@10", "passage_hit@5", "passage_hit@10", "passage_mrr@10")
        }
    )
    results_dir = info["section220"].pop("results_dir")
    info["fixed220"].pop("results_dir")
    result = run_record(
        {"experiment": "chunking", "retriever": "bm25", "tuned_on": "dev", "reported_on": "test"},
        info["section220"]["corpus_version"],
        schemes=info,
        systems=systems,
        comparisons=comparisons,
    )
    write_result(results_dir / "chunking" / "chunking_bm25.json", result)
    return result
