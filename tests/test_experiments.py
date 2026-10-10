"""The experiment harness on the synthetic corpus: no data, model or network."""

import numpy as np

from legalrag import demo
from legalrag.experiments.common import Bench, adjust, aggregate, compare
from legalrag.index.bm25 import BM25Index
from legalrag.paths import Paths


def make_bench(tmp_path) -> Bench:
    store = demo.store()
    return Bench(Paths(tmp_path), store, BM25Index.build(store.texts))


def test_cached_run_equals_fresh_run_and_shares_two_buffers(tmp_path):
    bench = make_bench(tmp_path)
    queries = demo.queries()
    texts, masks = [q.text for q in queries], bench.masks(queries)
    fresh = bench.first_stage("bm25", texts, masks, depth=10)
    cached = bench.first_stage("bm25", texts, masks, depth=10)
    for a, b in zip(fresh.rankings, cached.rankings, strict=True):
        assert a.idx.tolist() == b.idx.tolist() and np.allclose(a.scores, b.scores)
    # Regression: every cached ranking must be a view of one loaded array, not of
    # its own copy of the whole file.
    assert len({id(r.idx.base) for r in cached.rankings}) == 1
    assert len({id(r.scores.base) for r in cached.rankings}) == 1


def test_cache_key_separates_filters(tmp_path):
    bench = make_bench(tmp_path)
    q = demo.queries()[4:5]  # answered by a 2009 case
    text = [q[0].text]
    late = bench.first_stage("bm25", text, bench.masks(q), "as_of", depth=5)
    early = bench.first_stage(
        "bm25", text, [bench.store.mask(before="2005-01-01")], "early", depth=5
    )
    assert late.rankings[0].idx.tolist() != early.rankings[0].idx.tolist()


def test_evaluate_and_compare(tmp_path):
    bench = make_bench(tmp_path)
    queries = demo.queries()
    run = bench.first_stage("bm25", [q.text for q in queries], bench.masks(queries), depth=10)
    per_query = bench.evaluate(run, queries)
    summary = aggregate(per_query)
    assert summary["recall@1"]["mean"] == 1.0 and summary["recall@1"]["n"] == len(queries)
    assert summary["passage_hit@5"]["mean"] == 1.0
    worse = {k: [0.0] * len(v) for k, v in per_query.items()}
    cmp = adjust({"a vs b": compare(per_query, worse, "recall@1")})
    assert cmp["a vs b"]["diff"] == 1.0 and "p_holm" in cmp["a vs b"]
