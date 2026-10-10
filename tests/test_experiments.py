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


def test_negation_and_planted_corruptions(tmp_path):
    import random

    from legalrag.experiments import citations as C
    from legalrag.rag.evidence import select_evidence
    from legalrag.rag.generate import ExtractiveGenerator
    from legalrag.retrieval.core import top_k
    from legalrag.verify.verifier import CitationIndex, CitationVerifier

    assert C.negate("The sequence is mandatory.") == "The sequence is not mandatory."
    assert C.negate("Courts decide.") is None

    bench = make_bench(tmp_path)
    q = demo.queries()[4]
    ranking = top_k(bench.bm25.score(q.text), 20, bench.store.mask(before=q.date))
    evidence = select_evidence(ranking, bench.store, k=5, per_case=2)
    _, statements, _ = ExtractiveGenerator().generate(q.text, evidence)
    clean = next(s for s in statements if s.citations)
    verifier = CitationVerifier(CitationIndex(bench.store.cases))
    assert verifier.check_statements([clean], evidence)[0].status == "verified"
    rng = random.Random(0)
    for kind in C.LOOKUP:
        bad = C.corrupt(kind, clean, evidence, bench, rng)
        if bad is None:
            continue
        check = verifier.check_statements([bad], evidence)[0]
        assert C._detected(kind, check), kind
        assert check.status != "verified", kind


def test_judge_pairs_and_calibration():
    from legalrag.experiments import judge as J
    from legalrag.verify.judges import Judgment

    verdicts = {
        "supported": [Judgment("", 0.9), Judgment("", 0.8), Judgment("", 0.4)],
        "other_case": [Judgment("", 0.1), Judgment("", 0.2), Judgment("", 0.3)],
        "same_case": [Judgment("", 0.1), Judgment("", 0.35), Judgment("", 0.2)],
        "negated": [Judgment("", 0.6), Judgment("", 0.2), Judgment("", 0.1)],
    }
    at_half = J._rates(verdicts, 0.5)
    assert at_half["recall_on_supported"] == round(2 / 3, 4)
    assert at_half["accept_rate"]["negated"]["accepted"] == round(1 / 3, 4)
    assert at_half["accuracy"] == round(10 / 12, 4)
    t = J._calibrate(verdicts)
    assert J._rates(verdicts, t)["accuracy"] >= at_half["accuracy"]


def test_error_analysis_buckets():
    from legalrag.experiments.analysis import _bucket, _table

    labels = ["<3k", "3k-7k", "7k-15k", ">=15k"]
    assert [_bucket(v, [3000, 7000, 15000], labels) for v in (10, 3000, 14999, 20000)] == labels
    out = _table({"a": [1.0, 0.0, 1.0, 1.0], "b": []})
    assert out["a"]["n"] == 4 and out["a"]["recall@10"] == 0.75 and "b" not in out
