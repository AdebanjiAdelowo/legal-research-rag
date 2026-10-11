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


class _HashEncoder:
    """Deterministic stand-in for a sentence encoder: hashed bag of words."""

    def __init__(self, spec, dim=64, drop=()):
        self.spec, self.dim, self.drop = spec, dim, set(drop)

    def _vec(self, text):
        import hashlib

        from legalrag.index.bm25 import tokenize

        v = np.zeros(self.dim, dtype=np.float32)
        for tok in tokenize(text):
            if tok not in self.drop:
                v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
        return v / (np.linalg.norm(v) or 1.0)

    def encode_queries(self, texts, batch_size=64):
        return np.stack([self._vec(self.spec.query_prefix + t) for t in texts])

    def encode_docs(self, texts, batch_size=64, progress=False):
        return np.stack([self._vec(self.spec.doc_prefix + t) for t in texts])


def _encoder_bench(tmp_path, build_overrides=None):
    import json

    from legalrag.index import store as S
    from legalrag.index.dense import DenseIndex, EncoderSpec

    bench = make_bench(tmp_path)
    build = {
        "max_tokens": 512,
        "precision": "float32",
        "batch_size": 8,
        "stored_dtype": "float16",
        "normalised": True,
        "device": "cpu",
        "encode_seconds": 1.0,
        "docs_per_second": 19.0,
        "tokens": {"passages_truncated": 0},
        "peak_gpu_memory": None,
    }
    for key, drop in (("fake-general", ("immunity", "injunction")), ("fake-legal", ())):
        spec = EncoderSpec(key, f"test/{key}", "0" * 40, key.split("-")[1], "q: ", "d: ")
        enc = _HashEncoder(spec, drop=drop)
        index = DenseIndex(enc.encode_docs(bench.store.texts).astype(np.float16), spec, enc)
        meta = {
            "corpus_version": bench.store.corpus_version,
            "scheme": bench.store.scheme,
            "chunks": len(bench.store),
            "build": {**build, **(build_overrides or {}).get(key, {})},
        }
        index.save(S.index_dir(bench.paths, bench.store, f"dense-{key}"), meta)
        bench._dense[key] = index
    return bench, json


def test_encoder_comparison_reports_everything_under_identical_conditions(tmp_path):
    from legalrag.experiments.encoders import COMPARED, compare_encoders

    bench, _ = _encoder_bench(tmp_path)
    queries = demo.queries()
    out = compare_encoders(bench, queries, queries, "fake-general", "fake-legal", latency_queries=3)
    cond = out["identical_conditions"]
    assert (
        cond["chunks"] == len(bench.store) and cond["max_tokens"] == 512 and cond["prefixes_equal"]
    )
    assert (
        cond["dev_queries"]["sha256"] == cond["test_queries"]["sha256"]
    )  # same list passed twice here
    assert set(out["systems"]) == {
        "bm25",
        "dense-fake-general",
        "dense-fake-legal",
        "hybrid-fake-general",
        "hybrid-fake-legal",
    }
    for name in out["systems"]:
        assert {"recall@10", "recall@100", "mrr@10", "ndcg@10"} <= set(out["systems"][name])
    for kind in ("dense", "hybrid"):
        assert set(out["comparisons"][kind]) == {f"legal minus general [{m}]" for m in COMPARED}
        assert all("p_holm" in c and "ci_low" in c for c in out["comparisons"][kind].values())
    for key in ("fake-general", "fake-legal"):
        e = out["encoders"][key]
        assert e["spec"]["revision"] == "0" * 40 and e["index_build"]["encode_seconds"] == 1.0
        assert e["storage"]["dimension"] == 64 and e["storage"]["disk_mb"] >= 0
        assert e["latency"]["query_encoding"]["mean_ms"] >= 0 and e["latency"]["queries"] == 3
        assert 0.1 <= out["tuned_on_dev"][key]["alpha_bm25"] <= 0.9
    # The stand-in "general" encoder cannot see two legal terms, so it must not win.
    d = out["comparisons"]["dense"]["legal minus general [recall@10]"]["diff"]
    assert d >= 0


def test_encoder_comparison_refuses_unequal_conditions(tmp_path):
    import pytest

    from legalrag.experiments.encoders import check_identical_conditions, compare_encoders

    bench, _ = _encoder_bench(tmp_path, {"fake-legal": {"max_tokens": 256}})
    with pytest.raises(ValueError, match="max_tokens"):
        compare_encoders(bench, demo.queries(), demo.queries(), "fake-general", "fake-legal", 2)
    base = {"corpus_version": "a", "scheme": "s", "chunks": 5, "build": {}}
    with pytest.raises(ValueError, match="not the same passages"):
        check_identical_conditions({"x": base, "y": {**base, "chunks": 4}})


def test_fingerprint_detects_any_change():
    from legalrag.data.build import chunk_fingerprint, fingerprint_differences, query_fingerprint

    queries = demo.queries()
    base = query_fingerprint(queries)
    assert base == query_fingerprint(demo.queries())
    import copy

    changed = copy.deepcopy(queries)
    changed[0].gold_cases = ["cap:9002"]
    assert query_fingerprint(changed) != base
    assert query_fingerprint(queries[:-1]) != base
    chunks = demo.chunks()
    assert chunk_fingerprint(chunks) != chunk_fingerprint(chunks[:-1])
    expected = {"corpus_version": "x", "chunks": {"s": {"count": 3, "sha256": "aa"}}}
    assert fingerprint_differences(expected, expected) == []
    diffs = fingerprint_differences(
        {"corpus_version": "x", "chunks": {"s": {"count": 2, "sha256": "aa"}}}, expected
    )
    assert diffs == ["chunks.s.count: got 2, expected 3"]
    assert fingerprint_differences({"corpus_version": "x"}, expected)[0].startswith(
        "chunks: missing"
    )
