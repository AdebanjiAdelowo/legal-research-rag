"""RQ2: does a legal-domain embedding model outperform its general-purpose parent?

The comparison is controlled. Both encoders are run over the same passages, the same
queries and the same relevance labels, with the same sequence limit, numeric
precision, batch size, ranking depth and as-of date filter. Each uses the query and
document prefixes its model card documents; for the default pair these are the same
strings. The experiment refuses to run if any of those conditions differ between
the indexes it is given.

What is reported for each encoder:

* dense retrieval alone, and a hybrid with BM25 by normalised score fusion, on
  ``test``: Recall@1/5/10/20/100, MRR@10, nDCG@10, Precision@10 and paragraph hits;
* encoding time and rate for the corpus, sequence-length truncation, and peak GPU
  memory, from the index build;
* embedding storage on disk and in memory;
* query latency: encoding one query, and exact search over all passages;
* peak GPU memory while encoding queries.

What is tuned, and where: only the BM25 weight of the hybrid, separately for each
encoder, on ``dev``. Nothing is selected on ``test``; both encoders are reported.
The paired differences (legal minus general) carry bootstrap intervals and
randomisation-test p-values with Holm adjustment across the metrics compared.

One limit of the design cannot be removed here. The legal model was fine-tuned on
synthetic queries over U.S. court opinions, and those opinions may include ones in
this corpus. An advantage for it can therefore reflect exposure to the documents as
well as adaptation to the domain.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict

import numpy as np

from legalrag.data.build import query_fingerprint
from legalrag.experiments.common import Bench, Run, adjust, aggregate, compare, mean
from legalrag.index import store as S
from legalrag.index.dense import gpu_peak_mb, reset_gpu_peak
from legalrag.provenance import run_record, write_result
from legalrag.retrieval.core import score_fusion, top_k

ALPHAS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
FUSION_DEPTH = 200
DEPTH = 1000
TARGET = "ndcg@10"
COMPARED = ("ndcg@10", "mrr@10", "recall@1", "recall@10", "recall@100", "passage_hit@10")
# Build settings that must be equal for the comparison to be controlled.
MUST_MATCH = ("max_tokens", "precision", "batch_size", "stored_dtype", "normalised", "device")


def check_identical_conditions(metas: dict[str, dict]) -> dict:
    """Raise unless every index was built from the same passages under the same
    settings. Returns the shared conditions."""
    keys = list(metas)
    first = metas[keys[0]]
    shared = {k: first[k] for k in ("corpus_version", "scheme", "chunks")}
    for key in keys[1:]:
        for field in shared:
            if metas[key][field] != shared[field]:
                raise ValueError(
                    f"{key} was built for {field}={metas[key][field]!r}, "
                    f"{keys[0]} for {shared[field]!r}: not the same passages"
                )
    for field in MUST_MATCH:
        values = {key: (metas[key].get("build") or {}).get(field) for key in keys}
        if len(set(map(str, values.values()))) != 1:
            raise ValueError(f"encoders differ in build setting {field!r}: {values}")
        shared[field] = values[keys[0]]
    return shared


def _fuse(bm25: Run, dense: Run, alpha: float) -> Run:
    return Run(
        f"hybrid(alpha={alpha})",
        [
            score_fusion(a.head(FUSION_DEPTH), b.head(FUSION_DEPTH), alpha)
            for a, b in zip(bm25.rankings, dense.rankings, strict=True)
        ],
    )


def _latency(bench: Bench, key: str, queries, n: int) -> dict:
    """Per-query encode and search time, one query at a time, after a warm-up."""
    index = bench.dense(key)
    sample = queries[:n]
    masks = bench.masks(sample)
    for q in sample[:3]:
        index.score_embedded(index.encoder.encode_queries([q.text]))
    reset_gpu_peak()
    encode, search = [], []
    for q, mask in zip(sample, masks, strict=True):
        start = time.perf_counter()
        emb = index.encoder.encode_queries([q.text])
        encode.append(time.perf_counter() - start)
        start = time.perf_counter()
        top_k(index.score_embedded(emb)[0], DEPTH, mask)
        search.append(time.perf_counter() - start)

    def ms(values: list[float]) -> dict:
        v = np.array(values) * 1000
        return {
            "mean_ms": round(float(v.mean()), 2),
            "p50_ms": round(float(np.median(v)), 2),
            "p95_ms": round(float(np.quantile(v, 0.95)), 2),
        }

    return {
        "queries": len(sample),
        "query_encoding": ms(encode),
        "exact_search": ms(search),
        "search_backend": "numpy inner product on CPU over float32 embeddings",
        "peak_gpu_memory_query_encoding": gpu_peak_mb(),
    }


def _storage(bench: Bench, key: str) -> dict:
    directory = S.index_dir(bench.paths, bench.store, f"dense-{key}")
    emb = bench.dense(key).embeddings
    return {
        "passages": int(emb.shape[0]),
        "dimension": int(emb.shape[1]),
        "dtype_on_disk": str(emb.dtype),
        "disk_mb": round((directory / "embeddings.npy").stat().st_size / 2**20, 1),
        "memory_mb_float32_for_search": round(emb.shape[0] * emb.shape[1] * 4 / 2**20, 1),
    }


def compare_encoders(
    bench: Bench, dev, test, general: str, legal: str, latency_queries: int = 200
) -> dict:
    """Run the comparison for two encoders whose indexes already exist."""
    keys = [general, legal]
    metas = {}
    for key in keys:
        directory = S.index_dir(bench.paths, bench.store, f"dense-{key}")
        meta_file = directory / "meta.json"
        metas[key] = json.loads(meta_file.read_text()) if meta_file.exists() else {
            "corpus_version": bench.store.corpus_version,
            "scheme": bench.store.scheme,
            "chunks": len(bench.dense(key).embeddings),
            "build": None,
        }  # fmt: skip
    conditions = check_identical_conditions(metas)
    if conditions["chunks"] != len(bench.store):
        raise ValueError(
            f"indexes hold {conditions['chunks']} passages, the corpus {len(bench.store)}"
        )

    dev_texts, test_texts = [q.text for q in dev], [q.text for q in test]
    dev_masks, test_masks = bench.masks(dev), bench.masks(test)
    bm25_dev = bench.first_stage("bm25", dev_texts, dev_masks, depth=DEPTH)
    bm25_test = bench.first_stage("bm25", test_texts, test_masks, depth=DEPTH)
    per_query = {"bm25": bench.evaluate(bm25_test, test)}
    systems = {"bm25": aggregate(per_query["bm25"])}
    tuning, encoders = {}, {}

    for key in keys:
        name = f"dense-{key}"
        dense_dev = bench.first_stage(name, dev_texts, dev_masks, depth=DEPTH)
        dense_test = bench.first_stage(name, test_texts, test_masks, depth=DEPTH)
        grid = {a: mean(bench.evaluate(_fuse(bm25_dev, dense_dev, a), dev)[TARGET]) for a in ALPHAS}
        alpha = max(grid, key=grid.get)
        tuning[key] = {
            "dev_dense_ndcg@10": round(mean(bench.evaluate(dense_dev, dev)[TARGET]), 4),
            "alpha_bm25": alpha,
            "dev_alpha_grid": {str(a): round(v, 4) for a, v in grid.items()},
        }
        per_query[name] = bench.evaluate(dense_test, test)
        per_query[f"hybrid-{key}"] = bench.evaluate(_fuse(bm25_test, dense_test, alpha), test)
        systems[name] = aggregate(per_query[name])
        systems[f"hybrid-{key}"] = aggregate(per_query[f"hybrid-{key}"])
        spec = bench.dense(key).spec
        encoders[key] = {
            "spec": asdict(spec),
            "index_build": metas[key].get("build"),
            "storage": _storage(bench, key),
            "latency": _latency(bench, key, test, latency_queries),
        }
        # Release the float32 search copy before the next encoder is loaded.
        bench.dense(key)._emb32 = None

    comparisons = {}
    for kind in ("dense", "hybrid"):
        a, b = per_query[f"{kind}-{legal}"], per_query[f"{kind}-{general}"]
        comparisons[kind] = adjust(
            {f"legal minus general [{m}]": compare(a, b, m) for m in COMPARED}
        )
    for key in keys:
        comparisons[f"dense-{key} vs bm25"] = compare(per_query[f"dense-{key}"], per_query["bm25"])
        comparisons[f"hybrid-{key} vs bm25"] = compare(
            per_query[f"hybrid-{key}"], per_query["bm25"]
        )

    return {
        "identical_conditions": {
            **conditions,
            "dev_queries": {"count": len(dev), "sha256": query_fingerprint(dev)},
            "test_queries": {"count": len(test), "sha256": query_fingerprint(test)},
            "ranking_depth": DEPTH,
            "fusion_depth": FUSION_DEPTH,
            "as_of_filter": True,
            "relevance": "the case quoted by the citing court; paragraph labels by location",
            "prefixes_equal": len(
                {(e["spec"]["query_prefix"], e["spec"]["doc_prefix"]) for e in encoders.values()}
            )
            == 1,
        },
        "tuned_on_dev": tuning,
        "encoders": encoders,
        "systems": systems,
        "comparisons": comparisons,
    }


def run(
    general: str = "modernbert-general",
    legal: str = "modernbert-legal",
    scheme: str = "section220",
    limit: int | None = None,
    k1: float | None = None,
    b: float | None = None,
    latency_queries: int = 200,
):
    bench = Bench.load(scheme)
    for key in (general, legal):
        if not bench.has_dense(key):
            raise FileNotFoundError(
                f"no dense index for {key}; build it with `legalrag index {key}`"
            )
    if k1 is not None and b is not None:
        bench.bm25.set_params(float(k1), float(b))
    dev, test = bench.queries("dev", limit), bench.queries("test", limit)
    body = compare_encoders(bench, dev, test, general, legal, int(latency_queries))
    result = run_record(
        {
            "experiment": "encoders",
            "general": general,
            "legal": legal,
            "scheme": scheme,
            "bm25": {"k1": bench.bm25.k1, "b": bench.bm25.b},
            "n_dev": len(dev),
            "n_test": len(test),
            "chunks": len(bench.store),
            "tuned_on": "dev",
            "reported_on": "test",
        },
        bench.store.corpus_version,
        **body,
    )
    name = f"encoders_{general}_vs_{legal}" + (f"_limit{limit}" if limit else "")
    write_result(bench.paths.results / "encoders" / f"{name}.json", result)
    return result
