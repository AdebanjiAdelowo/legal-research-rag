"""Latency and resource use of each stage.

Timings are wall-clock per query on the machine that runs the experiment, taken
one query at a time (no batching across queries), after a warm-up. They describe
this implementation on this hardware and are not comparable across machines.
"""

from __future__ import annotations

import json
import resource
import sys
import time

import numpy as np

from legalrag.experiments.common import Bench
from legalrag.index import store as S
from legalrag.provenance import run_record, write_result
from legalrag.rag.evidence import select_evidence
from legalrag.rag.generate import ExtractiveGenerator
from legalrag.retrieval.core import rrf, top_k
from legalrag.verify.judges import make_judge
from legalrag.verify.verifier import CitationIndex, CitationVerifier


def _summary(seconds: list[float]) -> dict:
    ms = np.array(seconds) * 1000
    return {
        "mean_ms": round(float(ms.mean()), 2),
        "p50_ms": round(float(np.median(ms)), 2),
        "p95_ms": round(float(np.quantile(ms, 0.95)), 2),
        "n": len(seconds),
    }


def _timed(fn, items, warmup: int = 3):
    for item in items[:warmup]:
        fn(item)
    out, seconds = [], []
    for item in items:
        start = time.perf_counter()
        out.append(fn(item))
        seconds.append(time.perf_counter() - start)
    return out, seconds


def _dir_mb(path) -> float:
    return round(sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6, 1)


def run(limit: int = 200, encoder: str | None = None, reranker: str | None = None,
        judge: str = "lexical", top_n: int = 50):  # fmt: skip
    bench = Bench.load()
    queries = bench.queries("test", int(limit))
    masks = bench.masks(queries)
    items = list(zip(queries, masks, strict=True))
    stages: dict[str, dict] = {}

    lexical, t = _timed(lambda qm: top_k(bench.bm25.score(qm[0].text), 200, qm[1]), items)
    stages["bm25_retrieval"] = _summary(t)
    rankings = lexical

    if encoder and bench.has_dense(encoder):
        index = bench.dense(encoder)
        embedded, t = _timed(lambda qm: index.encoder.encode_queries([qm[0].text]), items)
        stages["dense_query_encoding"] = _summary(t)
        dense, t = _timed(
            lambda x: top_k(index.score_embedded(x[0])[0], 200, x[1]),
            list(zip(embedded, masks, strict=True)),
        )
        stages["dense_search_exact"] = _summary(t)
        rankings, t = _timed(
            lambda ab: rrf([ab[0], ab[1]], 60), list(zip(lexical, dense, strict=True))
        )
        stages["rank_fusion"] = _summary(t)

    if reranker:
        from legalrag.retrieval.rerank import Reranker

        model = Reranker(reranker)
        rankings, t = _timed(
            lambda rq: model.rerank(rq[1].text, rq[0], bench.store, int(top_n)),
            list(zip(rankings, queries, strict=True)),
        )
        stages[f"reranking_top{top_n}"] = {**_summary(t), "model": model.spec.model,
                                            "device": model.device}  # fmt: skip

    generator = ExtractiveGenerator()
    evidence, t = _timed(lambda r: select_evidence(r, bench.store, 5, 2), rankings)
    stages["evidence_selection"] = _summary(t)
    answers, t = _timed(
        lambda qe: generator.generate(qe[0].text, qe[1]), list(zip(queries, evidence, strict=True))
    )
    stages["generation_extractive"] = _summary(t)
    verifier = CitationVerifier(CitationIndex(bench.store.cases), make_judge(judge))
    _, t = _timed(
        lambda ae: verifier.check_statements(ae[0][1], ae[1]),
        list(zip(answers, evidence, strict=True)),
    )
    stages[f"citation_verification[{verifier.judge.name}]"] = _summary(t)

    root = bench.paths.indexes / bench.store.corpus_version / bench.store.scheme
    indexes = {}
    for directory in sorted(p for p in root.iterdir() if (p / "meta.json").exists()):
        meta = json.loads((directory / "meta.json").read_text())
        indexes[directory.name] = {"disk_mb": _dir_mb(directory), "build": meta.get("build")}
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result = run_record(
        {
            "experiment": "systems",
            "n_queries": len(queries),
            "encoder": encoder,
            "reranker": reranker,
            "judge": judge,
        },
        bench.store.corpus_version,
        stages=stages,
        indexes=indexes,
        storage_mb={
            "raw_opinions": _dir_mb(bench.paths.cap_cases),
            "processed": _dir_mb(bench.paths.processed),
            "indexes": _dir_mb(S.index_dir(bench.paths, bench.store, "x").parent),
        },
        corpus={"cases": len(bench.store.cases), "chunks": len(bench.store)},
        peak_resident_memory_gb=round(peak / (1e9 if sys.platform == "darwin" else 1e6), 2),
    )
    tag = "+".join(x for x in ("bm25", encoder, reranker) if x)
    write_result(bench.paths.results / "systems" / f"systems_{tag}.json", result)
    return result
