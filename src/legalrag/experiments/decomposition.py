"""RQ5: does query decomposition help multi-issue questions, and what does it cost
single-issue ones?

Multi-issue queries are built from one citing opinion: two or three passages of it,
each quoting a different earlier case, are joined into one query, and all of those
cases are relevant. This is a constructed benchmark: the join is by concatenation,
so the issues are easier to separate than in a question a person would write.

Conditions:

``none``       the whole query, one retrieval
``segments``   lexical-cohesion segments, fused with reciprocal rank fusion
``sentences``  one sub-query per sentence, fused
``oracle``     the true parts the query was assembled from: an upper bound that no
               deployed system can reach, since it uses the benchmark's construction

The same conditions are run on ordinary single-issue test queries to measure harm.
"""

from __future__ import annotations

from legalrag.eval.metrics import recall_at_k
from legalrag.experiments.common import Bench, Run, adjust, aggregate, compare
from legalrag.provenance import run_record, write_result
from legalrag.retrieval.core import case_ranking, rrf
from legalrag.retrieval.decompose import decompose

DEPTH = 200


def _decomposed(bench: Bench, queries, parts_of, name: str) -> tuple[Run, float]:
    """Retrieve each part of each query and fuse. Returns the run and the mean
    number of sub-queries."""
    flat, owner = [], []
    for i, q in enumerate(queries):
        for part in parts_of(q):
            flat.append(part)
            owner.append(i)
    masks = bench.masks(queries)
    run = bench.first_stage(
        "bm25", flat, [masks[i] for i in owner], masks_key=f"as_of|{name}", depth=DEPTH
    )
    grouped: list[list] = [[] for _ in queries]
    for ranking, i in zip(run.rankings, owner, strict=True):
        grouped[i].append(ranking)
    fused = [g[0] if len(g) == 1 else rrf(g, 60) for g in grouped]
    seconds = run.seconds_per_query * len(flat) / max(1, len(queries))
    return Run(name, fused, seconds), len(flat) / max(1, len(queries))


def _evaluate(bench: Bench, queries, with_oracle: bool):
    conditions = {
        "none": lambda q: [q.text],
        "segments": lambda q: decompose(q.text, "segments"),
        "sentences": lambda q: decompose(q.text, "sentences"),
    }
    if with_oracle:
        conditions["oracle"] = lambda q: [q.text, *q.parts]
    per_query, systems = {}, {}
    for name, fn in conditions.items():
        run, n_parts = _decomposed(bench, queries, fn, name)
        pq = bench.evaluate(run, queries)
        # Share of queries for which every relevant case is in the top 10 and top 20.
        for k in (10, 20):
            pq[f"all_gold@{k}"] = [
                float(recall_at_k(case_ranking(r, bench.store), q.gold_cases, k) == 1.0)
                for r, q in zip(run.rankings, queries, strict=True)
            ]
        per_query[name] = pq
        systems[name] = {
            **aggregate(pq),
            "sub_queries_mean": round(n_parts, 2),
            "seconds_per_query": round(run.seconds_per_query, 4),
        }
    comparisons = adjust(
        {
            f"{name} vs none [{m}]": compare(per_query[name], per_query["none"], m)
            for name in conditions
            if name != "none"
            for m in ("recall@10", "ndcg@10")
        }
    )
    return systems, comparisons


def run(limit: int | None = None, k1: float | None = None, b: float | None = None):
    bench = Bench.load()
    if k1 is not None and b is not None:
        bench.bm25.set_params(float(k1), float(b))
    multi = bench.queries("multi_test", limit)
    single = bench.queries("test", limit or 1000)
    multi_systems, multi_cmp = _evaluate(bench, multi, with_oracle=True)
    single_systems, single_cmp = _evaluate(bench, single, with_oracle=False)
    result = run_record(
        {
            "experiment": "decomposition",
            "retriever": "bm25",
            "bm25": {"k1": bench.bm25.k1, "b": bench.bm25.b},
            "fusion": "rrf(k=60)",
            "depth": DEPTH,
            "n_multi": len(multi),
            "n_single": len(single),
            "gold_cases_per_multi_query": round(
                sum(len(q.gold_cases) for q in multi) / max(1, len(multi)), 2
            ),
        },
        bench.store.corpus_version,
        multi_issue={"systems": multi_systems, "comparisons": multi_cmp},
        single_issue={"systems": single_systems, "comparisons": single_cmp},
    )
    write_result(bench.paths.results / "decomposition" / "decomposition_bm25.json", result)
    return result
