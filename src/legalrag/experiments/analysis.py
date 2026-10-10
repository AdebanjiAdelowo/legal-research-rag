"""Error analysis for a first-stage retriever on the test split.

Two views of the same run.

Outcome categories, one per query, by where the quoted case first appears:

``passage_top10``   a chunk holding the quoted paragraph is in the top 10
``case_top10``      the right case is in the top 10 cases, but through a different
                    passage: the opinion was found, the cited paragraph was not
``case_11_100``     the right case ranks 11 to 100: a deeper list or a reranker
                    could recover it
``case_beyond_100`` ranked below 100
``not_retrieved``   absent from the 1,000 chunks kept

Breakdowns of Recall@10 by properties of the query and of the relevant case, to see
where retrieval is weak: court of the relevant case, length of the relevant opinion
(long-document retrieval), how often the passage is quoted in the benchmark's
source, query length, and lexical overlap between the query and the quoted passage.
"""

from __future__ import annotations

from collections import defaultdict

from legalrag.eval.stats import bootstrap_ci
from legalrag.experiments.common import Bench
from legalrag.index.bm25 import tokenize
from legalrag.provenance import run_record, write_result
from legalrag.retrieval.core import case_ranking, chunk_hits


def _bucket(value: float, edges: list[float], labels: list[str]) -> str:
    for edge, label in zip(edges, labels, strict=False):
        if value < edge:
            return label
    return labels[-1]


def _table(groups: dict[str, list[float]], order: list[str] | None = None) -> dict:
    out = {}
    for key in order or sorted(groups):
        values = groups.get(key, [])
        if not values:
            continue
        mean, lo, hi = bootstrap_ci(values)
        out[key] = {
            "n": len(values),
            "recall@10": round(mean, 4),
            "ci_low": round(lo, 4),
            "ci_high": round(hi, 4),
        }
    return out


def run(retriever: str = "bm25", limit: int | None = None, k1: float | None = None,
        b: float | None = None):  # fmt: skip
    bench = Bench.load()
    if k1 is not None and b is not None:
        bench.bm25.set_params(float(k1), float(b))
    queries = bench.queries("test", limit)
    run_ = bench.first_stage(retriever, [q.text for q in queries], bench.masks(queries))

    chunks_per_case: dict[str, int] = defaultdict(int)
    for chunk in bench.store.chunks:
        chunks_per_case[chunk.case_id] += 1

    outcomes: dict[str, int] = defaultdict(int)
    by: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for q, ranking in zip(queries, run_.rankings, strict=True):
        gold = q.gold_cases[0]
        cases = case_ranking(ranking, bench.store)
        rank = cases.index(gold) + 1 if gold in cases else None
        passage = "hit" in chunk_hits(ranking.head(10), bench.store, set(q.gold_paras))
        if passage:
            outcomes["passage_top10"] += 1
        elif rank and rank <= 10:
            outcomes["case_top10"] += 1
        elif rank and rank <= 100:
            outcomes["case_11_100"] += 1
        elif rank:
            outcomes["case_beyond_100"] += 1
        else:
            outcomes["not_retrieved"] += 1

        hit = float(bool(rank and rank <= 10))
        case = bench.store.cases[gold]
        by["gold_court_level"][case.court_level].append(hit)
        by["citing_court_level"][q.court_level].append(hit)
        by["gold_opinion_words"][
            _bucket(case.word_count, [3000, 7000, 15000], ["<3k", "3k-7k", "7k-15k", ">=15k"])
        ].append(hit)
        by["gold_opinion_chunks"][
            _bucket(chunks_per_case[gold], [20, 50, 100], ["<20", "20-49", "50-99", ">=100"])
        ].append(hit)
        by["query_words"][
            _bucket(len(q.text.split()), [50, 90, 120], ["<50", "50-89", "90-119", "120"])
        ].append(hit)
        query_terms, quote_terms = set(tokenize(q.text)), set(tokenize(q.gold_quotes[0]))
        overlap = len(query_terms & quote_terms) / max(1, len(quote_terms))
        by["query_quote_term_overlap"][
            _bucket(overlap, [0.1, 0.3, 0.5], ["<0.1", "0.1-0.3", "0.3-0.5", ">=0.5"])
        ].append(hit)
        gap = int(q.date[:4]) - case.year
        by["years_between_case_and_query"][
            _bucket(gap, [5, 15, 30], ["<5", "5-14", "15-29", ">=30"])
        ].append(hit)

    n = len(queries)
    order = {
        "gold_opinion_words": ["<3k", "3k-7k", "7k-15k", ">=15k"],
        "gold_opinion_chunks": ["<20", "20-49", "50-99", ">=100"],
        "query_words": ["<50", "50-89", "90-119", "120"],
        "query_quote_term_overlap": ["<0.1", "0.1-0.3", "0.3-0.5", ">=0.5"],
        "years_between_case_and_query": ["<5", "5-14", "15-29", ">=30"],
    }
    result = run_record(
        {
            "experiment": "error_analysis",
            "retriever": run_.name,
            "n": n,
            "bm25": {"k1": bench.bm25.k1, "b": bench.bm25.b},
        },
        bench.store.corpus_version,
        outcomes={
            k: {"n": outcomes[k], "share": round(outcomes[k] / n, 4)}
            for k in (
                "passage_top10",
                "case_top10",
                "case_11_100",
                "case_beyond_100",
                "not_retrieved",
            )
        },
        recall_at_10_by={name: _table(groups, order.get(name)) for name, groups in by.items()},
    )
    write_result(bench.paths.results / "analysis" / f"error_analysis_{retriever}.json", result)
    return result
