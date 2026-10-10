"""RQ6 (time): future-case leakage and outdated authority.

Part 1, leakage. A query dated D could only have been answered with cases decided
before D. The ``history`` split holds queries from 1975 to 1999, when most of the
corpus did not yet exist. Each query is run with and without the as-of filter. The
unfiltered run shows how much of a result list would be cases from the query's
future, and how an evaluation that ignores dates would misstate quality: later
cases often restate the same rule, so they outrank the true source while being
counted as errors, or, in a benchmark that pooled relevance, as hits that no one
could have found at the time.

Part 2, outdated authority. Relevance is not authority. For each test query the
top results are checked against the short table of known Supreme Court overrulings
(``data/treatment.py``). Two counts are reported: cases already overruled when the
query arose, which a careful system should have flagged, and cases overruled only
later, which nothing in the corpus could reveal. The table is a small illustrative
list, so both counts are lower bounds.
"""

from __future__ import annotations

from legalrag.data.treatment import known_treatment
from legalrag.eval.stats import bootstrap_ci
from legalrag.experiments.common import Bench, adjust, aggregate, compare
from legalrag.index.store import _date_int
from legalrag.provenance import run_record, write_result
from legalrag.retrieval.core import case_ranking


def _future_stats(bench: Bench, run, queries, k: int = 10) -> dict:
    share, top1 = [], []
    for q, ranking in zip(queries, run.rankings, strict=True):
        cases = case_ranking(ranking, bench.store)[:k]
        future = [bench.store.cases[c].decision_date >= q.date for c in cases]
        share.append(sum(future) / max(1, len(future)))
        top1.append(float(future[0]) if future else 0.0)
    m, lo, hi = bootstrap_ci(share)
    t, tlo, thi = bootstrap_ci(top1)
    return {
        f"future_share@{k}": {"mean": round(m, 4), "ci_low": round(lo, 4), "ci_high": round(hi, 4)},
        "top1_is_future": {"mean": round(t, 4), "ci_low": round(tlo, 4), "ci_high": round(thi, 4)},
    }


def _eligible_share(bench: Bench, queries) -> float:
    dates = bench.store.case_date
    return float(sum((dates < _date_int(q.date)).mean() for q in queries) / max(1, len(queries)))


def _leakage(bench: Bench, split: str, limit: int | None) -> dict:
    queries = bench.queries(split, limit)
    texts = [q.text for q in queries]
    filtered = bench.first_stage("bm25", texts, bench.masks(queries, as_of=True), "as_of")
    unfiltered = bench.first_stage("bm25", texts, bench.masks(queries, as_of=False), "none")
    pq_f, pq_u = bench.evaluate(filtered, queries), bench.evaluate(unfiltered, queries)
    return {
        "n": len(queries),
        "date_range": [min(q.date for q in queries), max(q.date for q in queries)],
        "corpus_share_eligible_mean": round(_eligible_share(bench, queries), 4),
        "as_of_filter": aggregate(pq_f),
        "no_filter": {**aggregate(pq_u), **_future_stats(bench, unfiltered, queries)},
        "comparisons": adjust(
            {
                f"as-of vs no filter [{m}]": compare(pq_f, pq_u, m)
                for m in ("recall@10", "ndcg@10", "passage_hit@10")
            }
        ),
    }


def _outdated(bench: Bench, limit: int | None, k: int = 10) -> dict:
    queries = bench.queries("test", limit)
    run = bench.first_stage("bm25", [q.text for q in queries], bench.masks(queries), "as_of")
    already, later, gold_already, gold_later = [], [], 0, 0
    examples: dict[str, int] = {}
    for q, ranking in zip(queries, run.rankings, strict=True):
        year = int(q.date[:4])
        a = l_ = 0
        for case_id in case_ranking(ranking, bench.store)[:k]:
            case = bench.store.cases[case_id]
            if known_treatment(case.citations, year):
                a += 1
                examples[case.bluebook()] = examples.get(case.bluebook(), 0) + 1
            elif known_treatment(case.citations):
                l_ += 1
        already.append(float(a > 0))
        later.append(float(l_ > 0))
        gold = bench.store.cases[q.gold_cases[0]]
        gold_already += bool(known_treatment(gold.citations, year))
        gold_later += bool(known_treatment(gold.citations)) and not known_treatment(
            gold.citations, year
        )
    a_mean, a_lo, a_hi = bootstrap_ci(already)
    l_mean, l_lo, l_hi = bootstrap_ci(later)
    flagged_in_corpus = sum(bool(known_treatment(c.citations)) for c in bench.store.cases.values())
    return {
        "n": len(queries),
        "k": k,
        "treatment_table_cases_in_corpus": flagged_in_corpus,
        "queries_with_case_already_overruled_in_top_k": {
            "mean": round(a_mean, 4),
            "ci_low": round(a_lo, 4),
            "ci_high": round(a_hi, 4),
        },
        "queries_with_case_overruled_later_in_top_k": {
            "mean": round(l_mean, 4),
            "ci_low": round(l_lo, 4),
            "ci_high": round(l_hi, 4),
        },
        "gold_case_already_overruled_when_quoted": gold_already,
        "gold_case_overruled_after_query_date": gold_later,
        "most_frequent_already_overruled": sorted(examples.items(), key=lambda x: -x[1])[:5],
    }


def run(limit: int | None = None, k1: float | None = None, b: float | None = None):
    bench = Bench.load()
    if k1 is not None and b is not None:
        bench.bm25.set_params(float(k1), float(b))
    result = run_record(
        {
            "experiment": "temporal",
            "retriever": "bm25",
            "bm25": {"k1": bench.bm25.k1, "b": bench.bm25.b},
        },
        bench.store.corpus_version,
        leakage={split: _leakage(bench, split, limit) for split in ("history", "test")},
        outdated_authority=_outdated(bench, limit),
    )
    write_result(bench.paths.results / "robustness" / "temporal_bm25.json", result)
    return result
