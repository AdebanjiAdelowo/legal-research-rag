"""RQ6 (jurisdiction): binding authority, the wrong circuit, and questions the
corpus cannot answer.

Part 1 describes the benchmark: for what share of test queries is the case the
court actually quoted binding on that court, and for what share only persuasive?

Part 2 restricts retrieval to authority binding on the query's forum (Supreme Court
plus that circuit's court of appeals) and measures the trade: a smaller, more
authoritative pool against the loss of every query whose answer was persuasive
authority from elsewhere.

Part 3 is a mismatch: retrieval is restricted to the court of appeals of a
*different* circuit. The relevant case is then out of reach for almost every query,
which is what happens when a filter is set wrongly or a question is asked of the
wrong body of law. The question is whether the retrieval score reveals it.

Part 4 tests the rule-based scope detector on hand-written questions about state
law, foreign law and non-legal topics, against in-scope federal questions.
"""

from __future__ import annotations

from collections import Counter

from legalrag.data.courts import CIRCUITS, Court, authority_label
from legalrag.eval.stats import auroc, wilson
from legalrag.experiments import fixtures
from legalrag.experiments.common import Bench, aggregate, compare, mean
from legalrag.provenance import run_record, write_result
from legalrag.retrieval.query import interpret

NUMBERED = [c for c in CIRCUITS if c not in ("DC", "Fed")]


def other_circuit(circuit: str) -> str:
    """A fixed different circuit: the next numbered one, wrapping around."""
    if circuit not in NUMBERED:
        return "1"
    return NUMBERED[(NUMBERED.index(circuit) + 1) % len(NUMBERED)]


def _label(bench: Bench, q) -> str:
    gold = bench.store.cases[q.gold_cases[0]]
    forum = Court(q.court, q.court_level, q.circuit)
    return authority_label(Court(gold.court, gold.court_level, gold.circuit), forum)


def _scope_detector() -> dict:
    out = {}
    for group, questions in fixtures.OUT_OF_SCOPE.items():
        hits = sum(bool(interpret(q).out_of_scope) for q in questions)
        p, lo, hi = wilson(hits, len(questions))
        out[group] = {
            "n": len(questions),
            "flagged": hits,
            "recall": round(p, 3),
            "ci_low": round(lo, 3),
            "ci_high": round(hi, 3),
            "missed": [q for q in questions if not interpret(q).out_of_scope],
        }
    false_alarms = [q for q in fixtures.IN_SCOPE if interpret(q).out_of_scope]
    p, lo, hi = wilson(len(false_alarms), len(fixtures.IN_SCOPE))
    out["in_scope"] = {
        "n": len(fixtures.IN_SCOPE),
        "false_alarms": len(false_alarms),
        "false_alarm_rate": round(p, 3),
        "ci_low": round(lo, 3),
        "ci_high": round(hi, 3),
        "examples": false_alarms,
    }
    return out


def run(limit: int | None = None, k1: float | None = None, b: float | None = None):
    bench = Bench.load()
    if k1 is not None and b is not None:
        bench.bm25.set_params(float(k1), float(b))
    queries = [q for q in bench.queries("test", limit) if q.circuit and q.court_level != "scotus"]
    texts = [q.text for q in queries]
    labels = [_label(bench, q) for q in queries]
    gold_level = Counter(bench.store.cases[q.gold_cases[0]].court_level for q in queries)

    base = bench.first_stage("bm25", texts, bench.masks(queries), "as_of")
    binding = bench.first_stage(
        "bm25",
        texts,
        [bench.store.mask(before=q.date, binding_on_circuit=q.circuit) for q in queries],
        "as_of+binding",
    )
    wrong = bench.first_stage(
        "bm25",
        texts,
        [
            bench.store.mask(
                before=q.date, circuits=(other_circuit(q.circuit),), levels=("appeals",)
            )
            for q in queries
        ],
        "as_of+wrong-circuit",
    )
    pq_base, pq_bind, pq_wrong = (bench.evaluate(r, queries) for r in (base, binding, wrong))

    is_binding = [label == "binding" for label in labels]

    def subset(pq, keep):
        return {m: [v for v, k in zip(vals, keep, strict=True) if k] for m, vals in pq.items()}

    top1 = lambda run: [float(r.scores[0]) if len(r) else float("-inf") for r in run.rankings]  # noqa: E731
    matched, mismatched = top1(base), top1(wrong)
    result = run_record(
        {
            "experiment": "jurisdiction",
            "retriever": "bm25",
            "bm25": {"k1": bench.bm25.k1, "b": bench.bm25.b},
            "n": len(queries),
        },
        bench.store.corpus_version,
        gold_authority={
            "by_label": dict(Counter(labels)),
            "binding_share": round(sum(is_binding) / max(1, len(labels)), 4),
            "gold_court_level": dict(gold_level),
        },
        no_court_filter=aggregate(pq_base),
        binding_only_filter={
            "all_queries": aggregate(pq_bind),
            "difference_vs_no_filter": compare(pq_bind, pq_base, "recall@10"),
            "queries_whose_gold_is_binding": {
                "n": sum(is_binding),
                "binding_filter": aggregate(subset(pq_bind, is_binding)),
                "no_filter": aggregate(subset(pq_base, is_binding)),
                "difference": compare(
                    subset(pq_bind, is_binding), subset(pq_base, is_binding), "recall@10"
                ),
            },
            "queries_whose_gold_is_persuasive": {
                "n": len(labels) - sum(is_binding),
                "no_filter_recall@10": round(
                    mean(subset(pq_base, [not x for x in is_binding])["recall@10"]), 4
                ),
                "binding_filter_recall@10": round(
                    mean(subset(pq_bind, [not x for x in is_binding])["recall@10"]), 4
                ),
            },
        },
        wrong_circuit={
            "recall@10": round(mean(pq_wrong["recall@10"]), 4),
            "gold_still_reachable": round(mean(pq_wrong["recall@100"]), 4),
            "top1_score_matched_mean": round(mean(matched), 3),
            "top1_score_mismatched_mean": round(mean([s for s in mismatched if s > -1e9]), 3),
            "auroc_top1_score_matched_vs_mismatched": round(
                auroc(matched + mismatched, [1] * len(matched) + [0] * len(mismatched)), 4
            ),
        },
        scope_detector=_scope_detector(),
    )
    write_result(bench.paths.results / "robustness" / "jurisdiction_bm25.json", result)
    return result
