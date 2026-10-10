"""RQ7: can the system tell when the retrieved evidence is insufficient?

The sufficiency signal is the score of the best retrieved passage (from BM25, or
from the cross-encoder when a reranker is given). A threshold is tuned on ``dev``
and then fixed. On ``test`` the same query is run under conditions in which the
right answer is or is not reachable:

``answerable``       normal as-of retrieval; the quoted case is in the pool
``gold_withheld``    the quoted case is removed from the pool
``predates_gold``    the as-of date is moved to the day the quoted case was decided,
                     so that case and everything later is gone
``wrong_circuit``    only the court of appeals of another circuit is searched
``out_of_scope``     hand-written state-law, foreign-law and non-legal questions

"Unanswerable" is strict only for the last condition. When the quoted case is
withheld, other opinions that state the same rule usually remain, and a researcher
might be content with them; what is lost is the specific authority the court relied
on. A high score in those conditions is therefore partly correct behaviour and
partly the failure this experiment is looking for, and the two cannot be separated
without expert labels. The numbers are reported with that caveat.

Also reported: whether the signal predicts retrieval success (the quoted case
appearing in the top five), which is the selective-prediction view of the same gate.
"""

from __future__ import annotations

from legalrag.eval.stats import auroc
from legalrag.experiments import fixtures
from legalrag.experiments.common import Bench, Run
from legalrag.experiments.jurisdiction import other_circuit
from legalrag.provenance import run_record, write_result
from legalrag.rag.abstain import evidence_signal, gate_metrics, tune_threshold
from legalrag.retrieval.core import case_ranking
from legalrag.retrieval.query import interpret
from legalrag.retrieval.rerank import Reranker, apply_scores, head_texts

SIGNALS = ("top1", "mean3", "margin")


def _conditions(bench: Bench, queries) -> dict[str, list]:
    store = bench.store
    return {
        "answerable": [store.mask(before=q.date) for q in queries],
        "gold_withheld": [
            store.mask(before=q.date, exclude_cases=tuple(q.gold_cases)) for q in queries
        ],
        "predates_gold": [
            store.mask(before=min(store.cases[g].decision_date for g in q.gold_cases))
            for q in queries
        ],
        "wrong_circuit": [
            store.mask(
                before=q.date, circuits=(other_circuit(q.circuit or "1"),), levels=("appeals",)
            )
            for q in queries
        ],
    }


def _rerank(run: Run, texts: list[str], bench: Bench, reranker: Reranker | None, top_n: int) -> Run:
    if reranker is None:
        return run
    out = [
        apply_scores(r, reranker.score(t, head_texts(r, bench.store, top_n)))
        for r, t in zip(run.rankings, texts, strict=True)
    ]
    return Run(run.name + "+rerank", out)


def _signals(bench: Bench, queries, reranker, top_n: int) -> dict[str, dict[str, list[float]]]:
    texts = [q.text for q in queries]
    out: dict[str, dict[str, list[float]]] = {}
    for name, masks in _conditions(bench, queries).items():
        run = _rerank(bench.first_stage("bm25", texts, masks, name, depth=200), texts, bench,
                      reranker, top_n)  # fmt: skip
        out[name] = {s: [evidence_signal(r, s) for r in run.rankings] for s in SIGNALS}
        if name == "answerable":
            out[name]["gold_in_top5"] = [
                float(bool(set(case_ranking(r, bench.store)[:5]) & set(q.gold_cases)))
                for r, q in zip(run.rankings, queries, strict=True)
            ]
    return out


def _out_of_scope(bench: Bench, reranker, top_n: int) -> dict[str, list[float]]:
    questions = [q for group in fixtures.OUT_OF_SCOPE.values() for q in group]
    run = bench.first_stage("bm25", questions, None, "none", depth=200)
    run = _rerank(run, questions, bench, reranker, top_n)
    return {s: [evidence_signal(r, s) for r in run.rankings] for s in SIGNALS}


def run(limit: int | None = 600, reranker: str | None = None, top_n: int = 30,
        k1: float | None = None, b: float | None = None):  # fmt: skip
    bench = Bench.load()
    if k1 is not None and b is not None:
        bench.bm25.set_params(float(k1), float(b))
    model = Reranker(reranker) if reranker else None
    dev, test = bench.queries("dev", limit), bench.queries("test", limit)
    dev_sig, test_sig = _signals(bench, dev, model, top_n), _signals(bench, test, model, top_n)
    oos = _out_of_scope(bench, model, top_n)

    finite = lambda xs: [x if x > -1e30 else -1e30 for x in xs]  # noqa: E731
    by_signal = {}
    for signal in SIGNALS:
        # One threshold per signal, tuned on dev against the withheld condition, the
        # hardest one that can be constructed without extra data.
        tuned = tune_threshold(
            finite(dev_sig["answerable"][signal]), finite(dev_sig["gold_withheld"][signal])
        )
        conditions = {}
        for name in ("gold_withheld", "predates_gold", "wrong_circuit"):
            conditions[name] = gate_metrics(
                finite(test_sig["answerable"][signal]), finite(test_sig[name][signal]),
                tuned["threshold"],
            )  # fmt: skip
        conditions["out_of_scope"] = gate_metrics(
            finite(test_sig["answerable"][signal]), finite(oos[signal]), tuned["threshold"]
        )
        by_signal[signal] = {
            "dev_tuning": tuned,
            "test": conditions,
            "predicts_gold_in_top5_auroc": round(
                auroc(
                    finite(test_sig["answerable"][signal]),
                    [int(x) for x in test_sig["answerable"]["gold_in_top5"]],
                ),
                4,
            ),
        }

    # The scope rule runs before retrieval; combine it with the score gate.
    flagged = [bool(interpret(q).out_of_scope) for g in fixtures.OUT_OF_SCOPE.values() for q in g]
    threshold = by_signal["top1"]["dev_tuning"]["threshold"]
    by_score = [s < threshold for s in finite(oos["top1"])]
    n = len(flagged)
    result = run_record(
        {
            "experiment": "abstention",
            "retriever": "bm25",
            "bm25": {"k1": bench.bm25.k1, "b": bench.bm25.b},
            "reranker": reranker,
            "rerank_top_n": top_n if reranker else None,
            "n_dev": len(dev),
            "n_test": len(test),
            "n_out_of_scope": n,
        },
        bench.store.corpus_version,
        signals=by_signal,
        gold_in_top5_rate=round(sum(test_sig["answerable"]["gold_in_top5"]) / max(1, len(test)), 4),
        out_of_scope_combined={
            "abstained_by_rule": round(sum(flagged) / n, 4),
            "abstained_by_score": round(sum(by_score) / n, 4),
            "abstained_by_either": round(
                sum(f or s for f, s in zip(flagged, by_score, strict=True)) / n, 4
            ),
        },
    )
    tag = f"bm25+{reranker}" if reranker else "bm25"
    write_result(bench.paths.results / "abstention" / f"abstention_{tag}.json", result)
    return result
