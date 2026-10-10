"""RQ4: grounded summaries with verifiable citations, and what verification changes.

For each test query the pipeline retrieves, selects evidence and generates typed
statements. Two evidence conditions are run:

``normal``    as-of retrieval
``withheld``  the case the court actually quoted is removed from the pool, so the
              evidence is at best adjacent. A grounded generator should say less or
              abstain; an ungrounded one will answer anyway.

Measures (per answer, then averaged over answers; ratios are pooled):

* ``abstained``
* ``citation_precision``: evidence citations whose passage supports the statement
* ``citation_recall``: citable statements with at least one supporting citation
* ``unsupported_claim_rate``: citable statements with none
* ``hallucinated_citation_rate``: answers containing a marker that points at
  nothing, or a reporter citation that is not in the corpus or not in the evidence
* answer correctness, by proxy. There are no reference answers, so correctness is
  measured against what the court did: ``gold_case_cited`` (a statement cites the
  quoted case), ``gold_passage_cited`` (it cites the chunk holding the quoted
  paragraph) and ``gold_quote_coverage`` (share of the quotation's words that
  appear in the cited statements). These say whether the summary surfaces the
  authority the court relied on, not whether it is a correct statement of law.

Support is decided by an automatic judge (``eval_judge``). When ``filter_judge`` is
given, statements it does not confirm are removed (the strict policy) and the
measures are recomputed on what remains. The filter and the evaluation should use
different judges; with the same judge the improvement is true by construction.
"""

from __future__ import annotations

import time

import numpy as np

from legalrag.data.benchmark import tokens
from legalrag.eval.stats import bootstrap_ci, wilson
from legalrag.experiments.common import Bench
from legalrag.provenance import run_record, write_jsonl, write_result
from legalrag.rag import types as T
from legalrag.rag.evidence import select_evidence
from legalrag.rag.generate import make_generator
from legalrag.retrieval.rerank import Reranker, apply_scores, head_texts
from legalrag.verify.judges import SUPPORTED, make_judge
from legalrag.verify.verifier import CitationIndex, CitationVerifier, summarise


def _score(statements, evidence, checks, query, abstained: bool) -> dict:
    by_label = {e.label: e for e in evidence}
    gold_cases, gold_paras = set(query.gold_cases), set(query.gold_paras)
    summary = summarise(checks)
    cited = [by_label[c] for s in statements for c in s.citations if c in by_label]
    citable_text = " ".join(s.text for s in statements if s.type in T.CITABLE)
    quote = set(tokens(query.gold_quotes[0])) if query.gold_quotes else set()
    supported_gold = any(
        s.type in T.CITABLE
        and any(
            c.kind == "marker"
            and c.support == SUPPORTED
            and c.raw.strip("[]") in by_label
            and gold_paras & set(by_label[c.raw.strip("[]")].para_ids)
            for c in check.citations
        )
        for s, check in zip(statements, checks, strict=True)
    )
    return {
        "abstained": float(abstained),
        "statements": summary["statements"],
        "citable": summary["citable_statements"],
        "evidence_citations": summary["evidence_citations"],
        "supported_citations": summary["supported_citations"],
        "supported_statements": round(
            (summary["citation_recall"] or 0.0) * summary["citable_statements"]
        ),
        "hallucinated": float(summary["hallucinated_citations"] > 0),
        "hallucinated_count": summary["hallucinated_citations"],
        "gold_in_evidence": float(any(e.case_id in gold_cases for e in evidence)),
        "gold_passage_in_evidence": float(any(gold_paras & set(e.para_ids) for e in evidence)),
        "gold_case_cited": float(any(e.case_id in gold_cases for e in cited)),
        "gold_passage_cited": float(any(gold_paras & set(e.para_ids) for e in cited)),
        "gold_passage_cited_and_supported": float(supported_gold),
        "gold_quote_coverage": len(quote & set(tokens(citable_text))) / len(quote)
        if quote
        else 0.0,
    }


def _ratio(rows: list[dict], num: str, den: str, seed: int = 0) -> dict:
    """Pooled ratio with a bootstrap interval over answers."""
    n = np.array([r[num] for r in rows], dtype=float)
    d = np.array([r[den] for r in rows], dtype=float)
    if d.sum() == 0:
        return {"value": None, "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(rows), size=(2000, len(rows)))
    dens = d[idx].sum(axis=1)
    boots = n[idx].sum(axis=1)[dens > 0] / dens[dens > 0]
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {
        "value": round(float(n.sum() / d.sum()), 4),
        "ci_low": round(float(lo), 4),
        "ci_high": round(float(hi), 4),
        "n": int(d.sum()),
    }


def _mean(rows: list[dict], key: str) -> dict:
    m, lo, hi = bootstrap_ci([r[key] for r in rows])
    return {"value": round(m, 4), "ci_low": round(lo, 4), "ci_high": round(hi, 4), "n": len(rows)}


def _rate(rows: list[dict], key: str) -> dict:
    p, lo, hi = wilson(int(sum(r[key] for r in rows)), len(rows))
    return {"value": round(p, 4), "ci_low": round(lo, 4), "ci_high": round(hi, 4), "n": len(rows)}


def _aggregate(rows: list[dict]) -> dict:
    answered = [r for r in rows if not r["abstained"]]
    out = {
        "answers": len(rows),
        "abstained": _rate(rows, "abstained"),
        "gold_in_evidence": _rate(rows, "gold_in_evidence"),
        "gold_passage_in_evidence": _rate(rows, "gold_passage_in_evidence"),
    }
    if not answered:
        return out
    supported = [
        {**r, "unsupported": r["citable"] - r["supported_statements"]} for r in answered
    ]
    out.update(
        {
            "statements_per_answer": _mean(answered, "statements"),
            "citation_precision": _ratio(answered, "supported_citations", "evidence_citations"),
            "citation_recall": _ratio(answered, "supported_statements", "citable"),
            "unsupported_claim_rate": _ratio(supported, "unsupported", "citable"),
            "hallucinated_citation_rate": _rate(answered, "hallucinated"),
            "gold_case_cited": _rate(answered, "gold_case_cited"),
            "gold_passage_cited": _rate(answered, "gold_passage_cited"),
            "gold_passage_cited_and_supported": _rate(answered, "gold_passage_cited_and_supported"),
            "gold_quote_coverage": _mean(answered, "gold_quote_coverage"),
        }
    )
    return out


def run(
    generator: str = "extractive",
    limit: int = 300,
    eval_judge: str = "lexical",
    filter_judge: str | None = None,
    reranker: str | None = None,
    top_n: int = 30,
    evidence_k: int = 5,
    k1: float | None = None,
    b: float | None = None,
    tag: str | None = None,
):
    bench = Bench.load()
    if k1 is not None and b is not None:
        bench.bm25.set_params(float(k1), float(b))
    queries = bench.queries("test", int(limit))
    texts = [q.text for q in queries]
    gen = make_generator(generator)
    index = CitationIndex(bench.store.cases)
    evaluator = CitationVerifier(index, make_judge(eval_judge))
    filterer = CitationVerifier(index, make_judge(filter_judge)) if filter_judge else None
    model = Reranker(reranker) if reranker else None

    conditions = {
        "normal": bench.masks(queries),
        "withheld": [
            bench.store.mask(before=q.date, exclude_cases=tuple(q.gold_cases)) for q in queries
        ],
    }
    results, dump, latency = {}, [], []
    for name, masks in conditions.items():
        run_ = bench.first_stage("bm25", texts, masks, name if name != "normal" else "as_of",
                                 depth=200)  # fmt: skip
        rows, rows_filtered = [], []
        for q, ranking in zip(queries, run_.rankings, strict=True):
            if model is not None:
                ranking = apply_scores(
                    ranking, model.score(q.text, head_texts(ranking, bench.store, int(top_n)))
                )
            evidence = select_evidence(ranking, bench.store, k=int(evidence_k), per_case=2)
            start = time.perf_counter()
            abstained, statements, raw = gen.generate(q.text, evidence)
            latency.append(time.perf_counter() - start)
            checks = evaluator.check_statements(statements, evidence)
            rows.append(_score(statements, evidence, checks, q, abstained))
            if filterer is not None:
                keep = [
                    s
                    for s, c in zip(
                        statements, filterer.check_statements(statements, evidence), strict=True
                    )
                    if not (s.type in T.CITABLE and c.status != "verified")
                    and not (s.type not in T.CITABLE and c.status == "unsupported")
                ]
                nothing = not any(s.type in T.CITABLE for s in keep)
                rows_filtered.append(
                    _score(keep, evidence, evaluator.check_statements(keep, evidence), q,
                           abstained or nothing)  # fmt: skip
                )
            dump.append(
                {
                    "query_id": q.query_id,
                    "condition": name,
                    "abstained": abstained,
                    "raw_output": raw,
                    "statements": [
                        {"type": s.type, "text": s.text, "citations": s.citations,
                         "status": c.status}
                        for s, c in zip(statements, checks, strict=True)
                    ],  # fmt: skip
                    "evidence": [e.chunk_id for e in evidence],
                }
            )
        results[name] = {"verification_off": _aggregate(rows)}
        if filterer is not None:
            results[name]["verification_strict"] = _aggregate(rows_filtered)

    name = tag or generator.replace("/", "_").replace(":", "-")
    result = run_record(
        {
            "experiment": "rag",
            "generator": gen.name,
            "retriever": "bm25" + (f"+{reranker}" if reranker else ""),
            "bm25": {"k1": bench.bm25.k1, "b": bench.bm25.b},
            "evidence_k": int(evidence_k),
            "eval_judge": evaluator.judge.name,
            "filter_judge": filterer.judge.name if filterer else None,
            "n_queries": len(queries),
        },
        bench.store.corpus_version,
        conditions=results,
        generation_seconds={
            "mean": round(float(np.mean(latency)), 4),
            "p50": round(float(np.median(latency)), 4),
            "p95": round(float(np.quantile(latency, 0.95)), 4),
        },
    )
    out = bench.paths.results / "rag"
    write_result(out / f"rag_{name}.json", result)
    write_jsonl(out / f"answers_{name}.jsonl", dump)
    return result
