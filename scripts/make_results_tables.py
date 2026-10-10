"""Render the result tables in README.md from the JSON files under results/.

    python scripts/make_results_tables.py            # rewrite the README section
    python scripts/make_results_tables.py --check    # fail if the README is stale

Every number in the README's Results section comes from here, so a table cannot
drift from the run that produced it. An experiment with no result file is listed as
PENDING with the command that produces it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
START, END = "<!-- results:start -->", "<!-- results:end -->"


def load(path: str) -> dict | None:
    file = RESULTS / path
    return json.loads(file.read_text()) if file.exists() else None


def ci(cell: dict | None, digits: int = 3) -> str:
    if not cell or cell.get("mean", cell.get("value")) is None:
        return "n/a"
    value = cell.get("mean", cell.get("value"))
    return f"{value:.{digits}f} [{cell['ci_low']:.{digits}f}, {cell['ci_high']:.{digits}f}]"


def num(value, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def diff(c: dict) -> str:
    p = c.get("p_holm", c["p"])
    p_text = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
    return f"{c['diff']:+.3f} [{c['ci_low']:+.3f}, {c['ci_high']:+.3f}], {p_text}"


def table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def pending(what: str, command: str, reason: str) -> str:
    return f"**PENDING: {what}.** {reason}\n\n```bash\n{command}\n```"


NAMES = {
    "bm25": "BM25",
    "dense-minilm": "Dense, all-MiniLM-L6-v2 (general)",
    "dense-bge-small": "Dense, bge-small-en-v1.5 (general)",
    "dense-modernbert-general": "Dense, modernbert-embed-base (general)",
    "dense-modernbert-legal": "Dense, modernbert-embed-base legal fine-tune",
}


def system_name(key: str) -> str:
    if key in NAMES:
        return NAMES[key]
    for kind, label in (
        ("hybrid-rrf-", "Hybrid, rank fusion"),
        ("hybrid-score-", "Hybrid, score fusion"),
    ):
        if key.startswith(kind):
            return f"{label}: BM25 + {key.removeprefix(kind)}"
    return key


def retrieval() -> str:
    r = load("retrieval/retrieval_section220.json")
    if not r:
        return pending("retrieval comparison", "legalrag experiment retrieval", "Not run.")
    cfg, out = r["config"], []
    out.append(
        f"{cfg['n_test']:,} test queries over {cfg['chunks']:,} passages; settings tuned on "
        f"{cfg['n_dev']} development queries. Case-level metrics unless marked. Brackets "
        "are 95% bootstrap intervals."
    )
    rows = []
    for key, s in r["systems"].items():
        rows.append([
            system_name(key), ci(s["recall@1"]), ci(s["recall@10"]), ci(s["recall@100"]),
            ci(s["mrr@10"]), ci(s["ndcg@10"]), ci(s["precision@10"]), ci(s["passage_hit@10"]),
        ])  # fmt: skip
    out.append(table(
        ["System", "Recall@1", "Recall@10", "Recall@100", "MRR@10", "nDCG@10", "P@10",
         "Passage hit@10"], rows))  # fmt: skip
    t = r["tuning"]["bm25"]
    out.append(f"Tuned BM25: k1 = {t['k1']}, b = {t['b']}.")
    if r["comparisons"]:
        rows = [[name, diff(c)] for name, c in r["comparisons"].items()]
        out.append("Paired differences in nDCG@10 unless marked (Holm-adjusted):")
        out.append(table(["Comparison", "Difference"], rows))
    missing = [e for e in ("modernbert-general", "modernbert-legal")
               if f"dense-{e}" not in r["systems"]]  # fmt: skip
    if missing:
        out.append(pending(
            "legal against general encoder (RQ2)",
            "legalrag index modernbert-general modernbert-legal\nlegalrag experiment retrieval",
            "Needs two 596 MB model downloads and several hours of laptop encoding, or the "
            "Kaggle runner (`kaggle/README.md`).",
        ))  # fmt: skip
    leak = r["leakage_control"]
    rows = [[system_name(k), num(v["masked"]), num(v["unmasked"]), diff(v["difference"])]
            for k, v in leak.items()]  # fmt: skip
    out.append("**Leakage control.** The same queries with citations and case names left in:")
    out.append(
        table(["System", "nDCG@10 masked", "nDCG@10 unmasked", "Unmasked minus masked"], rows)
    )
    return "\n\n".join(out)


def rerank() -> str:
    files = (
        sorted((RESULTS / "rerank").glob("rerank_*.json")) if (RESULTS / "rerank").exists() else []
    )
    if not files:
        return pending("cross-encoder reranking (RQ3)", "legalrag experiment rerank k1=1.6 b=0.75",
                       "Not run.")  # fmt: skip
    out = []
    for f in files:
        r = json.loads(f.read_text())
        cfg = r["config"]
        rows = [
            ["First stage: " + system_name(cfg["first_stage"]), ci(r["first_stage"]["recall@10"]),
             ci(r["first_stage"]["mrr@10"]), ci(r["first_stage"]["ndcg@10"]),
             ci(r["first_stage"]["passage_hit@10"]), "n/a"],
            [f"+ {cfg['reranker']} on top {r['top_n']}", ci(r["reranked"]["recall@10"]),
             ci(r["reranked"]["mrr@10"]), ci(r["reranked"]["ndcg@10"]),
             ci(r["reranked"]["passage_hit@10"]),
             f"{r['reranked']['rerank_seconds_per_query'] * 1000:.0f} ms"],
        ]  # fmt: skip
        out.append(table(["System", "Recall@10", "MRR@10", "nDCG@10", "Passage hit@10",
                          "Added latency per query"], rows))  # fmt: skip
        out.append(
            f"Device: {cfg['device']}. Depth chosen on {cfg['n_dev']} development queries from "
            + ", ".join(f"{k}: nDCG@10 {v['ndcg@10']:.3f}" for k, v in r["dev_grid"].items())
            + f". Share of queries whose case is within the reranked passages: "
            f"{r['case_recall_ceiling_in_reranked_chunks']:.3f}."
        )
        out.append(table(["Comparison", "Difference"],
                         [[n, diff(c)] for n, c in r["comparisons"].items()]))  # fmt: skip
    return "\n\n".join(out)


def chunking() -> str:
    r = load("chunking/chunking_bm25.json")
    if not r:
        return pending("chunking ablation", "legalrag experiment chunking", "Not run.")
    rows = []
    for key, s in r["systems"].items():
        info = r["schemes"][key]
        rows.append([
            key, f"{info['chunks']:,}", f"{s['words@10']['mean']:.0f}", ci(s["ndcg@10"]),
            ci(s["recall@10"]), ci(s["passage_hit@10"]),
            num(s["passage_hits_per_1000_words@10"], 4),
        ])  # fmt: skip
    out = [table(["Scheme", "Passages", "Words in top 10", "nDCG@10", "Recall@10",
                  "Passage hit@10", "Passage hits per 1,000 words read"], rows)]  # fmt: skip
    out.append(table(["Comparison", "Difference"],
                     [[n, diff(c)] for n, c in r["comparisons"].items()]))  # fmt: skip
    return "\n\n".join(out)


def decomposition() -> str:
    r = load("decomposition/decomposition_bm25.json")
    if not r:
        return pending("query decomposition (RQ5)", "legalrag experiment decomposition", "Not run.")
    out = []
    for part, title in (
        ("multi_issue", "Multi-issue queries"),
        ("single_issue", "Single-issue queries"),
    ):
        n = r["config"]["n_multi" if part == "multi_issue" else "n_single"]
        rows = [
            [name, num(s["sub_queries_mean"], 1), ci(s["recall@10"]), ci(s["ndcg@10"]),
             ci(s["all_gold@20"])]
            for name, s in r[part]["systems"].items()
        ]  # fmt: skip
        out.append(f"**{title}** (n = {n}):")
        out.append(table(["Condition", "Sub-queries", "Recall@10", "nDCG@10",
                          "All relevant cases in top 20"], rows))  # fmt: skip
        out.append(table(["Comparison", "Difference"],
                         [[k, diff(c)] for k, c in r[part]["comparisons"].items()]))  # fmt: skip
    return "\n\n".join(out)


def temporal() -> str:
    r = load("robustness/temporal_bm25.json")
    if not r:
        return pending("temporal evaluation", "legalrag experiment temporal", "Not run.")
    rows = []
    for split, v in r["leakage"].items():
        rows.append([
            f"{split} ({v['date_range'][0][:4]} to {v['date_range'][1][:4]}), n = {v['n']}",
            f"{v['corpus_share_eligible_mean']:.0%}",
            num(v["as_of_filter"]["recall@10"]["mean"]), num(v["no_filter"]["recall@10"]["mean"]),
            ci(v["no_filter"]["future_share@10"]), ci(v["no_filter"]["top1_is_future"]),
        ])  # fmt: skip
    out = [table(["Queries", "Corpus decided before the query", "Recall@10 with as-of filter",
                  "Recall@10 without", "Top-10 results from the query's future",
                  "Top result from the future"], rows)]  # fmt: skip
    o = r["outdated_authority"]
    out.append(
        f"**Outdated authority.** Of {o['n']:,} test queries, "
        f"{o['queries_with_case_already_overruled_in_top_k']['mean']:.1%} had in their top "
        f"{o['k']} a case that the treatment table records as overruled or abrogated before "
        f"the query date, and {o['queries_with_case_overruled_later_in_top_k']['mean']:.1%} a "
        f"case overruled only afterwards. The table covers {o['treatment_table_cases_in_corpus']} "
        f"corpus cases, so these are lower bounds. In {o['gold_case_already_overruled_when_quoted']} "
        "queries the case the court itself quoted had already received negative treatment."
    )
    return "\n\n".join(out)


def jurisdiction() -> str:
    r = load("robustness/jurisdiction_bm25.json")
    if not r:
        return pending("jurisdiction experiment", "legalrag experiment jurisdiction", "Not run.")
    b, g, w = r["binding_only_filter"], r["gold_authority"], r["wrong_circuit"]
    bind, pers = b["queries_whose_gold_is_binding"], b["queries_whose_gold_is_persuasive"]
    rows = [
        ["No court filter", f"all, n = {r['config']['n']}", ci(r["no_court_filter"]["recall@10"])],
        ["Binding authority only", f"all, n = {r['config']['n']}", ci(b["all_queries"]["recall@10"])],
        ["No court filter", f"quoted case is binding, n = {bind['n']}",
         ci(bind["no_filter"]["recall@10"])],
        ["Binding authority only", f"quoted case is binding, n = {bind['n']}",
         ci(bind["binding_filter"]["recall@10"])],
        ["No court filter", f"quoted case is persuasive, n = {pers['n']}",
         num(pers["no_filter_recall@10"])],
        ["Binding authority only", f"quoted case is persuasive, n = {pers['n']}",
         num(pers["binding_filter_recall@10"])],
        ["Another circuit's court of appeals only", f"all, n = {r['config']['n']}",
         num(w["recall@10"])],
    ]  # fmt: skip
    out = [
        f"For {g['binding_share']:.1%} of test queries the case the court quoted was binding "
        "on that court (a Supreme Court decision or one of its own circuit).",
        table(["Retrieval pool", "Queries", "Recall@10"], rows),
        f"Under the wrong-circuit filter the best retrieval score separates matched from "
        f"mismatched searches with AUROC {w['auroc_top1_score_matched_vs_mismatched']:.2f}.",
    ]
    s = r["scope_detector"]
    rows = [[k.replace("_", " "), v["n"], f"{v['flagged']} ({v['recall']:.0%})"]
            for k, v in s.items() if k != "in_scope"]  # fmt: skip
    rows.append(["in scope (false alarms)", s["in_scope"]["n"],
                 f"{s['in_scope']['false_alarms']} ({s['in_scope']['false_alarm_rate']:.0%})"])  # fmt: skip
    out.append("**Scope detection** on hand-written questions:")
    out.append(table(["Question group", "n", "Flagged as outside the corpus"], rows))
    return "\n\n".join(out)


def abstention() -> str:
    files = (
        sorted((RESULTS / "abstention").glob("abstention_*.json"))
        if (RESULTS / "abstention").exists()
        else []
    )
    if not files:
        return pending("abstention (RQ7)", "legalrag experiment abstention", "Not run.")
    out = []
    for f in files:
        r = json.loads(f.read_text())
        cfg, top = r["config"], r["signals"]["top1"]
        signal = "cross-encoder score" if cfg["reranker"] else "BM25 score"
        rows = [
            [name.replace("_", " "), m["n_unanswerable"], num(m["auroc"], 2),
             f"{m['abstain_recall']:.0%}", f"{m['coverage_answerable']:.0%}"]
            for name, m in top["test"].items()
        ]  # fmt: skip
        out.append(
            f"Signal: {signal} of the best passage"
            + (f" ({cfg['reranker']})" if cfg["reranker"] else "")
            + f". Threshold tuned on {cfg['n_dev']} development queries; {cfg['n_test']} test queries."
        )
        out.append(table(["Condition in which the quoted case is unreachable", "n", "AUROC",
                          "Abstained (correct)", "Answerable queries still answered"], rows))  # fmt: skip
        out.append(
            f"The same score predicts whether the quoted case is in the top five with AUROC "
            f"{top['predicts_gold_in_top5_auroc']:.2f}."
        )
    c = json.loads(files[0].read_text())["out_of_scope_combined"]
    out.append(
        f"On the out-of-scope questions the scope rule alone abstains on "
        f"{c['abstained_by_rule']:.0%}; rule or score together on {c['abstained_by_either']:.0%}."
    )
    return "\n\n".join(out)


def verification() -> str:
    out = []
    judge_files = (
        sorted((RESULTS / "verification").glob("judge_validation_*.json"))
        if (RESULTS / "verification").exists()
        else []
    )
    if judge_files:
        r = json.loads(max(judge_files, key=lambda f: len(f.name)).read_text())
        rows = []
        for name, j in r["judges"].items():
            a = j["accept_rate"]
            rows.append([name, ci({"value": a["supported"]["accepted"], **a["supported"]}, 2),
                         f"{a['other_case']['accepted']:.2f}", f"{a['same_case']['accepted']:.2f}",
                         f"{a['negated']['accepted']:.2f}", f"{j['accuracy']:.2f}"])  # fmt: skip
        out.append(f"**Support judges** ({r['config']['n_per_kind']} pairs of each kind). "
                   "A good judge accepts the first column and rejects the other three.")  # fmt: skip
        out.append(table(["Judge", "Accepts true quotation", "Accepts: other case",
                          "Accepts: same case, wrong passage", "Accepts: negated", "Accuracy"], rows))  # fmt: skip
    else:
        out.append(pending("judge validation", "legalrag experiment judge", "Not run."))
    check_files = (
        sorted((RESULTS / "verification").glob("citation_checker_*.json"))
        if (RESULTS / "verification").exists()
        else []
    )
    for f in check_files:
        r = json.loads(f.read_text())
        d = r["detection"]
        rows = [[k.replace("_", " "), "exact lookup", v["n"], ci({"value": v["rate"], **v}, 3)]
                for k, v in d["exact_lookup_checks"].items()]  # fmt: skip
        rows += [[k.replace("_", " "), "support judge", v["n"], ci({"value": v["rate"], **v}, 3)]
                 for k, v in d["support_judge_checks"].items()]  # fmt: skip
        fa = r["false_alarm_on_clean_statements"]
        out.append(f"**Planted citation errors**, judge `{r['config']['judge']}`. False alarms on "
                   f"{fa['n']:,} clean verbatim statements: {fa['rate']:.1%}.")  # fmt: skip
        out.append(table(["Planted error", "Check", "n", "Detected"], rows))
    return "\n\n".join(out)


def rag() -> str:
    files = sorted((RESULTS / "rag").glob("rag_*.json")) if (RESULTS / "rag").exists() else []
    if not files:
        return pending("grounded generation (RQ4)", "legalrag experiment rag", "Not run.")
    keys = [
        ("abstained", "Abstained"),
        ("citation_precision", "Citation precision"),
        ("citation_recall", "Citation recall"),
        ("unsupported_claim_rate", "Unsupported-claim rate"),
        ("hallucinated_citation_rate", "Answers with a hallucinated citation"),
        ("gold_case_cited", "Cites the quoted case"),
        ("gold_passage_cited", "Cites the quoted passage"),
        ("gold_quote_coverage", "Quotation words reproduced"),
    ]
    out = []
    for f in files:
        r = json.loads(f.read_text())
        cfg = r["config"]
        columns, cells = [], []
        for cond, v in r["conditions"].items():
            for policy, agg in v.items():
                columns.append(f"{cond}, {policy.replace('_', ' ')}")
                cells.append(agg)
        rows = [[label] + [ci(c.get(key), 2) for c in cells] for key, label in keys]
        out.append(
            f"**Generator `{cfg['generator']}`**, retrieval {cfg['retriever']}, "
            f"{cfg['n_queries']} test queries, {cfg['evidence_k']} evidence passages. Support "
            f"judged by `{cfg['eval_judge']}`"
            + (f"; strict filter uses `{cfg['filter_judge']}`" if cfg["filter_judge"] else "")
            + f". Generation: {r['generation_seconds']['p50'] * 1000:.0f} ms median."
        )
        out.append(table(["Measure", *columns], rows))
    return "\n\n".join(out)


def systems() -> str:
    files = (
        sorted((RESULTS / "systems").glob("systems_*.json"))
        if (RESULTS / "systems").exists()
        else []
    )
    if not files:
        return pending("latency and resources", "legalrag experiment systems", "Not run.")
    r = json.loads(max(files, key=lambda f: len(f.name)).read_text())
    rows = [[k.replace("_", " "), f"{v['mean_ms']:.1f}", f"{v['p50_ms']:.1f}", f"{v['p95_ms']:.1f}"]
            for k, v in r["stages"].items()]  # fmt: skip
    out = [
        f"Per query, one at a time, on {r['device']['platform']} ({r['device']['device']}), "
        f"{r['config']['n_queries']} queries.",
        table(["Stage", "Mean ms", "Median ms", "p95 ms"], rows),
    ]
    rows = [[k, f"{v['disk_mb']:.0f}",
             f"{v['build']['docs_per_second']:.0f} passages/s on {v['build']['device']}"
             if v.get("build") and v["build"].get("docs_per_second") else "n/a"]
            for k, v in r["indexes"].items()]  # fmt: skip
    out.append(table(["Index", "Disk MB", "Build rate"], rows))
    s = r["storage_mb"]
    out.append(
        f"Storage: raw opinions {s['raw_opinions']:.0f} MB, processed corpus {s['processed']:.0f} "
        f"MB. Peak resident memory of the run: {r['peak_resident_memory_gb']:.1f} GB."
    )
    return "\n\n".join(out)


SECTIONS = [
    ("Retrieval: BM25, dense and hybrid (RQ1, RQ2)", retrieval),
    ("Cross-encoder reranking (RQ3)", rerank),
    ("Fixed-size against section-aware chunking", chunking),
    ("Query decomposition (RQ5)", decomposition),
    ("Time: future-case leakage and outdated authority (RQ6)", temporal),
    ("Jurisdiction (RQ6)", jurisdiction),
    ("Grounded generation and citation quality (RQ4)", rag),
    ("Citation verification", verification),
    ("Abstention (RQ7)", abstention),
    ("Latency and resources", systems),
]


def render() -> str:
    parts = []
    for title, fn in SECTIONS:
        parts.append(f"### {title}\n\n{fn()}")
    return "\n\n".join(parts)


def main() -> int:
    readme = ROOT / "README.md"
    text = readme.read_text()
    if START not in text or END not in text:
        print("README.md has no results markers", file=sys.stderr)
        return 1
    head, rest = text.split(START, 1)
    _, tail = rest.split(END, 1)
    new = f"{head}{START}\n{render()}\n{END}{tail}"
    if "--check" in sys.argv:
        if new != text:
            print("README.md results section is stale; run scripts/make_results_tables.py")
            return 1
        return 0
    readme.write_text(new)
    return 0


if __name__ == "__main__":
    sys.exit(main())
