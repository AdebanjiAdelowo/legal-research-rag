"""How reliable are the automatic support judges on legal text?

Labelled pairs are built from the benchmark with no human or model in the loop:

``supported``       a court's quotation of a passage, against the chunk that holds
                    the quoted paragraph. Quotations are near-verbatim, sometimes
                    with bracketed alterations and elisions.
``other_case``      the same quotation against a random chunk of a different case.
``same_case``       the same quotation against another chunk of the same opinion
                    that does not hold the quoted paragraph. Topically close.
``negated``         the quotation with "not" inserted, against the right chunk.

A judge should accept the first and reject the other three. This measures a lower
bound on difficulty: real generated claims paraphrase, and entailment under
paraphrase is harder than recognising a quotation. A judge that fails here is
unusable; a judge that passes here is not thereby shown to be reliable.
"""

from __future__ import annotations

import random

from legalrag.eval.stats import auroc, wilson
from legalrag.experiments.citations import negate
from legalrag.experiments.common import Bench
from legalrag.provenance import run_record, write_result
from legalrag.verify.judges import SUPPORTED, make_judge

KINDS = ("supported", "other_case", "same_case", "negated")


def build_pairs(bench: Bench, n: int, seed: int = 0) -> dict[str, list[tuple[str, str]]]:
    rng = random.Random(seed)
    by_case: dict[str, list] = {}
    for chunk in bench.store.chunks:
        by_case.setdefault(chunk.case_id, []).append(chunk)
    pairs: dict[str, list[tuple[str, str]]] = {k: [] for k in KINDS}
    for q in bench.queries("test"):
        if len(pairs["supported"]) >= n:
            break
        quote = q.gold_quotes[0]
        if not q.gold_paras or not 8 <= len(quote.split()) <= 80:
            continue
        gold = set(q.gold_paras)
        own = by_case[q.gold_cases[0]]
        holder = [c for c in own if gold & set(c.para_ids)]
        rest = [c for c in own if not gold & set(c.para_ids)]
        negated = negate(quote)
        if not holder or not rest or not negated:
            continue
        other = rng.choice(bench.store.chunks)
        while other.case_id == q.gold_cases[0]:
            other = rng.choice(bench.store.chunks)
        pairs["supported"].append((quote, holder[0].text))
        pairs["other_case"].append((quote, other.text))
        pairs["same_case"].append((quote, rng.choice(rest).text))
        pairs["negated"].append((negated, holder[0].text))
    return pairs


def run(n: int = 300, judges: str | list[str] = "lexical", seed: int = 0):
    bench = Bench.load()
    pairs = build_pairs(bench, int(n), seed)
    names = [judges] if isinstance(judges, str) else list(judges)
    out = {}
    for name in names:
        judge = make_judge(name)
        verdicts = {k: judge.judge_many(v) for k, v in pairs.items()}
        accepted = {k: sum(j.label == SUPPORTED for j in v) for k, v in verdicts.items()}
        rows = {}
        for kind in KINDS:
            p, lo, hi = wilson(accepted[kind], len(pairs[kind]))
            rows[kind] = {
                "n": len(pairs[kind]),
                "accepted": round(p, 4),
                "ci_low": round(lo, 4),
                "ci_high": round(hi, 4),
            }
        pos = [j.score for j in verdicts["supported"]]
        negatives = [k for k in KINDS if k != "supported"]
        total_neg = sum(len(pairs[k]) for k in negatives)
        false_accepts = sum(accepted[k] for k in negatives)
        true_accepts = accepted["supported"]
        out[judge.name] = {
            "threshold": judge.threshold,
            "accept_rate": rows,
            "recall_on_supported": rows["supported"]["accepted"],
            "precision_of_supported_label": round(
                true_accepts / max(1, true_accepts + false_accepts), 4
            ),
            "accuracy": round(
                (true_accepts + total_neg - false_accepts) / (len(pos) + total_neg), 4
            ),
            "auroc": {
                k: round(
                    auroc(pos + [j.score for j in verdicts[k]], [1] * len(pos) + [0] * len(pos)), 4
                )
                for k in negatives
            },
        }
        del judge
    result = run_record(
        {"experiment": "judge_validation", "n_per_kind": len(pairs["supported"]), "seed": seed},
        bench.store.corpus_version,
        judges=out,
    )
    tag = "+".join(n.replace("/", "_") for n in names)
    write_result(bench.paths.results / "verification" / f"judge_validation_{tag}.json", result)
    return result
