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
from legalrag.verify.judges import make_judge

KINDS = ("supported", "other_case", "same_case", "negated")


def build_pairs(
    bench: Bench, n: int, seed: int = 0, split: str = "test"
) -> dict[str, list[tuple[str, str]]]:
    rng = random.Random(seed)
    by_case: dict[str, list] = {}
    for chunk in bench.store.chunks:
        by_case.setdefault(chunk.case_id, []).append(chunk)
    pairs: dict[str, list[tuple[str, str]]] = {k: [] for k in KINDS}
    for q in bench.queries(split):
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


def _rates(verdicts, threshold: float) -> dict:
    """Acceptance by kind, accuracy and precision when accepting at ``threshold``."""
    accepted = {k: sum(j.score >= threshold for j in v) for k, v in verdicts.items()}
    n = {k: len(v) for k, v in verdicts.items()}
    negatives = [k for k in KINDS if k != "supported"]
    false_accepts = sum(accepted[k] for k in negatives)
    total_neg = sum(n[k] for k in negatives)
    rows = {}
    for kind in KINDS:
        p, lo, hi = wilson(accepted[kind], n[kind])
        rows[kind] = {
            "n": n[kind],
            "accepted": round(p, 4),
            "ci_low": round(lo, 4),
            "ci_high": round(hi, 4),
        }
    return {
        "threshold": round(threshold, 4),
        "accept_rate": rows,
        "recall_on_supported": rows["supported"]["accepted"],
        "precision_of_supported_label": round(
            accepted["supported"] / max(1, accepted["supported"] + false_accepts), 4
        ),
        "accuracy": round(
            (accepted["supported"] + total_neg - false_accepts) / (n["supported"] + total_neg), 4
        ),
    }


def _calibrate(verdicts) -> float:
    """Threshold with the best accuracy on the given pairs."""
    scores = sorted({round(j.score, 3) for v in verdicts.values() for j in v})
    return max(scores, key=lambda t: (_rates(verdicts, t)["accuracy"], -abs(t - 0.5)))


def run(n: int = 300, judges: str | list[str] = "lexical", seed: int = 0):
    """Validate judges on test pairs, at the default threshold and at a threshold
    calibrated on pairs built from development queries."""
    bench = Bench.load()
    dev_pairs = build_pairs(bench, int(n), seed, "dev")
    pairs = build_pairs(bench, int(n), seed, "test")
    names = [judges] if isinstance(judges, str) else list(judges)
    out = {}
    for name in names:
        judge = make_judge(name)
        calibrated = _calibrate({k: judge.judge_many(v) for k, v in dev_pairs.items()})
        verdicts = {k: judge.judge_many(v) for k, v in pairs.items()}
        pos = [j.score for j in verdicts["supported"]]
        out[judge.name] = {
            **_rates(verdicts, judge.threshold),
            "calibrated_on_dev": _rates(verdicts, calibrated),
            "auroc": {
                k: round(
                    auroc(pos + [j.score for j in verdicts[k]], [1] * len(pos) + [0] * len(pos)), 4
                )
                for k in KINDS
                if k != "supported"
            },
        }
        del judge
    result = run_record(
        {
            "experiment": "judge_validation",
            "n_per_kind": len(pairs["supported"]),
            "n_per_kind_dev": len(dev_pairs["supported"]),
            "seed": seed,
        },
        bench.store.corpus_version,
        judges=out,
    )
    tag = "+".join(n.replace("/", "_") for n in names)
    write_result(bench.paths.results / "verification" / f"judge_validation_{tag}.json", result)
    return result
