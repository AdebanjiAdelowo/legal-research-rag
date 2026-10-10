"""Does the citation verifier catch planted errors?

Clean answers come from the extractive generator, whose statements are verbatim
sentences from the passages they cite. Each clean statement is then corrupted in
one specific way, and the verifier is asked to check it. Because the corruption is
known, detection can be scored exactly.

Corruptions, and the check that should fire:

``fabricated_citation``   a reporter citation to a case that is not in the corpus
                          -> existence
``real_but_unretrieved``  a correct citation to a corpus case that was not among the
                          passages -> provenance
``wrong_name``            the cited case's own citation under another case's name
                          -> name agreement
``marker_out_of_range``   an evidence label that does not exist -> marker validity
``altered_quotation``     the sentence put in quotation marks with words replaced
                          -> quotation fidelity
``swapped_marker``        the statement attributed to a different passage that is
                          also in the evidence -> support judge
``swapped_claim``         the citation kept, the statement replaced by a sentence
                          from an unrelated case -> support judge
``negated``               "not" inserted into the statement -> support judge

The first five are exact lookups and should be caught every time. The last three
depend on the support judge; ``negated`` is the case a lexical judge cannot catch.

The clean statements give the false-alarm rate. They are verbatim, so this measures
the verifier on the easiest possible true claims; paraphrased true claims are
harder and are covered by the judge validation and generation experiments.
"""

from __future__ import annotations

import random
import re

from legalrag.eval.stats import wilson
from legalrag.experiments.common import Bench
from legalrag.provenance import run_record, write_result
from legalrag.rag import types as T
from legalrag.rag.evidence import select_evidence
from legalrag.rag.generate import ExtractiveGenerator
from legalrag.rag.types import Statement
from legalrag.verify.judges import make_judge
from legalrag.verify.verifier import CitationIndex, CitationVerifier

_AUX = re.compile(
    r"\b(is|are|was|were|must|may|can|should|will|would|does|do|did|has|have|had)\b(?! not)"
)
LOOKUP = ("fabricated_citation", "real_but_unretrieved", "wrong_name", "marker_out_of_range",
          "altered_quotation")  # fmt: skip
JUDGED = ("swapped_marker", "swapped_claim", "negated")


def negate(text: str) -> str | None:
    """Insert ``not`` after the first auxiliary verb, or None if there is none."""
    m = _AUX.search(text)
    return f"{text[: m.end()]} not{text[m.end() :]}" if m else None


def corrupt(kind: str, s: Statement, evidence, bench: Bench, rng: random.Random):
    """A corrupted copy of ``s``, or None when the corruption does not apply."""
    cases = bench.store.cases
    label = s.citations[0]
    base = s.text.rstrip(".")
    in_evidence = {e.case_id for e in evidence}
    if kind == "fabricated_citation":
        vol, page = rng.randint(700, 899), rng.randint(1, 900)
        return Statement(s.type, f"{base}, see Halden v. Morrow, {vol} U.S. {page}.", [label])
    if kind == "real_but_unretrieved":
        other = cases[rng.choice([c for c in bench.store.case_ids if c not in in_evidence])]
        return Statement(s.type, f"{base}, see {other.short_name}, {other.official_cite}.", [label])
    if kind == "wrong_name":
        own = cases[next(e.case_id for e in evidence if e.label == label)]
        text = f"{base}, see Halden v. Morrow, {own.official_cite}."
        return Statement(s.type, text, [label])
    if kind == "marker_out_of_range":
        return Statement(s.type, s.text, [f"E{len(evidence) + rng.randint(2, 9)}"])
    if kind == "altered_quotation":
        words = base.split()
        if len(words) < 10:
            return None
        filler = ["manifestly", "statutory", "defendant", "never", "ordinarily", "remand"]
        for i in rng.sample(range(len(words)), max(3, len(words) // 3)):
            words[i] = rng.choice(filler)
        return Statement(s.type, f"The court stated: “{' '.join(words)}”.", [label])
    if kind == "swapped_marker":
        others = [e.label for e in evidence if e.label != label and e.case_id not in
                  {x.case_id for x in evidence if x.label == label}]  # fmt: skip
        return Statement(s.type, s.text, [rng.choice(others)]) if others else None
    if kind == "swapped_claim":
        for _ in range(20):
            chunk = rng.choice(bench.store.chunks)
            if chunk.case_id in in_evidence:
                continue
            _, statements, _ = ExtractiveGenerator(min_overlap=0.0).generate(
                chunk.text, _as_evidence(chunk)
            )
            if statements and statements[0].type in T.CITABLE:
                return Statement(s.type, statements[0].text, [label])
        return None
    if kind == "negated":
        text = negate(s.text)
        return Statement(s.type, text, [label]) if text else None
    raise ValueError(kind)


def _as_evidence(chunk):
    from legalrag.rag.types import Evidence

    return [
        Evidence(
            "E1",
            chunk.chunk_id,
            chunk.case_id,
            "",
            "",
            "",
            "",
            "",
            None,
            "",
            chunk.opinion_type,
            chunk.section,
            chunk.para_ids,
            chunk.text,
            0.0,
        )
    ]


def _detected(kind: str, check) -> bool:
    """Whether the specific check for ``kind`` fired (not merely any flag)."""
    cites = check.citations
    reporters = [c for c in cites if c.kind == "reporter"]
    if kind == "fabricated_citation":
        return any(c.exists is False and not c.in_passage for c in reporters)
    if kind == "real_but_unretrieved":
        return any(c.exists and not c.in_evidence and not c.in_passage for c in reporters)
    if kind == "wrong_name":
        return any(c.name_matches is False for c in reporters)
    if kind == "marker_out_of_range":
        return any(c.kind == "marker" and c.exists is False for c in cites)
    if kind == "altered_quotation":
        return bool(check.quotes_missing)
    return check.status != "verified"


def run(limit: int | None = 300, judge: str = "lexical", seed: int = 0,
        k1: float | None = None, b: float | None = None):  # fmt: skip
    bench = Bench.load()
    if k1 is not None and b is not None:
        bench.bm25.set_params(float(k1), float(b))
    rng = random.Random(seed)
    queries = bench.queries("test", limit)
    run_ = bench.first_stage("bm25", [q.text for q in queries], bench.masks(queries), depth=200)
    verifier = CitationVerifier(CitationIndex(bench.store.cases), make_judge(judge))
    generator = ExtractiveGenerator()

    counts = {k: [0, 0] for k in (*LOOKUP, *JUDGED)}  # detected, total
    clean = [0, 0]  # flagged, total
    for q, ranking in zip(queries, run_.rankings, strict=True):
        evidence = select_evidence(ranking, bench.store, k=5, per_case=2)
        _, statements, _ = generator.generate(q.text, evidence)
        citable = [s for s in statements if s.type in T.CITABLE]
        if not citable:
            continue
        for check in verifier.check_statements(citable, evidence):
            clean[0] += check.status != "verified"
            clean[1] += 1
        batch, kinds = [], []
        for s in citable:
            for kind in counts:
                bad = corrupt(kind, s, evidence, bench, rng)
                if bad is not None:
                    batch.append(bad)
                    kinds.append(kind)
        for kind, check in zip(kinds, verifier.check_statements(batch, evidence), strict=True):
            counts[kind][0] += _detected(kind, check)
            counts[kind][1] += 1

    def rate(hits, n):
        p, lo, hi = wilson(hits, n)
        return {"n": n, "rate": round(p, 4), "ci_low": round(lo, 4), "ci_high": round(hi, 4)}

    result = run_record(
        {
            "experiment": "citation_checker",
            "judge": verifier.judge.name,
            "generator": "extractive",
            "n_queries": len(queries),
            "seed": seed,
        },
        bench.store.corpus_version,
        false_alarm_on_clean_statements=rate(*clean),
        detection={
            "exact_lookup_checks": {k: rate(*counts[k]) for k in LOOKUP},
            "support_judge_checks": {k: rate(*counts[k]) for k in JUDGED},
        },
    )
    write_result(bench.paths.results / "verification" / f"citation_checker_{judge}.json", result)
    return result
