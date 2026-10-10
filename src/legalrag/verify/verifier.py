"""Citation verification.

A generated statement can go wrong in distinct ways, and the checks are kept
separate so that passing one is never mistaken for passing another:

1. **Marker validity.** ``[E7]`` when only five passages were supplied.
2. **Citation existence.** A reporter citation written into the text is looked up in
   the corpus citation index. "Not found" means only that the corpus does not hold
   it; the case may exist. A citation in a reporter the parser does not know is
   reported as unresolvable.
3. **Provenance.** A citation that resolves to a corpus case which was *not* among
   the supplied passages came from the model's memory, not from the evidence.
4. **Name agreement.** ``Smith v. Jones, 410 U.S. 113`` where 410 U.S. 113 is a
   different case.
5. **Quotation fidelity.** Text in quotation marks must occur in a cited passage.
6. **Substantive support.** Whether the cited passage supports the claim, decided by
   a support judge (``judges.py``).

Checks 1 to 5 are exact lookups. Check 6 is an automatic judgment with a known error
rate and is labelled as such everywhere it is shown. A citation that exists and
points at a real passage is "existing"; only one that also passes check 6 is
"supported".
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass, field

from legalrag.data.benchmark import tokens
from legalrag.data.citations import REPORTERS, find_citations
from legalrag.data.schema import Case
from legalrag.rag import types as T
from legalrag.rag.types import Answer, Evidence, Statement
from legalrag.verify.judges import CONTRADICTED, SUPPORTED, Judge, LexicalJudge

_QUOTE = re.compile(r"[\"“]([^\"“”]{20,600})[\"”]")
_NAME_BEFORE = re.compile(
    r"([A-Z][\w.'’&-]*(?:\s+(?:[A-Z][\w.'’&-]*|of|the|and|for|ex|rel\.|&))*\s+v\.?\s+"
    r"[A-Z][\w.'’&-]*(?:\s+(?:[A-Z][\w.'’&-]*|of|the|and|&))*),?\s*$"
)
_STOP_NAME = {"united", "states", "the", "of", "and", "inc", "co", "corp", "v", "et", "al"}


class CitationIndex:
    """Reporter citation -> case, built from every citation of every corpus case."""

    def __init__(self, cases: dict[str, Case]):
        self.cases = cases
        self.by_key: dict[str, str] = {}
        for case in cases.values():
            for cite in case.citations:
                for c in find_citations(cite):
                    if c.key:
                        self.by_key.setdefault(c.key, case.case_id)

    def resolve(self, key: str | None) -> Case | None:
        case_id = self.by_key.get(key or "")
        return self.cases[case_id] if case_id else None


def _name_words(name: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]{3,}", name.lower()) if w not in _STOP_NAME}


def names_agree(written: str, case: Case) -> bool:
    """Whether a written case name plausibly names ``case``: at least half of its
    distinctive words occur in the case's short or full name."""
    want = _name_words(written)
    if not want:
        return True
    have = _name_words(case.short_name) | _name_words(case.name)
    return len(want & have) / len(want) >= 0.5


def quote_found(quote: str, passage: str, threshold: float = 0.8) -> bool:
    """Whether a quotation occurs in a passage, tolerating OCR noise and elisions:
    at least ``threshold`` of its word trigrams must be present."""
    q = tokens(quote)
    if len(q) < 3:
        return " ".join(q) in " ".join(tokens(passage))
    p = tokens(passage)
    grams = {tuple(q[i : i + 3]) for i in range(len(q) - 2)}
    have = {tuple(p[i : i + 3]) for i in range(len(p) - 2)}
    return len(grams & have) / len(grams) >= threshold


@dataclass
class CitationCheck:
    """Result for one citation attached to one statement."""

    kind: str  # "marker" | "reporter"
    raw: str
    exists: bool | None  # None: cannot be resolved against the corpus
    in_evidence: bool
    name_matches: bool | None = None
    resolves_to: str = ""
    support: str = ""  # supported | unsupported | contradicted | "" when not judged
    support_score: float | None = None
    problems: list[str] = field(default_factory=list)


@dataclass
class StatementCheck:
    index: int
    type: str
    citations: list[CitationCheck]
    quotes_missing: list[str]
    status: str  # verified | unsupported | uncited | not_applicable
    problems: list[str]


def _check_reporter_citations(
    text: str, evidence: list[Evidence], index: CitationIndex
) -> list[CitationCheck]:
    out = []
    evidence_cases = {e.case_id for e in evidence}
    for c in find_citations(text):
        if c.is_short_form:
            continue
        check = CitationCheck("reporter", c.raw, None, False)
        if c.reporter not in REPORTERS:
            check.problems.append("reporter not recognised; cannot be checked")
        else:
            case = index.resolve(c.key)
            check.exists = case is not None
            if case is None:
                check.problems.append(
                    "citation not found in the indexed corpus; it may be wrong or "
                    "simply outside the corpus"
                )
            else:
                check.resolves_to = case.bluebook()
                check.in_evidence = case.case_id in evidence_cases
                if not check.in_evidence:
                    check.problems.append(
                        "cited case was not among the supplied passages; the "
                        "reference did not come from the evidence"
                    )
                m = _NAME_BEFORE.search(text[: c.start])
                if m:
                    check.name_matches = names_agree(m.group(1), case)
                    if not check.name_matches:
                        check.problems.append(
                            f"citation belongs to {case.short_name}, not to the case named"
                        )
        out.append(check)
    return out


class CitationVerifier:
    def __init__(self, index: CitationIndex, judge: Judge | None = None):
        self.index = index
        self.judge = judge or LexicalJudge()

    def check_statements(
        self, statements: list[Statement], evidence: list[Evidence]
    ) -> list[StatementCheck]:
        by_label = {e.label: e for e in evidence}
        checks: list[StatementCheck] = []
        pending: list[tuple[int, int, str, str]] = []  # (statement, citation, claim, passage)
        for i, s in enumerate(statements):
            cites: list[CitationCheck] = []
            problems: list[str] = []
            for label in s.citations:
                e = by_label.get(label)
                if e is None:
                    cites.append(
                        CitationCheck(
                            "marker",
                            f"[{label}]",
                            False,
                            False,
                            problems=["no such evidence passage"],
                        )
                    )
                    continue
                cites.append(
                    CitationCheck("marker", f"[{label}]", True, True, resolves_to=e.reference)
                )
                if not e.is_majority:
                    problems.append(
                        f"[{label}] is from a {e.opinion_type}, not the opinion of the court"
                    )
                if e.treatment:
                    problems.append(
                        f"[{label}] {e.case_name}: later {e.treatment['treatment']} by "
                        f"{e.treatment['by']} ({e.treatment['year']})"
                    )
                if s.type in T.CITABLE:
                    pending.append((i, len(cites) - 1, s.text, e.text))
            cites += _check_reporter_citations(s.text, evidence, self.index)

            cited_text = " ".join(by_label[c].text for c in s.citations if c in by_label)
            missing = [
                q
                for q in _QUOTE.findall(s.text)
                if not cited_text or not quote_found(q, cited_text)
            ]
            checks.append(StatementCheck(i, s.type, cites, missing, "", problems))

        for (i, j, _, _), judgment in zip(
            pending, self.judge.judge_many([(c, p) for _, _, c, p in pending]), strict=True
        ):
            cite = checks[i].citations[j]
            cite.support = judgment.label
            cite.support_score = judgment.score
            if judgment.label == CONTRADICTED:
                cite.problems.append("the cited passage appears to contradict the statement")
            elif judgment.label != SUPPORTED:
                cite.problems.append("the cited passage does not appear to support the statement")

        for check, s in zip(checks, statements, strict=True):
            for q in check.quotes_missing:
                check.problems.append(f"quotation not found in the cited passage: “{q[:80]}”")
            for c in check.citations:
                check.problems += [f"{c.raw}: {p}" for p in c.problems]
            if s.type not in T.CITABLE:
                stray = [c for c in check.citations if c.kind == "reporter"]
                check.status = (
                    "unsupported" if stray and any(c.problems for c in stray) else "not_applicable"
                )
                continue
            markers = [c for c in check.citations if c.kind == "marker" and c.exists]
            bad_reporter = any(
                c.kind == "reporter" and (c.exists is False or c.name_matches is False
                                          or (c.exists and not c.in_evidence))
                for c in check.citations
            )  # fmt: skip
            if not markers:
                check.status = "uncited"
                check.problems.append("statement cites no valid evidence passage")
            elif (
                any(c.support == SUPPORTED for c in markers)
                and not check.quotes_missing
                and not bad_reporter
            ):
                check.status = "verified"
            else:
                check.status = "unsupported"
        return checks

    def verify(self, answer: Answer, policy: str = "flag") -> dict:
        """Check an answer in place and return a summary.

        ``policy="flag"`` attaches results and leaves the text alone.
        ``policy="strict"`` also removes citable statements that are not verified
        and records each removal as a ``not_established`` statement, so the reader
        sees that something was withheld and why.
        """
        checks = self.check_statements(answer.statements, answer.evidence)
        for s, c in zip(answer.statements, checks, strict=True):
            s.verification = asdict(c)
        summary = summarise(checks)
        summary["judge"] = self.judge.name
        summary["policy"] = policy
        summary["note"] = (
            "Existence checks are exact lookups against the indexed corpus. Support "
            "checks are automatic judgments with a measured error rate; they are not "
            "a legal determination."
        )
        if policy == "strict":
            kept: list[Statement] = []
            removed = 0
            for s, c in zip(answer.statements, checks, strict=True):
                if s.type in T.CITABLE and c.status != "verified":
                    removed += 1
                    continue
                if s.type not in T.CITABLE and c.status == "unsupported":
                    removed += 1
                    continue
                kept.append(s)
            if removed:
                kept.append(
                    Statement(
                        T.NOT_ESTABLISHED,
                        f"{removed} generated statement(s) were withheld because the "
                        "automatic check could not confirm them against the cited passages.",
                    )
                )
            summary["removed"] = removed
            answer.statements = kept
            if not any(s.type in T.CITABLE for s in kept):
                answer.abstained = True
                answer.abstain_reason = (
                    "no generated statement could be confirmed against the retrieved passages"
                )
        answer.verification = summary
        return summary


def summarise(checks: list[StatementCheck]) -> dict:
    """Counts over statements and citations.

    ``citation_precision`` is the share of evidence citations whose passage supports
    the statement; ``citation_recall`` the share of citable statements with at least
    one supporting citation; ``unsupported_claim_rate`` its complement;
    ``hallucinated_citations`` counts markers that point at nothing and reporter
    citations that do not resolve or did not come from the evidence.
    """
    status = Counter(c.status for c in checks)
    citable = [c for c in checks if c.type in T.CITABLE]
    markers = [x for c in citable for x in c.citations if x.kind == "marker" and x.exists]
    supported = sum(x.support == SUPPORTED for x in markers)
    all_cites = [x for c in checks for x in c.citations]
    hallucinated = sum(
        (x.kind == "marker" and not x.exists)
        or (x.kind == "reporter" and (x.exists is False or (x.exists and not x.in_evidence)))
        for x in all_cites
    )
    recall_hits = sum(
        any(x.kind == "marker" and x.support == SUPPORTED for x in c.citations) for c in citable
    )
    n = len(citable)
    return {
        "statements": len(checks),
        "citable_statements": n,
        "by_status": dict(status),
        "citations": len(all_cites),
        "evidence_citations": len(markers),
        "supported_citations": supported,
        "citation_precision": round(supported / len(markers), 4) if markers else None,
        "citation_recall": round(recall_hits / n, 4) if n else None,
        "unsupported_claim_rate": round(1 - recall_hits / n, 4) if n else None,
        "hallucinated_citations": hallucinated,
        "mismatched_names": sum(x.name_matches is False for x in all_cites),
        "unresolvable_citations": sum(x.kind == "reporter" and x.exists is None for x in all_cites),
        "quotations_not_found": sum(len(c.quotes_missing) for c in checks),
    }
