"""Rendering an answer as a Markdown research report for download."""

from __future__ import annotations

import time

from legalrag.rag import types as T
from legalrag.rag.types import Answer

_HEADINGS = [
    (T.REASONING, "Judicial reasoning in the retrieved cases"),
    (T.FACT, "Facts of the retrieved cases"),
    (T.PROVISION, "Legal provisions referred to"),
    (T.INTERPRETATION, "System interpretation (not authority)"),
    (T.NOT_ESTABLISHED, "Not established by the retrieved evidence"),
]
_MARK = {"verified": "supported", "unsupported": "NOT CONFIRMED", "uncited": "NO CITATION"}


def to_markdown(answer: Answer) -> str:
    lines = ["# Legal research report", "", f"> **Limitation.** {answer.limitation}", ""]
    for w in answer.warnings:
        lines += [f"> {w}", ""]
    lines += ["## Question", "", answer.question, ""]

    plan = answer.plan
    notes = list(plan.get("notes") or [])
    if plan.get("text") and plan["text"] != answer.question:
        notes.insert(0, f"retrieval text: “{plan['text']}”")
    if notes:
        lines += ["## How the question was read", ""] + [f"- {n}" for n in notes] + [""]

    if answer.abstained:
        lines += [
            "## No answer given",
            "",
            f"The system abstained: {answer.abstain_reason}.",
            "",
            "Abstention means the indexed opinions did not provide sufficient support. "
            "It does not mean that no authority exists.",
            "",
        ]
    by_label = {e.label: e for e in answer.evidence}
    for kind, heading in _HEADINGS:
        items = [s for s in answer.statements if s.type == kind]
        if not items:
            continue
        lines += [f"## {heading}", ""]
        for s in items:
            cites = ", ".join(
                f"[{c}] {by_label[c].reference}" if c in by_label else f"[{c}] (no such passage)"
                for c in s.citations
            )
            text = f"“{s.text}”" if s.quoted else s.text
            status = _MARK.get((s.verification or {}).get("status", ""), "")
            line = f"- {text}"
            if cites:
                line += f" ({cites})"
            if status:
                line += f" — automatic check: {status}"
            lines.append(line)
            for problem in (s.verification or {}).get("problems", []):
                lines.append(f"  - ⚠ {problem}")
        lines.append("")

    if answer.evidence:
        lines += ["## Evidence passages", ""]
        for e in answer.evidence:
            flags = [e.opinion_type]
            if e.authority != "unspecified":
                flags.append(e.authority)
            if e.treatment:
                flags.append(
                    f"later {e.treatment['treatment']} by {e.treatment['by']} "
                    f"({e.treatment['year']})"
                )
            lines += [
                f"### [{e.label}] {e.reference}",
                "",
                f"- Court: {e.court}",
                f"- Decided: {e.decision_date}",
                f"- Opinion: {'; '.join(flags)}",
                f"- Section: {e.section or 'none'}; paragraphs: {', '.join(e.para_ids)}",
                f"- Source: {e.source_url}",
                "",
                "> " + e.text.replace("\n", "\n> "),
                "",
            ]

    v = answer.verification
    if v:
        lines += [
            "## Citation verification",
            "",
            f"- Judge: {v.get('judge')} (policy: {v.get('policy')})",
            f"- Citable statements: {v.get('citable_statements')}",
            f"- Evidence citations supported: {v.get('supported_citations')} of "
            f"{v.get('evidence_citations')}",
            f"- Citations not traceable to the supplied evidence: "
            f"{v.get('hallucinated_citations')}",
            f"- Quotations not found: {v.get('quotations_not_found')}",
            "",
            v.get("note", ""),
            "",
        ]

    if answer.ranked_cases:
        lines += ["## Ranked cases", "", "| # | Case | Court | Decided | Authority | Note |",
                  "|---|---|---|---|---|---|"]  # fmt: skip
        for i, c in enumerate(answer.ranked_cases, start=1):
            note = ""
            if c.get("treatment"):
                t = c["treatment"]
                note = f"later {t['treatment']} by {t['by']} ({t['year']})"
            lines.append(
                f"| {i} | {c['reference']} | {c['court']} | {c['decision_date']} | "
                f"{c['authority']} | {note} |"
            )
        lines.append("")

    timings = ", ".join(f"{k} {v * 1000:.0f} ms" for k, v in answer.timings.items())
    lines += [
        "---",
        f"Generated {time.strftime('%Y-%m-%d %H:%M')} by legal-research-rag; "
        f"generator: {answer.generator}; {timings}.",
    ]
    return "\n".join(lines) + "\n"
