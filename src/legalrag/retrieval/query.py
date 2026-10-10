"""Reading jurisdiction and date constraints out of a natural-language question.

"Qualified immunity standard in the Ninth Circuit before 2009" carries two
constraints that plain retrieval ignores: which court's law is wanted and the date
the question is asked as of. This module extracts them with explicit rules, so the
interpretation can be shown to the user and corrected, and so it is testable.

It also detects questions about law outside the corpus (state courts, other
countries). Retrieval would still return the nearest federal cases for such a
question, which is exactly the failure a legal research tool must not make silently.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from legalrag.data.courts import CIRCUITS, classify_court

_ORDINAL_WORDS = {
    "first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5", "sixth": "6",
    "seventh": "7", "eighth": "8", "ninth": "9", "tenth": "10", "eleventh": "11",
}  # fmt: skip
_CIRCUIT_RE = re.compile(
    r"\b(?:(?P<word>first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh)"
    r"|(?P<num>1[01]|[1-9])(?:st|nd|rd|th|d)|(?P<dc>d\.?\s?c\.?|district of columbia)"
    r"|(?P<fed>fed(?:eral|\.)))\s+cir(?:cuit|\.)",
    re.IGNORECASE,
)
_ORD = "first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh"
_PAIR_RE = re.compile(rf"\b({_ORD})\s+(?:and|or)\s+({_ORD})\s+circuits?\b", re.IGNORECASE)
_SCOTUS_RE = re.compile(r"\b(?:u\.?s\.? )?supreme court\b|\bscotus\b", re.IGNORECASE)
_OWN_DECISIONS = re.compile(
    r"\b(?:decisions?|cases?|opinions?|rulings?|precedents?)\s+(?:of|from|by)\b|"
    r"\b(?:decided|held|written|issued)\s+by\b|"
    r"\bcir(?:cuit|\.)\s+(?:decisions?|cases?|opinions?|rulings?|precedents?|"
    r"(?:has |have )?(?:held|hold|decided|ruled))\b",
    re.IGNORECASE,
)
_BINDING_RE = re.compile(r"\b(?:binding|controlling|mandatory)\b", re.IGNORECASE)
_DISTRICT_RE = re.compile(
    r"\b(?:(?:northern|southern|eastern|western|middle|central)\s+)?district of "
    r"(?!columbia circuit)([A-Z][a-z]+(?: [A-Z][a-z]+)?)"
)

_YEAR = r"((?:17|18|19|20)\d{2})"
_DATE_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(rf"\bbetween {_YEAR} and {_YEAR}\b", re.I), "between"),
    (re.compile(rf"\bfrom {_YEAR} (?:to|through|until) {_YEAR}\b", re.I), "between"),
    (re.compile(rf"\b(?:before|prior to|pre-|earlier than|until) ?{_YEAR}\b", re.I), "before"),
    (re.compile(rf"\bas of {_YEAR}\b", re.I), "as_of"),
    (re.compile(rf"\b(?:after|since|post-|later than|from) ?{_YEAR}\b", re.I), "after"),
    (re.compile(r"\bin the ((?:17|18|19|20)\d)0s\b", re.I), "decade"),
    (re.compile(rf"\b(?:decided )?in {_YEAR}\b", re.I), "in"),
]

# Sources of law the corpus does not contain. Each pattern maps to a short reason.
_OUT_OF_SCOPE: list[tuple[re.Pattern, str]] = [
    (
        re.compile(
            r"\b(?:england|english law|united kingdom|u\.k\.|uk law|scotland|ireland|"
            r"canad(?:a|ian)|australia(?:n)?|india(?:n law)?|european union|eu law|"
            r"gdpr|german(?:y)?|french law|france|ital(?:y|ian)|nigeria(?:n)?|"
            r"south africa(?:n)?|china|chinese law|japan(?:ese)?|singapore|"
            r"court of justice of the european union|cjeu|echr|house of lords)\b",
            re.I,
        ),
        "the question concerns law outside the United States",
    ),
    (
        re.compile(
            r"\b(?:(?:alabama|alaska|arizona|arkansas|california|colorado|connecticut|"
            r"delaware|florida|georgia|hawaii|idaho|illinois|indiana|iowa|kansas|"
            r"kentucky|louisiana|maine|maryland|massachusetts|michigan|minnesota|"
            r"mississippi|missouri|montana|nebraska|nevada|new hampshire|new jersey|"
            r"new mexico|new york|north carolina|north dakota|ohio|oklahoma|oregon|"
            r"pennsylvania|rhode island|south carolina|south dakota|tennessee|texas|"
            r"utah|vermont|virginia|washington|west virginia|wisconsin|wyoming)"
            r"(?:'s)? (?:supreme court|court of appeals?|state law|law|statute|"
            r"civil code|penal code|appellate (?:court|division))|state[- ]law|"
            r"state court)\b",
            re.I,
        ),
        "the question concerns state law; the corpus holds federal decisions only",
    ),
]


@dataclass
class QueryPlan:
    """How a question was read. ``text`` is what retrieval is run on."""

    original: str
    text: str
    before: str | None = None  # keep cases decided before this ISO date
    after: str | None = None
    circuits: list[str] = field(default_factory=list)
    levels: list[str] = field(default_factory=list)
    binding_on: str | None = None  # circuit whose binding authority is requested
    forum_circuit: str | None = None
    out_of_scope: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def filters(self) -> dict:
        """Keyword arguments for ``ChunkStore.mask``."""
        out: dict = {}
        if self.before:
            out["before"] = self.before
        if self.after:
            out["after"] = self.after
        if self.binding_on:
            out["binding_on_circuit"] = self.binding_on
        else:
            if self.circuits:
                out["circuits"] = tuple(self.circuits)
            if self.levels:
                out["levels"] = tuple(self.levels)
        return out


def _circuit(m: re.Match) -> str:
    if m.group("word"):
        return _ORDINAL_WORDS[m.group("word").lower()]
    if m.group("num"):
        return m.group("num")
    return "DC" if m.group("dc") else "Fed"


def interpret(question: str) -> QueryPlan:
    """Extract jurisdiction and date constraints from ``question``."""
    original = question
    # "Second and Ninth Circuits" names two courts with one head noun.
    question = _PAIR_RE.sub(lambda m: f"{m.group(1)} Circuit and {m.group(2)} Circuit", question)
    plan = QueryPlan(original=original, text=question)
    spans: list[tuple[int, int]] = []

    for pattern, reason in _OUT_OF_SCOPE:
        if pattern.search(question):
            plan.out_of_scope.append(reason)

    circuits: dict[str, None] = {}
    for m in _CIRCUIT_RE.finditer(question):
        circuits.setdefault(_circuit(m))
        spans.append(m.span())
    for m in _DISTRICT_RE.finditer(question):
        court = classify_court("district court " + m.group(0))
        if court.circuit:
            plan.forum_circuit = court.circuit
            plan.notes.append(f"{m.group(0)} lies in circuit {court.circuit}")
    scotus = bool(_SCOTUS_RE.search(question))
    plan.circuits = [c for c in circuits if c in CIRCUITS]

    if _BINDING_RE.search(question) and (plan.circuits or plan.forum_circuit):
        plan.binding_on = plan.circuits[0] if plan.circuits else plan.forum_circuit
        plan.forum_circuit = plan.binding_on
        plan.notes.append(
            f"restricted to authority binding in circuit {plan.binding_on}: "
            "Supreme Court and that circuit's court of appeals"
        )
    elif len(plan.circuits) == 1 and not scotus and not _OWN_DECISIONS.search(question):
        # "the standard in the Ninth Circuit" asks for the law that applies there,
        # which includes Supreme Court decisions, not only that court's own.
        plan.binding_on = plan.forum_circuit = plan.circuits[0]
        plan.notes.append(
            f"read as the law applicable in circuit {plan.binding_on}: Supreme Court "
            "and that circuit's court of appeals"
        )
    elif plan.circuits:
        plan.levels = ["appeals"]
        plan.forum_circuit = plan.circuits[0]
        if scotus:
            plan.binding_on = plan.circuits[0] if len(plan.circuits) == 1 else None
            if plan.binding_on is None:
                plan.levels, plan.circuits = [], []
                plan.notes.append("several circuits and the Supreme Court named; no court filter")
        if plan.circuits and not plan.binding_on:
            plan.notes.append("court filter: decisions of circuit " + ", ".join(plan.circuits))
    elif scotus and re.search(r"\b(?:only|decisions? of|held by|according to)\b", question, re.I):
        plan.levels = ["scotus"]
        plan.notes.append("court filter: Supreme Court")

    for pattern, kind in _DATE_RULES:
        m = pattern.search(question)
        if not m:
            continue
        if kind == "between":
            lo, hi = sorted((int(m.group(1)), int(m.group(2))))
            plan.after, plan.before = f"{lo}-01-01", f"{hi + 1}-01-01"
        elif kind == "before":
            plan.before = f"{m.group(1)}-01-01"
        elif kind == "as_of":
            plan.before = f"{int(m.group(1)) + 1}-01-01"
        elif kind == "after":
            plan.after = f"{m.group(1)}-01-01"
        elif kind == "decade":
            decade = int(m.group(1)) * 10
            plan.after, plan.before = f"{decade}-01-01", f"{decade + 10}-01-01"
        elif kind == "in":
            year = int(m.group(1))
            plan.after, plan.before = f"{year}-01-01", f"{year + 1}-01-01"
        spans.append(m.span())
        plan.notes.append(
            "date filter: "
            + " and ".join(
                s for s in (
                    f"decided on or after {plan.after}" if plan.after else "",
                    f"decided before {plan.before}" if plan.before else "",
                ) if s
            )
        )  # fmt: skip
        break

    # Drop the constraint phrases from the retrieval text: "Ninth Circuit before
    # 2009" would otherwise pull in every opinion that mentions that court.
    text = question
    for start, end in sorted(spans, reverse=True):
        lead = re.search(r"(?:\b(?:in|of|by|from|within|on|for)\s+)?(?:the\s+)?$", text[:start])
        text = text[: lead.start() if lead else start] + " " + text[end:]
    text = re.sub(r"\s*[,;]\s*(?=[?.!])", "", text)
    text = re.sub(r"\s+(?=[?.!](?:\s|$))", "", text)
    plan.text = re.sub(r"\s{2,}", " ", text).strip(" ,;") or question
    return plan
