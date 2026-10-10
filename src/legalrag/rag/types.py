"""Data model of an answer.

An answer is a list of typed statements, not free prose. The type records what kind
of thing is being asserted, because those kinds carry different weight in legal
research and must not blur into each other:

``fact``            what happened in a cited case, as that opinion states it
``reasoning``       a court's rule, holding or explanation, as that opinion states it
``provision``       a statute, rule or constitutional text an opinion quotes or names
``interpretation``  the system's own synthesis; never presented as authority
``not_established`` something the question asks that the evidence does not answer

Only the first three may carry citations, and they must: a statement of those types
without one is counted as unsupported.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

FACT = "fact"
REASONING = "reasoning"
PROVISION = "provision"
INTERPRETATION = "interpretation"
NOT_ESTABLISHED = "not_established"

CITABLE = (FACT, REASONING, PROVISION)
STATEMENT_TYPES = (*CITABLE, INTERPRETATION, NOT_ESTABLISHED)

LIMITATION = (
    "Research aid only. This is not legal advice and does not replace the judgment of "
    "a qualified lawyer. Coverage is limited to the indexed U.S. federal opinions; "
    "relevant authority may be missing, and cited cases may no longer be good law."
)


@dataclass
class Evidence:
    """One passage placed in front of the generator, with its provenance."""

    label: str  # "E1", "E2", ...
    chunk_id: str
    case_id: str
    case_name: str
    citation: str  # official reporter citation
    reference: str  # "Roe v. Wade, 410 U.S. 113 (1973)"
    court: str
    court_level: str
    circuit: str | None
    decision_date: str
    opinion_type: str  # majority | concurrence | dissent | ...
    section: str
    para_ids: list[str]
    text: str
    score: float
    authority: str = "unspecified"  # binding | persuasive | unspecified
    treatment: dict | None = None  # known later negative treatment, if any
    source_url: str = ""

    @property
    def is_majority(self) -> bool:
        return self.opinion_type in ("majority", "unanimous", "per curiam", "per_curiam")


@dataclass
class Statement:
    type: str
    text: str
    citations: list[str] = field(default_factory=list)  # evidence labels
    quoted: bool = False  # True when the text is a verbatim extract
    verification: dict | None = None


@dataclass
class Answer:
    question: str
    abstained: bool
    abstain_reason: str = ""
    statements: list[Statement] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    plan: dict = field(default_factory=dict)  # how the question was interpreted
    ranked_cases: list[dict] = field(default_factory=list)
    verification: dict = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)
    generator: str = ""
    raw_output: str = ""
    limitation: str = LIMITATION
    warnings: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    def cited_labels(self) -> set[str]:
        return {c for s in self.statements for c in s.citations}
