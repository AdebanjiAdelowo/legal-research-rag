"""Record types shared across ingestion, indexing and evaluation.

Identifiers are stable across rebuilds:

* a case is ``cap:<id>`` where ``<id>`` is the Caselaw Access Project case id;
* a paragraph is ``<case_id>:<opinion index>:<paragraph index>``, counted from the
  parsed opinion text, so a paragraph id pins a specific span of a specific opinion;
* a chunk is ``<case_id>#<scheme>-<ordinal>`` and always carries the paragraph ids
  it covers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

HEAD_MATTER = "head_matter"


@dataclass
class Paragraph:
    para_id: str
    text: str
    opinion_index: int  # -1 for head matter (syllabus, parties, counsel)
    opinion_type: str  # majority | concurrence | dissent | ... | head_matter
    section: str  # heading path such as "II/A"; "" before the first heading
    is_heading: bool = False


@dataclass
class Case:
    case_id: str
    name: str
    short_name: str
    decision_date: str  # ISO date; may be a year or year-month for old cases
    court: str
    court_level: str
    circuit: str | None
    jurisdiction: str
    citations: list[str]  # official first, then parallel
    docket_number: str
    paragraphs: list[Paragraph]
    cites_to: list[dict[str, Any]] = field(default_factory=list)
    authors: list[str] = field(default_factory=list)
    source_url: str = ""
    text_sha256: str = ""
    simhash: str = ""
    word_count: int = 0

    @property
    def official_cite(self) -> str:
        return self.citations[0] if self.citations else ""

    @property
    def year(self) -> int:
        return int(self.decision_date[:4])

    def bluebook(self) -> str:
        """Short reference: ``Roe v. Wade, 410 U.S. 113 (1973)``."""
        return f"{self.short_name}, {self.official_cite} ({self.year})"

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> Case:
        d = dict(d)
        d["paragraphs"] = [Paragraph(**p) for p in d["paragraphs"]]
        return cls(**d)


@dataclass
class Chunk:
    chunk_id: str
    case_id: str
    text: str
    para_ids: list[str]
    opinion_type: str
    section: str
    scheme: str  # chunking scheme that produced it

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Query:
    query_id: str
    text: str  # masked text used for retrieval
    raw_text: str  # text before citation masking; kept for the leakage control
    date: str  # decision date of the citing opinion: the "as of" date of the query
    court: str  # forum the query arises in
    court_level: str
    circuit: str | None
    gold_cases: list[str]
    gold_paras: list[str]  # paragraph ids holding the quoted passage, may be empty
    gold_quotes: list[str]
    split: str
    kind: str = "single"  # single | multi
    # For a multi-issue query, the separate contexts it was assembled from. Used only
    # as an oracle upper bound for query decomposition, never by a deployed system.
    parts: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> Query:
        return cls(**d)
