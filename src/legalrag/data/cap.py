"""Caselaw Access Project: fetching and parsing opinions.

The project (Harvard Law School Library Innovation Lab) publishes U.S. case law as
static JSON at https://static.case.law under CC0. Each case file carries the court,
decision date, parallel citations, the opinions with their type and author, and a
``cites_to`` list in which outgoing citations are already resolved to case ids.

Opinion text is OCR of the printed reporter: paragraphs are newline-separated and
footnotes are appended after the body. Parsing keeps one record per paragraph and
never merges or re-flows them, so a paragraph id always refers to a span a reader
can find in the source.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from legalrag.data.citations import cap_path
from legalrag.data.courts import classify_court
from legalrag.data.schema import HEAD_MATTER, Case, Paragraph

BASE_URL = "https://static.case.law"

# A heading is a short line that is only an outline marker ("II", "A.", "1.") or an
# outline marker followed by a short capitalised title ("III. DISCUSSION").
_ROMAN = r"(?:[IVX]{1,6})"
_HEADING_RE = re.compile(
    rf"^(?:(?P<roman>{_ROMAN})|(?P<alpha>[A-Z])|(?P<num>\d{{1,2}}))\.?"
    r"(?:\s+(?P<title>[A-Z][A-Za-z ,'’&/-]{2,70}))?$"
)
_NAMED_HEADINGS = {
    "background", "discussion", "conclusion", "facts", "analysis", "introduction",
    "standard of review", "jurisdiction", "procedural history", "opinion",
    "factual background", "procedural background", "statement of the case",
}  # fmt: skip


def case_url(slug: str, volume: int, file_name: str) -> str:
    return f"{BASE_URL}/{slug}/{volume}/cases/{file_name}.json"


def volume_metadata_url(slug: str, volume: int) -> str:
    return f"{BASE_URL}/{slug}/{volume}/CasesMetadata.json"


def candidate_url(cite: str) -> tuple[str, str, int] | None:
    """First guess at a case URL from its citation, with the slug and volume.

    Files are named ``<first page, zero-padded to 4>-<ordinal>``; the ordinal
    distinguishes cases that start on the same page, so the guess can be wrong and
    callers must check the returned case id.
    """
    parsed = cap_path(cite)
    if parsed is None:
        return None
    slug, volume, page = parsed
    return case_url(slug, volume, f"{page:04d}-01"), slug, volume


def heading_level(line: str) -> tuple[int, str] | None:
    """``(level, label)`` if ``line`` is a section heading, else None.

    Levels: 1 for roman numerals and named headings, 2 for capital letters, 3 for
    arabic numerals. A single capital "I" is read as roman one.
    """
    s = line.strip()
    if not s or len(s) > 80:
        return None
    if s.lower().rstrip(".:") in _NAMED_HEADINGS:
        return 1, s.rstrip(".:").title()
    m = _HEADING_RE.match(s)
    if not m:
        return None
    title = m.group("title")
    if title and not (title.isupper() or title.istitle()):
        return None  # ordinary sentence that happens to start with "A " or "I "
    if m.group("roman"):
        return 1, m.group("roman")
    if m.group("alpha"):
        return 2, m.group("alpha")
    return 3, m.group("num")


def _paragraphs(case_id: str, index: int, kind: str, text: str) -> list[Paragraph]:
    out: list[Paragraph] = []
    path: dict[int, str] = {}
    for n, raw in enumerate(line for line in text.split("\n") if line.strip()):
        line = raw.strip()
        heading = heading_level(line) if kind != HEAD_MATTER else None
        if heading:
            level, label = heading
            path = {k: v for k, v in path.items() if k < level}
            path[level] = label
        section = "/".join(path[k] for k in sorted(path))
        tag = "h" if index < 0 else str(index)
        out.append(
            Paragraph(
                para_id=f"{case_id}:{tag}:{n}",
                text=line,
                opinion_index=index,
                opinion_type=kind,
                section=section,
                is_heading=heading is not None,
            )
        )
    return out


def parse_case(raw: dict[str, Any], source_url: str = "") -> Case:
    """Convert one Caselaw Access Project case file into a ``Case``."""
    case_id = f"cap:{raw['id']}"
    body = raw.get("casebody") or {}
    paragraphs: list[Paragraph] = []
    if body.get("head_matter"):
        paragraphs += _paragraphs(case_id, -1, HEAD_MATTER, body["head_matter"])
    authors: list[str] = []
    for i, op in enumerate(body.get("opinions") or []):
        kind = (op.get("type") or "majority").lower()
        if op.get("author"):
            authors.append(op["author"].strip().rstrip(",."))
        paragraphs += _paragraphs(case_id, i, kind, op.get("text") or "")

    cites = raw.get("citations") or []
    ordered = [c["cite"] for c in cites if c.get("type") == "official"]
    ordered += [c["cite"] for c in cites if c.get("type") == "parallel"]
    court_name = (raw.get("court") or {}).get("name", "")
    court = classify_court(court_name)
    text = "\n".join(p.text for p in paragraphs)
    analysis = raw.get("analysis") or {}
    return Case(
        case_id=case_id,
        name=raw.get("name", ""),
        short_name=raw.get("name_abbreviation") or raw.get("name", ""),
        decision_date=raw.get("decision_date", ""),
        court=court_name,
        court_level=court.level,
        circuit=court.circuit,
        jurisdiction=(raw.get("jurisdiction") or {}).get("name_long", ""),
        citations=ordered,
        docket_number=raw.get("docket_number", "") or "",
        paragraphs=paragraphs,
        cites_to=[
            {"cite": c.get("cite", ""), "case_ids": c.get("case_ids", [])}
            for c in raw.get("cites_to") or []
        ],
        authors=authors,
        source_url=source_url,
        text_sha256=hashlib.sha256(text.encode()).hexdigest(),
        simhash=analysis.get("simhash", ""),
        word_count=len(text.split()),
    )


def load_raw_cases(directory: Path):
    """Yield ``(raw dict, source url, requested id)`` for every fetched case file.

    The requested id is the benchmark's id for the case; it usually equals the id in
    the file, and differs where the project has re-issued the id.
    """
    for path in sorted(directory.glob("*.json")):
        with path.open() as f:
            payload = json.load(f)
        yield payload["case"], payload.get("url", ""), payload.get("requested_id", path.stem)
