"""Two ways of cutting an opinion into retrieval units.

``fixed``
    A sliding window of ``size`` words with ``overlap`` words of overlap over the
    opinion text, blind to paragraph, section and opinion boundaries. This is the
    usual default and serves as the baseline.

``section``
    Paragraphs are packed greedily up to ``size`` words, but a chunk never crosses a
    section heading, never mixes the majority opinion with a concurrence or dissent,
    and never splits a paragraph unless that paragraph alone exceeds ``max_words``.
    The heading path is kept as metadata. Groups under ``min_words`` words (author
    lines, bare dispositions) are not emitted.

Both schemes record the ids of the paragraphs each chunk covers, so provenance and
paragraph-level evaluation work identically for either.

The two differ in mean chunk length for the same ``size`` setting, because the
section scheme closes chunks early at boundaries. Comparisons therefore report the
number of words placed in front of the reader at each cutoff, not only rank metrics.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator

from legalrag.data.schema import HEAD_MATTER, Case, Chunk, Paragraph

_SENTENCE_END = re.compile(r"(?:(?<=[.!?])|(?<=[.!?][\"”’)]))\s+(?=[A-Z“\"(\[])")
# Abbreviations after which a capital letter does not start a new sentence.
_NO_BREAK = re.compile(
    r"(?:\b(?:v|vs|Inc|Co|Corp|Ltd|No|Nos|Cir|Ct|Dist|Supp|Stat|Rev|Rep|Mr|Mrs|Ms|Dr|"
    r"Jr|Sr|St|U\.S|U\. S|F|Ed|App|Cf|cf|See|see|Id|id|Ibid|ibid|e\.g|i\.e|et al|"
    r"[A-Z])\.)$"
)


def split_sentences(text: str) -> list[str]:
    """Sentence splitter that does not break inside legal citations.

    ``Roe v. Wade, 410 U.S. 113`` contains three full stops that are not sentence
    ends. Candidate boundaries after a known abbreviation or a single capital
    letter are rejected.
    """
    out: list[str] = []
    start = 0
    for m in _SENTENCE_END.finditer(text):
        head = text[start : m.start()].rstrip('"”’)')
        if _NO_BREAK.search(head):
            continue
        out.append(text[start : m.start()].strip())
        start = m.end()
    tail = text[start:].strip()
    if tail:
        out.append(tail)
    return [s for s in out if s]


def _split_long(text: str, max_words: int) -> list[str]:
    """Split an over-long paragraph at sentence boundaries into pieces of at most
    ``max_words`` words (a single longer sentence is cut on whitespace)."""
    pieces: list[str] = []
    current: list[str] = []
    count = 0
    for sentence in split_sentences(text):
        words = sentence.split()
        while len(words) > max_words:
            if current:
                pieces.append(" ".join(current))
                current, count = [], 0
            pieces.append(" ".join(words[:max_words]))
            words = words[max_words:]
        if count + len(words) > max_words and current:
            pieces.append(" ".join(current))
            current, count = [], 0
        current.append(" ".join(words))
        count += len(words)
    if current:
        pieces.append(" ".join(current))
    return pieces


def _body(case: Case, include_head_matter: bool) -> list[Paragraph]:
    return [p for p in case.paragraphs if include_head_matter or p.opinion_type != HEAD_MATTER]


def section_chunks(
    case: Case,
    size: int = 220,
    max_words: int = 330,
    include_head_matter: bool = False,
    min_words: int = 12,
) -> Iterator[Chunk]:
    """Section-aware chunks. See the module docstring."""
    scheme = f"section{size}"
    n = 0
    group: list[Paragraph] = []
    words = 0

    def flush() -> Iterator[Chunk]:
        nonlocal n, group, words
        body = [p for p in group if not p.is_heading]
        # Author lines ("Mr. Justice Blackmun delivered the opinion of the Court.")
        # and one-word dispositions carry nothing retrievable and would otherwise
        # rank well for being short.
        if body and sum(len(p.text.split()) for p in body) >= min_words:
            first = group[0]
            yield Chunk(
                chunk_id=f"{case.case_id}#{scheme}-{n}",
                case_id=case.case_id,
                text="\n".join(p.text for p in group),
                para_ids=[p.para_id for p in body],
                opinion_type=first.opinion_type,
                section=body[0].section,
                scheme=scheme,
            )
            n += 1
        group, words = [], 0

    previous: Paragraph | None = None
    for para in _body(case, include_head_matter):
        boundary = previous is not None and (
            para.opinion_index != previous.opinion_index
            or para.section != previous.section
            or para.is_heading
        )
        length = len(para.text.split())
        if boundary or (group and words + length > size):
            yield from flush()
        if length > max_words:
            yield from flush()
            for piece in _split_long(para.text, size):
                yield Chunk(
                    chunk_id=f"{case.case_id}#{scheme}-{n}",
                    case_id=case.case_id,
                    text=piece,
                    para_ids=[para.para_id],
                    opinion_type=para.opinion_type,
                    section=para.section,
                    scheme=scheme,
                )
                n += 1
        else:
            group.append(para)
            words += length
        previous = para
    yield from flush()


def fixed_chunks(
    case: Case, size: int = 220, overlap: int = 40, include_head_matter: bool = False
) -> Iterator[Chunk]:
    """Fixed-size word windows over the whole document."""
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    scheme = f"fixed{size}"
    tokens: list[str] = []
    owner: list[Paragraph] = []
    for para in _body(case, include_head_matter):
        for word in para.text.split():
            tokens.append(word)
            owner.append(para)
    step = size - overlap
    n = 0
    for start in range(0, len(tokens), step):
        end = min(start + size, len(tokens))
        owners = owner[start:end]
        seen: dict[str, None] = {}
        for p in owners:
            if not p.is_heading:
                seen.setdefault(p.para_id)
        if seen:
            yield Chunk(
                chunk_id=f"{case.case_id}#{scheme}-{n}",
                case_id=case.case_id,
                text=" ".join(tokens[start:end]),
                para_ids=list(seen),
                opinion_type=owners[0].opinion_type,
                section=owners[0].section,
                scheme=scheme,
            )
            n += 1
        if end == len(tokens):
            break


def chunk_cases(cases: Iterable[Case], scheme: str, size: int = 220, **kw) -> Iterator[Chunk]:
    """Chunk every case with ``scheme`` in {"section", "fixed"}."""
    fn = {"section": section_chunks, "fixed": fixed_chunks}[scheme]
    for case in cases:
        yield from fn(case, size=size, **kw)
