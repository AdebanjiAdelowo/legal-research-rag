"""Splitting a multi-issue question into sub-queries.

A research question often joins several legal issues ("whether the search was
lawful, and if not, whether the officers have qualified immunity"). A single query
vector or bag of words blends them, and the top results can all address one issue.
Decomposition retrieves for each issue separately and fuses the lists.

Three splitters are provided, none of which needs a language model:

``segments``
    Sentences are grouped into contiguous segments at points of low lexical overlap
    between neighbours (a small TextTiling). This is the default.

``sentences``
    One sub-query per sentence, short sentences merged into their neighbour.

``enumeration``
    Splits on explicit markers: "(1) ... (2) ...", "first ... second ...",
    semicolons, "and whether".

The full question is always kept as one of the sub-queries, so a single-issue
question is never made worse by a bad split alone; fusion can still dilute it, which
the decomposition experiment measures.
"""

from __future__ import annotations

import re

from legalrag.data.chunking import split_sentences
from legalrag.index.bm25 import tokenize

_ENUM = re.compile(
    r"\s*(?:\(\d\)|\([a-d]\)|;|\b(?:first|second|third|fourth|finally|secondly|thirdly),"
    r"|,?\s+and (?=whether\b)|,?\s+and if (?:so|not),?)\s*",
    re.IGNORECASE,
)


def by_sentences(text: str, min_words: int = 8, max_parts: int = 4) -> list[str]:
    parts: list[str] = []
    for sentence in split_sentences(text):
        if (parts and len(sentence.split()) < min_words) or (
            parts and len(parts[-1].split()) < min_words
        ):
            parts[-1] += " " + sentence
        else:
            parts.append(sentence)
    return _merge_to(parts, max_parts)


def by_enumeration(text: str, min_words: int = 4, max_parts: int = 4) -> list[str]:
    parts = [p.strip() for p in _ENUM.split(text) if p and len(p.split()) >= min_words]
    return _merge_to(parts, max_parts)


def by_segments(text: str, target_words: int = 25, max_parts: int = 4) -> list[str]:
    """Group sentences into segments, cutting where adjacent sentences share least.

    The number of segments is ``round(words / target_words)``, at least one and at
    most ``max_parts``; boundaries are placed at the lowest Jaccard similarities
    between neighbouring sentences' content words.
    """
    sentences = split_sentences(text)
    n_parts = max(1, min(max_parts, round(len(text.split()) / target_words)))
    if len(sentences) <= 1 or n_parts == 1:
        return [text.strip()]
    bags = [set(tokenize(s)) for s in sentences]
    gaps = []
    for i in range(len(sentences) - 1):
        union = bags[i] | bags[i + 1]
        gaps.append((len(bags[i] & bags[i + 1]) / len(union) if union else 0.0, i))
    cuts = sorted(i for _, i in sorted(gaps)[: n_parts - 1])
    parts, start = [], 0
    for cut in cuts:
        parts.append(" ".join(sentences[start : cut + 1]))
        start = cut + 1
    parts.append(" ".join(sentences[start:]))
    return [p for p in parts if p]


def _merge_to(parts: list[str], max_parts: int) -> list[str]:
    parts = [p for p in parts if p]
    while len(parts) > max_parts:
        # Merge the shortest part into its shorter neighbour.
        i = min(range(len(parts)), key=lambda j: len(parts[j].split()))
        j = (
            i - 1
            if i == len(parts) - 1 or (i > 0 and len(parts[i - 1]) <= len(parts[i + 1]))
            else i
        )
        parts[j : j + 2] = [parts[j] + " " + parts[j + 1]]
    return parts


SPLITTERS = {"segments": by_segments, "sentences": by_sentences, "enumeration": by_enumeration}


def decompose(text: str, method: str = "segments", keep_original: bool = True) -> list[str]:
    """Sub-queries for ``text``. With one detected issue the result is ``[text]``."""
    parts = SPLITTERS[method](text)
    if len(parts) <= 1:
        return [text]
    return ([text] if keep_original else []) + parts
