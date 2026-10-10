"""Duplicate detection for opinions.

The same decision can enter a corpus twice: once from the official reporter and once
from a parallel one, or as an original and an amended opinion. Exact duplicates are
found by a hash of normalised text. Near duplicates are found with a 64-bit SimHash
over word 5-grams, compared within buckets so the cost stays near linear.

Near duplicates are reported, and only the exact ones are dropped by default: an
amended opinion is a different legal document from the one it replaces, and deciding
between them is a question of authority rather than of text similarity.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from collections.abc import Iterable

import numpy as np

from legalrag.data.schema import Case

_WORD = re.compile(r"[a-z0-9]+")


def normalised_text(case: Case) -> str:
    body = " ".join(p.text for p in case.paragraphs if p.opinion_index >= 0)
    return " ".join(_WORD.findall(body.lower()))


def text_hash(case: Case) -> str:
    return hashlib.sha256(normalised_text(case).encode()).hexdigest()


_WORD_HASH: dict[str, int] = {}
_MULTIPLIERS = np.array([pow(1_000_003, j, 1 << 64) for j in range(16)], dtype=np.uint64)


def _word_hash(word: str) -> int:
    h = _WORD_HASH.get(word)
    if h is None:
        h = int.from_bytes(hashlib.blake2b(word.encode(), digest_size=8).digest(), "big")
        _WORD_HASH[word] = h
    return h


def simhash(text: str, shingle: int = 5) -> int:
    """64-bit SimHash of word shingles.

    Each word is hashed once; a shingle's hash is a polynomial combination of its
    word hashes in wrapping 64-bit arithmetic, which keeps the whole computation in
    numpy.
    """
    words = text.split()
    if not words:
        return 0
    h = np.fromiter((_word_hash(w) for w in words), dtype=np.uint64, count=len(words))
    n = max(1, len(words) - shingle + 1)
    grams = np.zeros(n, dtype=np.uint64)
    for j in range(min(shingle, len(words))):
        grams += h[j : j + n] * _MULTIPLIERS[j]
    # Mix so that low bits depend on every word, then count set bits per position.
    grams ^= grams >> np.uint64(29)
    grams *= np.uint64(0xBF58476D1CE4E5B9)
    grams ^= grams >> np.uint64(32)
    bits = np.unpackbits(grams.view(np.uint8).reshape(-1, 8), axis=1, bitorder="little")
    ones = bits.sum(axis=0, dtype=np.int64)
    return sum(1 << bit for bit in range(64) if 2 * int(ones[bit]) > n)


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def find_duplicates(
    cases: Iterable[Case], max_distance: int = 3
) -> tuple[list[tuple[str, str]], list[tuple[str, str, int]]]:
    """Return ``(exact, near)`` duplicate pairs.

    ``exact`` pairs share a normalised-text hash; the first id of each pair is the
    one to keep (earliest in input order). ``near`` pairs have SimHash distance at
    most ``max_distance`` and are not exact duplicates.

    Candidates for near-duplication are found by splitting the hash into four
    16-bit bands: two hashes within distance 3 must agree on at least one band.
    """
    by_hash: dict[str, str] = {}
    exact: list[tuple[str, str]] = []
    hashes: dict[str, int] = {}
    for case in cases:
        text = normalised_text(case)
        digest = hashlib.sha256(text.encode()).hexdigest()
        if digest in by_hash:
            exact.append((by_hash[digest], case.case_id))
            continue
        by_hash[digest] = case.case_id
        hashes[case.case_id] = simhash(text)

    bands: dict[tuple[int, int], list[str]] = defaultdict(list)
    for case_id, h in hashes.items():
        for band in range(4):
            bands[(band, (h >> (16 * band)) & 0xFFFF)].append(case_id)
    near: dict[tuple[str, str], int] = {}
    for ids in bands.values():
        if len(ids) < 2 or len(ids) > 200:
            continue
        for i, a in enumerate(ids):
            for b in ids[i + 1 :]:
                d = hamming(hashes[a], hashes[b])
                if d <= max_distance:
                    near[(a, b) if a < b else (b, a)] = d
    return exact, [(a, b, d) for (a, b), d in sorted(near.items())]
