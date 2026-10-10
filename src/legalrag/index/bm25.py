"""Okapi BM25 over a sparse term-frequency matrix.

The index stores raw term frequencies, so ``k1`` and ``b`` can be changed after the
index is built: tuning on development queries costs one re-weighting pass, not a
rebuild. Scoring a query is a sum of a few sparse columns.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from scipy import sparse

_TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")

STOPWORDS = frozenset(
    [
        "a",
        "about",
        "above",
        "after",
        "again",
        "against",
        "all",
        "am",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "because",
        "been",
        "before",
        "being",
        "below",
        "between",
        "both",
        "but",
        "by",
        "can",
        "did",
        "do",
        "does",
        "doing",
        "down",
        "during",
        "each",
        "few",
        "for",
        "from",
        "further",
        "had",
        "has",
        "have",
        "having",
        "he",
        "her",
        "here",
        "hers",
        "him",
        "his",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "itself",
        "just",
        "me",
        "more",
        "most",
        "my",
        "no",
        "nor",
        "not",
        "now",
        "of",
        "off",
        "on",
        "once",
        "only",
        "or",
        "other",
        "our",
        "out",
        "over",
        "own",
        "s",
        "same",
        "she",
        "should",
        "so",
        "some",
        "such",
        "t",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "to",
        "too",
        "under",
        "until",
        "up",
        "very",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "whom",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
    ]
)


def stem(token: str) -> str:
    """Harman's S-stemmer: conflates plural and singular forms only.

    A full Porter stemmer merges legally distinct terms ("statute"/"statutory",
    "liable"/"liability" are fine, but "probation"/"probate" or "executor"/
    "execution" are not), so only the plural rule is applied.
    """
    if len(token) > 4 and token.endswith("ies") and not token.endswith(("eies", "aies")):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("es") and not token.endswith(("aes", "ees", "oes")):
        return token[:-1]
    if len(token) > 3 and token.endswith("s") and not token.endswith(("us", "ss")):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    out = []
    for tok in _TOKEN.findall(text.lower()):
        tok = tok.split("'")[0]
        if tok and tok not in STOPWORDS:
            out.append(stem(tok))
    return out


class BM25Index:
    def __init__(
        self,
        tf: sparse.csc_matrix,
        vocab: dict[str, int],
        doc_len: np.ndarray,
        k1: float = 1.2,
        b: float = 0.75,
    ):
        self.tf = tf
        self.vocab = vocab
        self.doc_len = doc_len.astype(np.float32)
        self.n_docs = tf.shape[0]
        self.weights: sparse.csc_matrix
        self.set_params(k1, b)

    @classmethod
    def build(cls, texts: Iterable[str], k1: float = 1.2, b: float = 0.75) -> BM25Index:
        vocab: dict[str, int] = {}
        rows: list[np.ndarray] = []
        cols: list[np.ndarray] = []
        vals: list[np.ndarray] = []
        lengths: list[int] = []
        for i, text in enumerate(texts):
            counts = Counter(tokenize(text))
            lengths.append(sum(counts.values()))
            if not counts:
                continue
            ids = np.fromiter(
                (vocab.setdefault(t, len(vocab)) for t in counts), dtype=np.int32, count=len(counts)
            )
            cols.append(ids)
            rows.append(np.full(len(counts), i, dtype=np.int32))
            vals.append(np.fromiter(counts.values(), dtype=np.float32, count=len(counts)))
        n = len(lengths)
        tf = sparse.csc_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(n, len(vocab)),
            dtype=np.float32,
        )
        return cls(tf, vocab, np.asarray(lengths), k1, b)

    def set_params(self, k1: float, b: float) -> None:
        """Recompute BM25 weights for new ``k1`` and ``b``."""
        self.k1, self.b = k1, b
        tf = self.tf
        df = np.diff(tf.indptr).astype(np.float32)
        idf = np.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5))
        norm = k1 * (1.0 - b + b * self.doc_len / max(1e-9, self.doc_len.mean()))
        col_of = np.repeat(np.arange(tf.shape[1]), np.diff(tf.indptr))
        data = idf[col_of] * tf.data * (k1 + 1.0) / (tf.data + norm[tf.indices])
        self.weights = sparse.csc_matrix(
            (data.astype(np.float32), tf.indices, tf.indptr), shape=tf.shape
        )

    def score(self, query: str) -> np.ndarray:
        """BM25 score of every document for ``query``."""
        counts = Counter(t for t in tokenize(query) if t in self.vocab)
        scores = np.zeros(self.n_docs, dtype=np.float32)
        if not counts:
            return scores
        w = self.weights
        for term, qtf in counts.items():
            j = self.vocab[term]
            lo, hi = w.indptr[j], w.indptr[j + 1]
            scores[w.indices[lo:hi]] += w.data[lo:hi] * qtf
        return scores

    def score_batch(self, queries: list[str]) -> np.ndarray:
        return np.stack([self.score(q) for q in queries])

    def save(self, directory: Path, meta: dict) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        sparse.save_npz(directory / "tf.npz", self.tf)
        np.save(directory / "doc_len.npy", self.doc_len)
        (directory / "vocab.json").write_text(json.dumps(self.vocab))
        (directory / "meta.json").write_text(
            json.dumps({**meta, "kind": "bm25", "k1": self.k1, "b": self.b}, indent=2)
        )

    @classmethod
    def load(cls, directory: Path, k1: float | None = None, b: float | None = None) -> BM25Index:
        meta = json.loads((directory / "meta.json").read_text())
        return cls(
            sparse.load_npz(directory / "tf.npz").tocsc(),
            json.loads((directory / "vocab.json").read_text()),
            np.load(directory / "doc_len.npy"),
            k1 if k1 is not None else meta["k1"],
            b if b is not None else meta["b"],
        )
