"""Chunk store and versioned index directories.

An index lives at ``data/indexes/<corpus version>/<chunk scheme>/<index name>/`` and
its ``meta.json`` repeats the corpus version, scheme and chunk count. Loading checks
all three against the chunk store in use.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from legalrag.data.build import load_chunks, load_corpus, load_manifest
from legalrag.data.schema import Case, Chunk
from legalrag.index.bm25 import BM25Index
from legalrag.index.dense import ENCODERS, DenseIndex
from legalrag.paths import Paths


def _date_int(date: str) -> int:
    """ISO date (possibly year or year-month only) as yyyymmdd."""
    parts = ([*date.split("-"), "01", "01"])[:3]
    try:
        return int(parts[0]) * 10000 + int(parts[1]) * 100 + int(parts[2])
    except ValueError:
        return 0


@dataclass
class ChunkStore:
    """Chunks of one scheme with the case metadata needed for filtering."""

    scheme: str
    corpus_version: str
    chunks: list[Chunk]
    cases: dict[str, Case]
    case_ids: list[str]  # index -> case id
    chunk_case: np.ndarray  # chunk index -> case index
    case_date: np.ndarray  # yyyymmdd per case
    case_level: np.ndarray  # court level per case
    case_circuit: np.ndarray  # circuit per case, "" when none

    @classmethod
    def load(cls, paths: Paths, scheme: str) -> ChunkStore:
        return cls.from_parts(
            scheme, load_manifest(paths)["corpus_version"], load_chunks(paths, scheme),
            load_corpus(paths),
        )  # fmt: skip

    @classmethod
    def from_parts(
        cls, scheme: str, version: str, chunks: list[Chunk], cases: dict[str, Case]
    ) -> ChunkStore:
        case_ids = sorted(cases)
        pos = {c: i for i, c in enumerate(case_ids)}
        return cls(
            scheme=scheme,
            corpus_version=version,
            chunks=chunks,
            cases=cases,
            case_ids=case_ids,
            chunk_case=np.array([pos[c.case_id] for c in chunks], dtype=np.int32),
            case_date=np.array([_date_int(cases[c].decision_date) for c in case_ids]),
            case_level=np.array([cases[c].court_level for c in case_ids]),
            case_circuit=np.array([cases[c].circuit or "" for c in case_ids]),
        )

    def __len__(self) -> int:
        return len(self.chunks)

    @property
    def texts(self) -> list[str]:
        return [c.text for c in self.chunks]

    def case_of(self, chunk_index: int) -> Case:
        return self.cases[self.case_ids[self.chunk_case[chunk_index]]]

    def mask(
        self,
        before: str | None = None,
        after: str | None = None,
        levels: tuple[str, ...] | None = None,
        circuits: tuple[str, ...] | None = None,
        binding_on_circuit: str | None = None,
        exclude_cases: tuple[str, ...] = (),
    ) -> np.ndarray | None:
        """Boolean mask over chunks, or None when no filter is active.

        ``before`` keeps cases decided strictly before the date: the cases that
        existed when a question dated ``before`` arose. ``binding_on_circuit`` keeps
        Supreme Court decisions and decisions of that circuit's court of appeals.
        """
        keep = np.ones(len(self.case_ids), dtype=bool)
        active = False
        if before:
            keep &= self.case_date < _date_int(before)
            active = True
        if after:
            keep &= self.case_date >= _date_int(after)
            active = True
        if levels:
            keep &= np.isin(self.case_level, levels)
            active = True
        if circuits:
            keep &= np.isin(self.case_circuit, circuits)
            active = True
        if binding_on_circuit:
            keep &= (self.case_level == "scotus") | (
                (self.case_level == "appeals") & (self.case_circuit == binding_on_circuit)
            )
            active = True
        if exclude_cases:
            pos = {c: i for i, c in enumerate(self.case_ids)}
            for case_id in exclude_cases:
                if case_id in pos:
                    keep[pos[case_id]] = False
            active = True
        return keep[self.chunk_case] if active else None


def index_dir(paths: Paths, store: ChunkStore, name: str) -> Path:
    return paths.indexes / store.corpus_version / store.scheme / name


def _meta(store: ChunkStore) -> dict:
    return {"corpus_version": store.corpus_version, "scheme": store.scheme, "chunks": len(store)}


def _check(directory: Path, store: ChunkStore) -> None:
    meta = json.loads((directory / "meta.json").read_text())
    for key, value in _meta(store).items():
        if meta.get(key) != value:
            raise ValueError(
                f"index at {directory} was built for {key}={meta.get(key)!r}, "
                f"but the chunk store has {key}={value!r}; rebuild the index"
            )


def bm25_index(paths: Paths, store: ChunkStore, rebuild: bool = False) -> BM25Index:
    directory = index_dir(paths, store, "bm25")
    if (directory / "meta.json").exists() and not rebuild:
        _check(directory, store)
        return BM25Index.load(directory)
    index = BM25Index.build(c.text for c in store.chunks)
    index.save(directory, _meta(store))
    return index


def dense_index(
    paths: Paths, store: ChunkStore, encoder_key: str, rebuild: bool = False, batch_size: int = 64,
    max_tokens: int | None = None,
) -> DenseIndex:  # fmt: skip
    directory = index_dir(paths, store, f"dense-{encoder_key}")
    if (directory / "meta.json").exists() and not rebuild:
        _check(directory, store)
        return DenseIndex.load(directory)
    work = directory / "parts"
    index, stats = DenseIndex.build(
        store.texts, ENCODERS[encoder_key], batch_size, max_tokens=max_tokens, work_dir=work
    )
    index.save(directory, {**_meta(store), "build": stats})
    for part in work.glob("part-*.npy"):
        part.unlink()
    if work.exists():
        work.rmdir()
    return index
