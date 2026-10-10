"""Filesystem layout. Everything under ``data/`` is rebuilt from pinned sources and is
never committed; ``results/`` holds the small JSON outputs that the README reports."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Paths:
    root: Path

    @classmethod
    def default(cls) -> Paths:
        return cls(Path(os.environ.get("LEGALRAG_HOME", REPO_ROOT)))

    @property
    def raw(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def cap_cases(self) -> Path:
        return self.raw / "cap" / "cases"

    @property
    def processed(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def indexes(self) -> Path:
        return self.root / "data" / "indexes"

    @property
    def results(self) -> Path:
        return self.root / "results"

    @property
    def corpus(self) -> Path:
        return self.processed / "corpus.jsonl"

    @property
    def manifest(self) -> Path:
        return self.processed / "manifest.json"

    def chunks(self, scheme: str) -> Path:
        return self.processed / f"chunks_{scheme}.jsonl"

    def queries(self, split: str) -> Path:
        return self.processed / f"queries_{split}.jsonl"
