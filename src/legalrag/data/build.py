"""Corpus and benchmark build: raw files in, versioned processed files out.

Steps: parse every fetched opinion, drop exact duplicates, chunk under each scheme,
sample queries from LePaRD, locate gold paragraphs, and write a manifest holding
counts, filter statistics and a corpus version.

The corpus version is a hash over the text hashes of the cases kept. Index files
record the version they were built from and refuse to load against another, so an
index can never silently be paired with a corpus it does not describe.
"""

from __future__ import annotations

import hashlib
import json
import statistics
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from legalrag.data import benchmark, cap, fetch
from legalrag.data.chunking import chunk_cases
from legalrag.data.dedup import find_duplicates
from legalrag.data.schema import Case, Chunk, Query
from legalrag.paths import Paths
from legalrag.provenance import read_jsonl, write_jsonl

# name -> (scheme, size in words, extra arguments). The first two are built with the
# corpus. ``fixed170`` is a length-matched control for the chunking comparison: its
# mean passage length equals that of ``section220``. It is built on first use.
CHUNK_SCHEMES: dict[str, tuple[str, int, dict]] = {
    "section220": ("section", 220, {}),
    "fixed220": ("fixed", 220, {}),
    "fixed170": ("fixed", 170, {"overlap": 30}),
}
BUILT_WITH_CORPUS = ("section220", "fixed220")
QUERY_SIZES = {"dev": 800, "test": 2000, "history": 800}
MULTI_SIZES = {"dev": 300, "test": 600}


def load_corpus(paths: Paths) -> dict[str, Case]:
    return {d["case_id"]: Case.from_json(d) for d in read_jsonl(paths.corpus)}


def write_chunks(paths: Paths, name: str, cases: dict[str, Case]) -> dict:
    """Chunk every case under one scheme, write the file, return length statistics."""
    scheme, size, extra = CHUNK_SCHEMES[name]
    lengths: list[int] = []

    def rows():
        for chunk in chunk_cases(cases.values(), scheme, size=size, **extra):
            chunk.scheme = name
            chunk.chunk_id = chunk.chunk_id.replace(f"#{scheme}{size}-", f"#{name}-")
            lengths.append(len(chunk.text.split()))
            yield chunk.to_json()

    n = write_jsonl(paths.chunks(name), rows())
    return {
        "chunks": n,
        "words_mean": round(statistics.mean(lengths), 1),
        "words_median": statistics.median(lengths),
        "words_p95": sorted(lengths)[int(0.95 * n)],
    }


def load_chunks(paths: Paths, scheme: str, cases: dict[str, Case] | None = None) -> list[Chunk]:
    """Chunks of one scheme. A known scheme that was not built with the corpus is
    generated from the parsed corpus on first use."""
    if not paths.chunks(scheme).exists() and scheme in CHUNK_SCHEMES:
        write_chunks(paths, scheme, cases or load_corpus(paths))
    return [Chunk(**d) for d in read_jsonl(paths.chunks(scheme))]


def load_queries(paths: Paths, split: str) -> list[Query]:
    return [Query.from_json(d) for d in read_jsonl(paths.queries(split))]


def load_manifest(paths: Paths) -> dict:
    return json.loads(paths.manifest.read_text())


def corpus_version(cases: dict[str, Case]) -> str:
    h = hashlib.sha256()
    for case_id in sorted(cases):
        h.update(case_id.encode())
        h.update(cases[case_id].text_sha256.encode())
    return h.hexdigest()[:12]


def parse_corpus(raw_dir: Path) -> tuple[dict[str, Case], dict[str, str], dict]:
    """Parse fetched files. Returns ``(cases, aliases, report)`` where ``aliases``
    maps a benchmark case id to the corpus case id for the same decision."""
    cases: dict[str, Case] = {}
    aliases: dict[str, str] = {}
    empty = 0
    for raw, url, requested in cap.load_raw_cases(raw_dir):
        case = cap.parse_case(raw, url)
        if not any(p.opinion_index >= 0 and not p.is_heading for p in case.paragraphs):
            empty += 1
            continue
        cases.setdefault(case.case_id, case)
        aliases[f"cap:{requested}"] = case.case_id
    exact, near = find_duplicates(cases.values())
    for keep, drop in exact:
        cases.pop(drop, None)
        aliases = {k: (keep if v == drop else v) for k, v in aliases.items()}
    report = {
        "parsed": len(cases) + len(exact),
        "benchmark_ids_remapped": sum(k != v for k, v in aliases.items()),
        "empty_opinion_text": empty,
        "exact_duplicates_dropped": [list(p) for p in exact],
        "near_duplicates_reported": [list(p) for p in near],
    }
    return cases, aliases, report


def _describe(cases: dict[str, Case]) -> dict:
    words = [c.word_count for c in cases.values()]
    years = [c.year for c in cases.values()]
    return {
        "cases": len(cases),
        "paragraphs": sum(len(c.paragraphs) for c in cases.values()),
        "words_total": sum(words),
        "words_median": statistics.median(words),
        "words_p90": sorted(words)[int(0.9 * len(words))],
        "year_min": min(years),
        "year_max": max(years),
        "by_court_level": dict(Counter(c.court_level for c in cases.values())),
        "by_circuit": dict(Counter(c.circuit or "none" for c in cases.values())),
        "with_section_headings": sum(
            any(p.is_heading for p in c.paragraphs) for c in cases.values()
        ),
    }


def build(paths: Paths, lepard_path: Path | None = None) -> dict:
    """Run the whole build and return the manifest."""
    lepard_path = lepard_path or paths.raw / "lepard" / fetch.LEPARD_FILE
    cases, aliases, dedup_report = parse_corpus(paths.cap_cases)
    version = corpus_version(cases)
    write_jsonl(paths.corpus, (c.to_json() for c in cases.values()))

    chunk_stats = {name: write_chunks(paths, name, cases) for name in BUILT_WITH_CORPUS}

    candidates, quotes, multi, stats = benchmark.collect(
        fetch.read_lepard(lepard_path), cases, aliases
    )
    queries, gold = benchmark.build_queries(
        candidates, quotes, multi, cases, QUERY_SIZES, MULTI_SIZES
    )
    query_stats: dict[str, dict] = {}
    for split, items in queries.items():
        write_jsonl(paths.queries(split), (q.to_json() for q in items))
        lengths = [len(q.text.split()) for q in items]
        query_stats[split] = {
            "queries": len(items),
            "with_gold_paragraph": sum(bool(q.gold_paras) for q in items),
            "words_median": statistics.median(lengths) if lengths else 0,
            "date_min": min((q.date for q in items), default=""),
            "date_max": max((q.date for q in items), default=""),
            "gold_scotus_share": round(
                sum(cases[q.gold_cases[0]].court_level == "scotus" for q in items)
                / max(1, len(items)),
                3,
            ),
        }
    located = sum(bool(v) for v in gold.values())

    manifest = {
        "corpus_version": version,
        "sources": {
            "lepard": {
                "repo": fetch.LEPARD_REPO,
                "revision": fetch.LEPARD_REVISION,
                "file": fetch.LEPARD_FILE,
                "sha256": fetch.LEPARD_SHA256,
                "license": "CC BY-NC-SA 4.0",
            },
            "caselaw_access_project": {"base_url": cap.BASE_URL, "license": "CC0 1.0"},
        },
        "corpus": {**_describe(cases), **dedup_report},
        "chunks": chunk_stats,
        "benchmark_filters": asdict(stats),
        "passages_seen": len(gold),
        "passages_located_in_source": located,
        "queries": query_stats,
        "query_sizes_requested": {"single": QUERY_SIZES, "multi": MULTI_SIZES},
    }
    paths.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
