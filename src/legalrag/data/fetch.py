"""Download the benchmark file and the opinions it cites.

Two sources, both pinned:

* LePaRD ``top_10000_data.csv.gz`` at a fixed Hugging Face revision, checked against
  a recorded SHA-256;
* one Caselaw Access Project JSON file per cited case.

Fetching is resumable: a case already on disk is skipped, and every miss is written
to ``fetch_log.jsonl`` with the reason, so coverage can be reported exactly.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from legalrag.data import cap

LEPARD_REPO = "rmahari/LePaRD"
LEPARD_REVISION = "0194f95c3091acceab3b887c9b09ef432cf84052"
LEPARD_FILE = "top_10000_data.csv.gz"
LEPARD_SHA256 = "498265154a547c89b3486b75dda7a6bccfd258cb4246491c246fc136891bfa99"
LEPARD_BYTES = 539_120_360

USER_AGENT = "legal-research-rag/0.1 (research; static.case.law bulk-friendly fetch)"


def lepard_url() -> str:
    return f"https://huggingface.co/datasets/{LEPARD_REPO}/resolve/{LEPARD_REVISION}/{LEPARD_FILE}"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download_lepard(raw_dir: Path) -> Path:
    """Download the benchmark file if absent and verify its checksum."""
    target = raw_dir / "lepard" / LEPARD_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        with httpx.stream("GET", lepard_url(), follow_redirects=True, timeout=120) as r:
            r.raise_for_status()
            with target.open("wb") as f:
                for block in r.iter_bytes(1 << 20):
                    f.write(block)
    digest = sha256_file(target)
    if digest != LEPARD_SHA256:
        raise ValueError(f"{target} has SHA-256 {digest}, expected {LEPARD_SHA256}")
    return target


def read_lepard(path: Path):
    """Yield benchmark rows as dicts."""
    csv.field_size_limit(1 << 30)
    with gzip.open(path, "rt", newline="") as f:
        yield from csv.DictReader(f)


def source_cases(path: Path) -> dict[str, str]:
    """Map every cited ("source") case id in the benchmark to its citation string."""
    out: dict[str, str] = {}
    for row in read_lepard(path):
        out.setdefault(row["source_id"], row["source_cite"])
    return out


def _name_words(name: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]{3,}", name.lower()) if w not in ("united", "states")}


class CapFetcher:
    """Fetch case files by id and citation, with a per-volume metadata fallback."""

    def __init__(self, out_dir: Path, client: httpx.Client | None = None, retries: int = 3):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        # Binding to an IPv4 address avoids a long stall per request on networks that
        # advertise IPv6 routes they cannot carry.
        self.client = client or httpx.Client(
            headers={"User-Agent": USER_AGENT},
            timeout=60,
            follow_redirects=True,
            transport=httpx.HTTPTransport(local_address="0.0.0.0", retries=2),
        )
        self.retries = retries
        self._volumes: dict[tuple[str, int], list[dict]] = {}

    def _get(self, url: str) -> httpx.Response | None:
        for attempt in range(self.retries):
            try:
                r = self.client.get(url)
            except httpx.HTTPError:
                time.sleep(1.5 * (attempt + 1))
                continue
            if r.status_code == 200:
                return r
            if r.status_code in (403, 404):
                return None
            time.sleep(1.5 * (attempt + 1))
        return None

    def _volume_index(self, slug: str, volume: int) -> list[dict]:
        key = (slug, volume)
        if key not in self._volumes:
            r = self._get(cap.volume_metadata_url(slug, volume))
            self._volumes[key] = r.json() if r else []
        return self._volumes[key]

    def _locate(self, slug: str, volume: int, page: int, case_id: str, cite: str):
        """File name of a case in a volume: by id, else by first page and name.

        Some benchmark ids differ from the id the project now serves for the same
        decision. Those are matched on first page plus at least half of the name
        words, and the benchmark id is kept as an alias.
        """
        entries = self._volume_index(slug, volume)
        for entry in entries:
            if str(entry["id"]) == case_id:
                return entry["file_name"], "ok"
        want = _name_words(cite.rsplit(",", 1)[0])
        for entry in entries:
            if str(entry.get("first_page")) != str(page):
                continue
            have = _name_words(entry.get("name_abbreviation", "") + " " + entry.get("name", ""))
            if want and len(want & have) / len(want) >= 0.5:
                return entry["file_name"], "ok_by_page"
        return None, "not_in_volume"

    def fetch(self, case_id: str, cite: str) -> dict[str, str]:
        """Fetch one case. Returns a log record with ``status`` ok | cached | <reason>."""
        target = self.out_dir / f"{case_id}.json"
        record = {"case_id": case_id, "cite": cite}
        if target.exists():
            return {**record, "status": "cached"}
        guess = cap.candidate_url(cite)
        if guess is None:
            return {**record, "status": "reporter_not_hosted"}
        url, slug, volume = guess
        status = "ok"
        r = self._get(url)
        if r is None or str(r.json().get("id")) != case_id:
            page = cap.cap_path(cite)[2]
            file_name, status = self._locate(slug, volume, page, case_id, cite)
            if file_name is None:
                return {**record, "status": status}
            url = cap.case_url(slug, volume, file_name)
            r = self._get(url)
            if r is None:
                return {**record, "status": "http_error"}
        target.write_text(json.dumps({"url": url, "requested_id": case_id, "case": r.json()}))
        return {**record, "status": status, "url": url}


def fetch_corpus(lepard_path: Path, out_dir: Path, workers: int = 8) -> dict[str, int]:
    """Fetch every cited case. Returns counts by status."""
    sources = source_cases(lepard_path)
    fetcher = CapFetcher(out_dir)
    counts: dict[str, int] = {}
    with (
        (out_dir.parent / "fetch_log.jsonl").open("w") as log,
        ThreadPoolExecutor(workers) as pool,
    ):
        for record in pool.map(lambda kv: fetcher.fetch(*kv), sorted(sources.items())):
            counts[record["status"]] = counts.get(record["status"], 0) + 1
            if record["status"] != "cached":
                log.write(json.dumps(record) + "\n")
    return counts
