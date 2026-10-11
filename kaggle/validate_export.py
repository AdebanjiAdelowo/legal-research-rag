"""Validate a downloaded Kaggle export before importing anything from it.

    python kaggle/validate_export.py <folder holding legal-research-rag-export/ and the .zip>
    python kaggle/validate_export.py <folder> --import     # then copy results into results/

Checks, all of which must pass:

1. the export holds only ``.json``, ``.txt``, ``.log`` and ``.md`` files under 5 MB;
2. the zip archive and the downloaded files are byte-identical;
3. the commit evaluated exists in this repository, and the runner and ``uv.lock`` used
   on Kaggle hash to the files at that commit;
4. the rebuilt corpus and benchmark fingerprint equals ``kaggle/expected_fingerprint.json``;
5. every recorded step exited 0, or is listed as a failed step;
6. each result file names the same commit and corpus version;
7. no sampled benchmark or opinion text occurs in the export (skipped, and reported
   as skipped, when the local data files are absent).

With ``--import``, result JSON files are copied to ``results/`` and the provenance
files to ``results/kaggle_runs/<plan>_<commit>/`` with a checksum list. Results of a
failed step are never imported, because none exist.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = {".json", ".txt", ".log", ".md"}
MAX_MB = 5


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_show(commit: str, path: str) -> bytes | None:
    out = subprocess.run(
        ["git", "show", f"{commit}:{path}"], cwd=ROOT, capture_output=True, check=False
    )
    return out.stdout if out.returncode == 0 else None


def text_probes(n_queries: int = 300, seed: int = 1) -> list[str] | None:
    """Short word windows from local benchmark and opinion text, or None if absent."""
    processed = ROOT / "data" / "processed"
    files = [processed / "queries_test.jsonl", processed / "queries_dev.jsonl"]
    chunks = processed / "chunks_section220.jsonl"
    if not all(f.exists() for f in [*files, chunks]):
        return None
    rng = random.Random(seed)

    def windows(text: str, n: int = 9, k: int = 2) -> list[str]:
        words = text.split()
        if len(words) <= n + 2:
            return []
        return [" ".join(words[i : i + n]).lower()
                for i in (rng.randrange(0, len(words) - n) for _ in range(k))]  # fmt: skip

    probes: list[str] = []
    for f in files:
        rows = [json.loads(line) for line in f.open()]
        for row in rng.sample(rows, min(n_queries, len(rows))):
            probes += windows(row["text"]) + windows(row["raw_text"])
            probes += [q.lower() for q in row["gold_quotes"] if len(q.split()) >= 8][:1]
    with chunks.open() as f:
        for i, line in enumerate(f):
            if i % 500 == 0:
                probes += windows(json.loads(line)["text"], 10, 1)
    return [re.sub(r"\s+", " ", p) for p in probes]


def validate(folder: Path) -> tuple[list[str], list[str], dict]:
    """Return ``(problems, notes, provenance)``. No problems means the export is valid."""
    problems: list[str] = []
    notes: list[str] = []
    export = folder / "legal-research-rag-export"
    archive = folder / "legal-research-rag-export.zip"
    if not export.is_dir():
        return [f"{export} not found"], notes, {}
    files = sorted(p for p in export.rglob("*") if p.is_file())

    bad = [str(p.relative_to(export)) for p in files
           if p.suffix not in ALLOWED or p.stat().st_size > MAX_MB * 2**20]  # fmt: skip
    if bad:
        problems.append(f"unexpected files in the export: {bad}")

    if archive.exists():
        with zipfile.ZipFile(archive) as z:
            names = sorted(n for n in z.namelist() if not n.endswith("/"))
            local = sorted(str(p.relative_to(export)) for p in files)
            if names != local:
                problems.append("the archive and the downloaded files list different paths")
            elif any(sha(z.read(n)) != sha((export / n).read_bytes()) for n in names):
                problems.append("the archive and the downloaded files differ in content")
        notes.append(f"archive sha256 {sha(archive.read_bytes())}")
    else:
        notes.append("no zip archive downloaded; file-against-archive check skipped")

    prov_file = export / "provenance.json"
    if not prov_file.exists():
        return [*problems, "provenance.json missing"], notes, {}
    prov = json.loads(prov_file.read_text())
    commit = prov.get("commit", "")
    runner = git_show(commit, "kaggle/run_gpu_experiments.py")
    lock = git_show(commit, "uv.lock")
    if runner is None:
        problems.append(f"commit {commit!r} is not in this repository")
    else:
        if sha(runner) != prov.get("runner_sha256"):
            problems.append("the runner used on Kaggle differs from the one at that commit")
        if lock is not None and sha(lock) != prov.get("uv_lock_sha256"):
            problems.append("uv.lock used on Kaggle differs from the one at that commit")
        notes.append(f"commit {commit}, plan {prov.get('plan')}")

    expected = git_show(commit, "kaggle/expected_fingerprint.json") if runner else None
    fp_file = export / "fingerprint.json"
    if not fp_file.exists():
        problems.append("fingerprint.json missing")
    elif expected is not None and json.loads(fp_file.read_text()) != json.loads(expected):
        problems.append("the corpus or benchmark fingerprint differs from the expected one")
    else:
        notes.append("fingerprint matches the expected corpus and benchmark")

    failed = set(prov.get("failed_steps") or [])
    for step in prov.get("steps", []):
        if step["exit"] != 0 and step["step"] not in failed:
            problems.append(
                f"step {step['step']} exited {step['exit']} and is not recorded as failed"
            )
    if failed:
        notes.append(f"FAILED STEPS (no results expected for them): {sorted(failed)}")

    results = sorted((export / "results").rglob("*.json")) if (export / "results").exists() else []
    if not results:
        problems.append("no result files in the export")
    for path in results:
        data = json.loads(path.read_text())
        got = (data.get("git") or {}).get("commit")
        if got and got != commit:
            problems.append(f"{path.name} was produced at {got}, not {commit}")
        if (data.get("git") or {}).get("dirty_paths"):
            problems.append(f"{path.name} was produced from a dirty tree")
    notes.append(f"{len(results)} result files: {[p.name for p in results]}")

    probes = text_probes()
    if probes is None:
        notes.append("text-leak scan SKIPPED: local data/processed files are not present")
    else:
        hay = re.sub(r"\s+", " ", "\n".join(p.read_text(errors="ignore") for p in files)).lower()
        hits = [p for p in probes if p in hay]
        if hits:
            problems.append(
                f"{len(hits)} of {len(probes)} text probes occur in the export: {hits[:3]}"
            )
        else:
            notes.append(f"text-leak scan: 0 of {len(probes)} probes found")
    return problems, notes, prov


def import_results(folder: Path, prov: dict) -> list[str]:
    export = folder / "legal-research-rag-export"
    copied: list[str] = []
    for path in sorted((export / "results").rglob("*.json")):
        target = ROOT / "results" / path.relative_to(export / "results")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copied.append(str(target.relative_to(ROOT)))
    run_dir = ROOT / "results" / "kaggle_runs" / f"{prov['plan']}_{prov['commit'][:8]}"
    (run_dir / "indexes").mkdir(parents=True, exist_ok=True)
    for name in ("provenance.json", "fingerprint.json", "requirements-frozen.txt", "manifest.json"):
        if (export / name).exists():
            shutil.copy2(export / name, run_dir / name)
    for meta in sorted((export / "indexes").glob("*/meta.json")):
        shutil.copy2(meta, run_dir / "indexes" / f"{meta.parent.name}.json")
    archive = folder / "legal-research-rag-export.zip"
    sums = {
        "commit_evaluated": prov["commit"],
        "plan": prov["plan"],
        "failed_steps": prov.get("failed_steps", []),
        "export_archive_sha256": sha(archive.read_bytes()) if archive.exists() else None,
        "results": {c: sha((ROOT / c).read_bytes()) for c in copied},
        "provenance_files": {
            str(p.relative_to(run_dir)): sha(p.read_bytes())
            for p in sorted(run_dir.rglob("*")) if p.is_file() and p.name != "SHA256SUMS.json"
        },
    }  # fmt: skip
    (run_dir / "SHA256SUMS.json").write_text(json.dumps(sums, indent=2) + "\n")
    return [*copied, str(run_dir.relative_to(ROOT))]


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) != 1:
        print(__doc__.split("\n\n")[1])
        return 2
    folder = Path(args[0]).expanduser()
    problems, notes, prov = validate(folder)
    for note in notes:
        print("  " + note)
    if problems:
        print("INVALID:")
        for p in problems:
            print("  - " + p)
        return 1
    print("VALID")
    if "--import" in sys.argv:
        for path in import_results(folder, prov):
            print("  imported " + path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
