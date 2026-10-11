"""Stage the files Kaggle needs, locally. Uploads nothing.

    python kaggle/prepare.py --username <kaggle username> --out /path/outside/the/repo

Writes two folders under ``--out``:

``source/``   a private Kaggle dataset: ``legal-research-rag.bundle`` (a Git bundle of
              the current commit and its history), ``COMMIT`` (its hash) and
              ``dataset-metadata.json``
``kernel/``   the runner at that commit and ``kernel-metadata.json``

The bundle holds exactly what Git tracks: code, documentation and aggregate result
files. It contains no dataset, no opinion text and no model weights.

The script refuses to run on a dirty working tree, because the point is that Kaggle
evaluates one identifiable commit. It then prints the two ``kaggle`` commands to run;
it does not run them.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = "legal-research-rag.bundle"


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def stage(out: Path, username: str, allow_dirty: bool = False) -> dict:
    dirty = git("status", "--porcelain")
    if dirty and not allow_dirty:
        raise SystemExit(f"commit or stash these changes first:\n{dirty}")
    commit = git("rev-parse", "HEAD")
    if out.resolve().is_relative_to(ROOT):
        raise SystemExit("--out must be outside the repository")
    source, kernel = out / "source", out / "kernel"
    for folder in (source, kernel):
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True)

    subprocess.run(
        ["git", "bundle", "create", str(source / BUNDLE), "HEAD"], cwd=ROOT, check=True,
        capture_output=True,
    )  # fmt: skip
    (source / "COMMIT").write_text(commit + "\n")
    (source / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "title": "legal-research-rag-src",
                "id": f"{username}/legal-research-rag-src",
                "licenses": [{"name": "MIT"}],
            },
            indent=2,
        )
        + "\n"
    )

    # The runner is taken from the commit, not from the working tree.
    runner = subprocess.run(
        ["git", "show", f"{commit}:kaggle/run_gpu_experiments.py"], cwd=ROOT, check=True,
        capture_output=True,
    ).stdout  # fmt: skip
    (kernel / "run_gpu_experiments.py").write_bytes(runner)
    (kernel / "kernel-metadata.json").write_text(
        json.dumps(
            {
                "id": f"{username}/legal-research-rag-gpu",
                "title": "legal-research-rag-gpu",
                "code_file": "run_gpu_experiments.py",
                "language": "python",
                "kernel_type": "script",
                "is_private": True,
                "enable_gpu": True,
                "enable_internet": True,
                "dataset_sources": [f"{username}/legal-research-rag-src"],
                "competition_sources": [],
                "kernel_sources": [],
            },
            indent=2,
        )
        + "\n"
    )
    return {"commit": commit, "source": source, "kernel": kernel}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--username", required=True, help="Kaggle username")
    parser.add_argument("--out", required=True, type=Path, help="staging folder outside the repo")
    args = parser.parse_args()
    staged = stage(args.out, args.username)
    size = (staged["source"] / BUNDLE).stat().st_size / 1e6
    print(f"commit   {staged['commit']}")
    print(f"bundle   {staged['source'] / BUNDLE} ({size:.1f} MB)")
    print(f"kernel   {staged['kernel']}")
    print("\nNothing has been uploaded. To run on Kaggle:")
    print(f"  kaggle datasets create -p {staged['source']}        # first time (private)")
    print(f"  kaggle datasets version -p {staged['source']} -m {staged['commit'][:8]}   # later")
    print(f"  kaggle kernels push -p {staged['kernel']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
