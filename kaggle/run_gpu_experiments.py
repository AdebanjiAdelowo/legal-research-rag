"""Kaggle GPU runner: the experiments that are too heavy for a laptop.

Runs inside a Kaggle notebook with a GPU and internet enabled. It installs the
package from an attached dataset holding this repository's source, downloads the
pinned benchmark file and the cited opinions, builds the corpus, and then runs:

1. dense indexes for every encoder, including the legal and general ModernBERT pair;
2. the retrieval comparison (BM25, dense, hybrid; legal against general encoder);
3. reranking with a larger cross-encoder over the best hybrid;
4. judge validation with the large entailment model;
5. grounded generation with an open instruction-tuned model, with and without
   strict verification, and with the relevant case withheld.

Results are written to /kaggle/working/results and zipped. Copy the JSON files into
``results/`` of the repository to include them in the report tables.

Before the heavy steps the script prints free disk and GPU memory and stops if the
estimates in ``kaggle/README.md`` cannot be met.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

SOURCE = Path(os.environ.get("LEGALRAG_SOURCE", "/kaggle/input/legal-research-rag-src"))
WORK = Path(os.environ.get("LEGALRAG_HOME", "/kaggle/working/legal-research-rag"))
GENERATOR = os.environ.get("LEGALRAG_GENERATOR", "hf:Qwen/Qwen2.5-7B-Instruct")
ENCODERS = ["bge-small", "modernbert-general", "modernbert-legal"]
MIN_FREE_DISK_GB = 25
MIN_GPU_GB = 14


def sh(*args: str, check: bool = True) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run(args, check=check, cwd=WORK, env={**os.environ, "LEGALRAG_HOME": str(WORK)})


def preflight() -> None:
    free = shutil.disk_usage("/kaggle/working").free / 1e9
    print(f"free disk: {free:.1f} GB")
    if free < MIN_FREE_DISK_GB:
        sys.exit(f"need {MIN_FREE_DISK_GB} GB of free disk")
    import torch

    if not torch.cuda.is_available():
        sys.exit("no GPU: enable an accelerator in the notebook settings")
    gpu = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {torch.cuda.get_device_name(0)}, {gpu:.1f} GB")
    if gpu < MIN_GPU_GB:
        sys.exit(f"need a GPU with at least {MIN_GPU_GB} GB")


def main() -> None:
    if WORK.exists():
        shutil.rmtree(WORK)
    shutil.copytree(SOURCE, WORK)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-e", f"{WORK}[ml]"], check=True
    )
    preflight()
    run = [sys.executable, "-m", "legalrag.cli"]
    os.environ["LEGALRAG_MAX_GB"] = "24"

    sh(*run, "fetch", "--workers", "8")
    sh(*run, "build")
    sh(*run, "index", "bm25")
    for encoder in ENCODERS:
        sh(*run, "index", encoder, "--batch-size", "128")

    sh(*run, "experiment", "retrieval", "encoders=" + ",".join(ENCODERS) + ",")
    sh(*run, "experiment", "rerank", "reranker=bge-reranker-v2-m3", "encoder=modernbert-legal")
    sh(*run, "experiment", "rerank", "reranker=minilm-l6", "encoder=modernbert-legal")
    sh(*run, "experiment", "judge", "judges=lexical,nli-xsmall,nli-base,nli-large")
    sh(*run, "experiment", "citations", "judge=nli-large")
    sh(*run, "experiment", "abstention", "reranker=bge-reranker-v2-m3")
    sh(
        *run, "experiment", "rag", f"generator={GENERATOR}", "limit=300", "eval_judge=nli-large",
        "filter_judge=nli-base", "reranker=bge-reranker-v2-m3", "tag=open-model-gpu",
    )  # fmt: skip
    sh(*run, "experiment", "systems", "encoder=modernbert-legal", "reranker=bge-reranker-v2-m3")

    out = Path("/kaggle/working/results")
    if out.exists():
        shutil.rmtree(out)
    # Per-answer files quote benchmark text and stay out of the archive.
    shutil.copytree(WORK / "results", out, ignore=shutil.ignore_patterns("answers_*", "logs"))
    shutil.make_archive("/kaggle/working/legal-research-rag-results", "zip", out)
    print("done: /kaggle/working/legal-research-rag-results.zip")


if __name__ == "__main__":
    main()
