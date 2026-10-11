"""Kaggle GPU runner.

Default plan (``LEGALRAG_PLAN=encoders``): the controlled comparison of a legal-domain
embedding model with its general-purpose parent (RQ2).

What the script guarantees, and stops if it cannot:

* **Exact commit.** The code is checked out at one commit, taken from a Git bundle in
  an attached private dataset or from a private GitHub repository through Kaggle
  Secrets. The checked-out commit must equal the requested one and be clean. This
  file then hands over to the copy of itself inside that checkout, so the code that
  runs the experiments is the committed one whatever was pushed as the kernel.
* **Locked dependencies.** Packages are installed from ``uv.lock`` into a fresh
  environment; the resolved versions are exported with the results.
* **Same corpus and benchmark as the local results.** After the build, hashes of the
  passages, the development queries and the test queries are compared with
  ``kaggle/expected_fingerprint.json``. Any difference aborts the run.
* **Nothing dropped silently.** Every passage is encoded; the index build fails if the
  embedding count differs from the passage count, and it records how many passages
  exceed the sequence limit.
* **Identical conditions.** Both encoders use one batch size, sequence limit and
  precision. If the GPU runs out of memory, both are rebuilt at a smaller batch size.
* **No restricted text in the export.** Work happens under ``/tmp``. Only result
  files, index metadata, the build manifest, logs and provenance are copied to
  ``/kaggle/working``, and the export is scanned before it is zipped.

Environment variables (all optional):

``LEGALRAG_PLAN``           ``encoders`` (default) or ``full``
``LEGALRAG_SOURCE``         ``bundle`` (default) or ``github``
``LEGALRAG_SOURCE_DATASET`` dataset folder holding ``legal-research-rag.bundle`` and
                            ``COMMIT`` (default: found under ``/kaggle/input``)
``LEGALRAG_COMMIT``         commit to evaluate (default: the bundle's ``COMMIT`` file)
``LEGALRAG_GITHUB_REPO``    ``owner/name`` for the ``github`` source
``LEGALRAG_BATCH_SIZE``     first batch size to try (default 128)
``LEGALRAG_GENERATOR``      generator for the ``full`` plan

Run ``python kaggle/run_gpu_experiments.py --dry-run`` anywhere to print the plan
without executing it.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

INPUT = Path("/kaggle/input")
WORK = Path(os.environ.get("LEGALRAG_WORK", "/tmp/legalrag"))
REPO = WORK / "repo"
VENV = WORK / "venv"
EXPORT = Path(os.environ.get("LEGALRAG_EXPORT", "/kaggle/working/legal-research-rag-export"))
BUNDLE = "legal-research-rag.bundle"

GENERAL, LEGAL = "modernbert-general", "modernbert-legal"
BM25 = ("k1=1.6", "b=0.75")  # tuned on development queries; see results/retrieval
BATCH_SIZES = (128, 64, 32, 16)
MIN_FREE_DISK_GB = 20
MIN_GPU_GB = 12
# Files that may leave the machine. Anything else under the export root aborts the run.
EXPORT_SUFFIXES = {".json", ".txt", ".log", ".md"}
MAX_EXPORT_FILE_MB = 5

STEPS: list[dict] = []


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def sh(args: list[str], cwd: Path | None = None, env: dict | None = None, name: str = "",
       check: bool = True, quiet: bool = False) -> int:  # fmt: skip
    """Run a command, tee its output to a step log, and record its duration."""
    if not quiet:
        log("+ " + " ".join(args))
    start = time.time()
    log_file = WORK / "logs" / f"{len(STEPS):02d}-{name or Path(args[0]).name}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w") as out:
        proc = subprocess.Popen(
            args, cwd=cwd, env={**os.environ, **(env or {})}, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True,
        )  # fmt: skip
        for line in proc.stdout:
            out.write(line)
            if not quiet:
                print(line, end="", flush=True)
        code = proc.wait()
    STEPS.append({"step": name or args[0], "seconds": round(time.time() - start, 1), "exit": code})
    if check and code != 0:
        raise SystemExit(f"step {name or args[0]} failed with exit code {code}; see {log_file}")
    return code


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def secret(name: str) -> str | None:
    """A Kaggle Secret, or None outside Kaggle or when it is not attached."""
    try:
        from kaggle_secrets import UserSecretsClient

        return UserSecretsClient().get_secret(name)
    except Exception:
        return os.environ.get(name)


# -- source ------------------------------------------------------------------


def find_bundle() -> Path:
    override = os.environ.get("LEGALRAG_SOURCE_DATASET")
    roots = [Path(override)] if override else sorted(INPUT.glob("*")) + sorted(INPUT.glob("*/*"))
    for root in roots:
        if (root / BUNDLE).exists():
            return root
    raise SystemExit(
        f"no {BUNDLE} under {INPUT}; attach the source dataset made by kaggle/prepare.py"
    )


def checkout_from_bundle() -> str:
    root = find_bundle()
    commit = os.environ.get("LEGALRAG_COMMIT") or (root / "COMMIT").read_text().strip()
    sh(["git", "clone", "--quiet", str(root / BUNDLE), str(REPO)], name="clone")
    sh(["git", "checkout", "--quiet", "--detach", commit], cwd=REPO, name="checkout")
    return commit


def checkout_from_github() -> str:
    repo, commit = os.environ.get("LEGALRAG_GITHUB_REPO"), os.environ.get("LEGALRAG_COMMIT")
    token = secret("GITHUB_TOKEN")
    if not (repo and commit and token):
        raise SystemExit(
            "the github source needs LEGALRAG_GITHUB_REPO, LEGALRAG_COMMIT and a Kaggle "
            "Secret named GITHUB_TOKEN with read access to that repository"
        )
    # The token is passed in the URL to git only, never printed, and the remote is
    # removed once the commit has been fetched.
    url = f"https://x-access-token:{token}@github.com/{repo}.git"
    REPO.mkdir(parents=True, exist_ok=True)
    sh(["git", "init", "--quiet"], cwd=REPO, name="git-init")
    sh(["git", "fetch", "--quiet", "--depth", "1", url, commit], cwd=REPO, name="git-fetch",
       quiet=True)  # fmt: skip
    sh(["git", "checkout", "--quiet", "--detach", "FETCH_HEAD"], cwd=REPO, name="checkout")
    return commit


def verify_checkout(commit: str) -> str:
    """The working tree is the requested commit and is clean. Returns the full hash."""
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.strip()
    if head != commit and not head.startswith(commit):
        raise SystemExit(f"checked out {head}, expected {commit}")
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.strip()
    if dirty:
        raise SystemExit(f"checkout is not clean:\n{dirty}")
    return head


# -- environment ---------------------------------------------------------------


def preflight() -> dict:
    free = shutil.disk_usage(WORK).free / 1e9
    info: dict = {"free_disk_gb": round(free, 1), "memory_limit_gb": memory_limit_gb()}
    if free < MIN_FREE_DISK_GB:
        raise SystemExit(f"{free:.1f} GB free under {WORK}; need {MIN_FREE_DISK_GB}")
    try:
        query = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, check=False,
        )  # fmt: skip
        visible = query.returncode == 0 and bool(query.stdout.strip())
    except OSError:
        visible = False
    if not visible:
        raise SystemExit("no GPU visible: enable an accelerator in the notebook settings")
    name, memory, driver = (x.strip() for x in query.stdout.strip().splitlines()[0].split(","))
    gpu_gb = float(memory.split()[0]) / 1024
    info.update({"gpu": name, "gpu_memory_gb": round(gpu_gb, 1), "driver": driver,
                 "gpu_count": len(query.stdout.strip().splitlines())})  # fmt: skip
    if gpu_gb < MIN_GPU_GB:
        raise SystemExit(f"GPU has {gpu_gb:.1f} GB; need at least {MIN_GPU_GB}")
    log(f"preflight: {info}")
    return info


def install() -> dict:
    """Install the locked dependencies into a fresh environment."""
    sh([sys.executable, "-m", "pip", "install", "--quiet", "uv"], name="install-uv")
    env = {"UV_PROJECT_ENVIRONMENT": str(VENV), "UV_LINK_MODE": "copy"}
    sh([sys.executable, "-m", "uv", "sync", "--frozen", "--no-dev", "--extra", "ml"], cwd=REPO,
       env=env, name="uv-sync")  # fmt: skip
    freeze = subprocess.run(
        [sys.executable, "-m", "uv", "pip", "freeze", "--python", str(VENV / "bin" / "python")],
        capture_output=True, text=True, check=True,
    ).stdout  # fmt: skip
    cuda = subprocess.run(
        [str(VENV / "bin" / "python"), "-c",
         "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()  # fmt: skip
    if not cuda.endswith("True"):
        raise SystemExit(f"the locked torch build cannot see the GPU: {cuda}")
    return {"freeze": freeze, "torch": cuda, "uv_lock_sha256": sha256(REPO / "uv.lock")}


def memory_limit_gb() -> int:
    """Abort threshold for a step: 85% of the machine's RAM."""
    try:
        with open("/proc/meminfo") as f:
            total_kb = int(next(line for line in f if line.startswith("MemTotal")).split()[1])
        return max(4, int(total_kb / 1e6 * 0.85))
    except (OSError, StopIteration, ValueError):
        return 10


def legalrag(*args: str, name: str, check: bool = True) -> int:
    env = {"LEGALRAG_HOME": str(WORK / "home"), "LEGALRAG_MAX_GB": str(memory_limit_gb()),
           "HF_HOME": str(WORK / "hf"), "TOKENIZERS_PARALLELISM": "false"}  # fmt: skip
    return sh([str(VENV / "bin" / "legalrag"), *args], cwd=REPO, env=env, name=name, check=check)


# -- data ----------------------------------------------------------------------


def prepare_data() -> str:
    """Build the corpus and benchmark, or copy a prebuilt private dataset, then check
    that it is the one the local results were computed on."""
    home = WORK / "home"
    prebuilt = next(iter(sorted(INPUT.glob("*/processed/manifest.json"))), None)
    if prebuilt is not None:
        log(f"using prebuilt processed data from {prebuilt.parent}")
        shutil.copytree(prebuilt.parent, home / "data" / "processed", dirs_exist_ok=True)
        source = "attached private dataset"
    else:
        legalrag("fetch", "--workers", "8", name="fetch")
        legalrag("build", name="build")
        source = "rebuilt from the pinned LePaRD revision and the Caselaw Access Project"
    legalrag("fingerprint", "--check", "kaggle/expected_fingerprint.json",
             "--write", str(WORK / "fingerprint.json"), name="fingerprint")  # fmt: skip
    return source


# -- plans ---------------------------------------------------------------------


def index_encoders(first_batch: int) -> int:
    """Encode the corpus with both models at one batch size, halving it for both if
    either runs out of GPU memory. Returns the batch size used."""
    sizes = [b for b in BATCH_SIZES if b <= first_batch] or [first_batch]
    for batch in sizes:
        failed = False
        for key in (GENERAL, LEGAL):
            code = legalrag("index", key, "--batch-size", str(batch), "--max-tokens", "512",
                            name=f"index-{key}-b{batch}", check=False)  # fmt: skip
            if code != 0:
                failed = True
                break
        if not failed:
            return batch
        log(f"encoding failed at batch size {batch}; rebuilding both encoders smaller")
        for directory in (WORK / "home" / "data" / "indexes").glob("*/*/dense-modernbert-*"):
            shutil.rmtree(directory)
    raise SystemExit("encoding failed at every batch size; see the index logs")


def plan_encoders() -> None:
    legalrag("index", "bm25", name="index-bm25")
    batch = index_encoders(int(os.environ.get("LEGALRAG_BATCH_SIZE", BATCH_SIZES[0])))
    log(f"both encoders indexed at batch size {batch}")
    legalrag("experiment", "encoders", f"general={GENERAL}", f"legal={LEGAL}", *BM25,
             name="experiment-encoders")  # fmt: skip


def plan_full() -> None:
    plan_encoders()
    generator = os.environ.get("LEGALRAG_GENERATOR", "hf:Qwen/Qwen2.5-7B-Instruct")
    legalrag("index", "bge-small", "--batch-size", "128", name="index-bge-small")
    legalrag("experiment", "retrieval", f"encoders=bge-small,{GENERAL},{LEGAL}", name="retrieval")
    for reranker in ("bge-reranker-v2-m3", "minilm-l6"):
        legalrag("experiment", "rerank", f"reranker={reranker}", f"encoder={LEGAL}",
                 "fusion=score", "alpha=0.7", *BM25, name=f"rerank-{reranker}")  # fmt: skip
    legalrag("experiment", "judge", "judges=lexical,nli-xsmall,nli-base,nli-large", name="judge")
    legalrag("experiment", "citations", "judge=nli-large", "limit=150", *BM25, name="citations")
    legalrag("experiment", "abstention", "reranker=bge-reranker-v2-m3", *BM25, name="abstention")
    legalrag("experiment", "rag", f"generator={generator}", "limit=300", "eval_judge=nli-large",
             "filter_judge=nli-xsmall", *BM25, "tag=open-model-gpu", name="rag")  # fmt: skip
    legalrag("experiment", "systems", f"encoder={LEGAL}", "reranker=bge-reranker-v2-m3",
             name="systems")  # fmt: skip


PLANS = {"encoders": plan_encoders, "full": plan_full}


# -- export --------------------------------------------------------------------


def export(provenance: dict) -> Path:
    """Copy results, index metadata, the manifest, logs and provenance, and nothing else."""
    home = WORK / "home"
    if EXPORT.exists():
        shutil.rmtree(EXPORT)
    (EXPORT / "results").mkdir(parents=True)
    for path in (home / "results").rglob("*.json"):
        if path.name.startswith(("answers_", "per_query")):
            continue
        target = EXPORT / "results" / path.relative_to(home / "results")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    for meta in (home / "data" / "indexes").glob("*/*/*/meta.json"):
        target = EXPORT / "indexes" / meta.parent.name / "meta.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(meta, target)
    for name in ("manifest.json",):
        source = home / "data" / "processed" / name
        if source.exists():
            shutil.copy2(source, EXPORT / name)
    if (WORK / "fingerprint.json").exists():
        shutil.copy2(WORK / "fingerprint.json", EXPORT / "fingerprint.json")
    shutil.copytree(WORK / "logs", EXPORT / "logs")
    (EXPORT / "requirements-frozen.txt").write_text(provenance.pop("freeze", ""))
    (EXPORT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")

    offending = [
        str(p.relative_to(EXPORT))
        for p in EXPORT.rglob("*")
        if p.is_file()
        and (p.suffix not in EXPORT_SUFFIXES or p.stat().st_size > MAX_EXPORT_FILE_MB * 2**20)
    ]
    if offending:
        shutil.rmtree(EXPORT)
        raise SystemExit(f"refusing to export unexpected files: {offending}")
    archive = shutil.make_archive(str(EXPORT), "zip", EXPORT)
    return Path(archive)


def dry_run(plan: str) -> None:
    print(f"plan: {plan}")
    print(f"source: {os.environ.get('LEGALRAG_SOURCE', 'bundle')}")
    print(f"work directory: {WORK} (not exported)")
    print(f"export: {EXPORT}.zip  suffixes {sorted(EXPORT_SUFFIXES)}, each under "
          f"{MAX_EXPORT_FILE_MB} MB")  # fmt: skip
    print(f"encoders: {GENERAL} against {LEGAL}; batch sizes tried {BATCH_SIZES}; 512 tokens")
    steps = [
        "check out and verify the commit", "hand over to the runner at that commit", "preflight",
        "uv sync --frozen --extra ml", "fetch and build (or attached prebuilt dataset)",
        "fingerprint --check", "index bm25", "index both encoders", "experiment encoders",
    ]  # fmt: skip
    if plan == "full":
        steps.append("remaining GPU experiments")
    print("steps: " + "; ".join([*steps, "export"]))


def bootstrap(plan: str) -> None:
    """Check out the commit, then hand over to the runner inside that checkout.

    Whatever copy of this file Kaggle was given, the code that installs, builds and
    measures is the one committed at the commit being evaluated.
    """
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)
    source = os.environ.get("LEGALRAG_SOURCE", "bundle")
    commit = checkout_from_github() if source == "github" else checkout_from_bundle()
    head = verify_checkout(commit)
    log(f"evaluating commit {head} (plan {plan}, source {source})")
    committed = REPO / "kaggle" / "run_gpu_experiments.py"
    code = subprocess.run(
        [sys.executable, str(committed), "--from-checkout", head],
        env={**os.environ, "LEGALRAG_SOURCE_USED": source},
        check=False,
    ).returncode
    raise SystemExit(code)


def run_from_checkout(plan: str, commit: str) -> None:
    started = time.time()
    this = Path(__file__).resolve()
    if this != (REPO / "kaggle" / "run_gpu_experiments.py").resolve():
        raise SystemExit("--from-checkout must be run from the checked-out repository")
    provenance: dict = {
        "plan": plan,
        "source": os.environ.get("LEGALRAG_SOURCE_USED", "bundle"),
        "commit": verify_checkout(commit),
        "runner_sha256": sha256(this),
    }
    provenance["machine"] = preflight()
    provenance.update(install())
    provenance["data_source"] = prepare_data()
    PLANS[plan]()
    provenance["steps"] = STEPS
    provenance["total_seconds"] = round(time.time() - started, 1)
    provenance["kaggle"] = {k: v for k, v in os.environ.items() if k.startswith("KAGGLE_KERNEL")}
    archive = export(provenance)
    log(f"done in {provenance['total_seconds'] / 60:.1f} minutes: {archive}")


def main() -> None:
    plan = os.environ.get("LEGALRAG_PLAN", "encoders")
    if plan not in PLANS:
        raise SystemExit(f"unknown LEGALRAG_PLAN {plan!r}; choose from {sorted(PLANS)}")
    if "--dry-run" in sys.argv:
        dry_run(plan)
    elif "--from-checkout" in sys.argv:
        run_from_checkout(plan, sys.argv[sys.argv.index("--from-checkout") + 1])
    else:
        bootstrap(plan)


if __name__ == "__main__":
    main()
