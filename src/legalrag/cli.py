"""Command line entry point: ``legalrag <command>``."""

from __future__ import annotations

import argparse
import json
import sys

from legalrag import net
from legalrag.paths import Paths

EXPERIMENTS = (
    "retrieval", "chunking", "rerank", "decomposition", "temporal", "jurisdiction",
    "citations", "judge", "rag", "abstention", "systems",
)  # fmt: skip


def _memory_footprint_gb(pid: int) -> float | None:
    """Physical footprint of a process, including compressed and swapped pages.

    Resident size alone hides a runaway allocation once the system starts
    swapping, so on macOS the figure ``top`` calls MEM is used.
    """
    import subprocess

    try:
        if sys.platform == "darwin":
            out = subprocess.run(
                ["top", "-l", "1", "-pid", str(pid), "-stats", "mem"],
                capture_output=True, text=True, timeout=20, check=False,
            ).stdout.split()[-1]  # fmt: skip
            unit = {"K": 1e-6, "M": 1e-3, "G": 1.0}[out.rstrip("+-")[-1]]
            return float(out.rstrip("+-")[:-1]) * unit
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS"):
                    return int(line.split()[1]) / 1e6
    except (OSError, ValueError, KeyError, IndexError, subprocess.SubprocessError):
        return None
    return None


def _start_memory_guard() -> None:
    """Abort the process if its memory footprint passes ``LEGALRAG_MAX_GB`` (default
    10). On a shared machine a runaway job should fail by itself, not take the
    system down with it."""
    import os
    import threading
    import time

    limit = float(os.environ.get("LEGALRAG_MAX_GB", "10"))
    if limit <= 0:
        return

    def watch() -> None:
        while True:
            time.sleep(10)
            used = _memory_footprint_gb(os.getpid())
            if used is not None and used > limit:
                print(
                    f"aborting: memory footprint {used:.1f} GB exceeds the limit of "
                    f"{limit:.0f} GB (set LEGALRAG_MAX_GB to change it)",
                    file=sys.stderr, flush=True,
                )  # fmt: skip
                os._exit(3)

    threading.Thread(target=watch, daemon=True).start()


def _fetch(args) -> None:
    from legalrag.data import fetch

    paths = Paths.default()
    lepard = fetch.download_lepard(paths.raw)
    print(json.dumps(fetch.fetch_corpus(lepard, paths.cap_cases, args.workers), indent=2))


def _build(args) -> None:
    from legalrag.data.build import build

    manifest = build(Paths.default())
    print(json.dumps({k: manifest[k] for k in ("corpus_version", "chunks", "queries")}, indent=2))


def _index(args) -> None:
    from legalrag.index import store as S

    paths = Paths.default()
    store = S.ChunkStore.load(paths, args.scheme)
    for name in args.names:
        if name == "bm25":
            S.bm25_index(paths, store, rebuild=args.rebuild)
        else:
            S.dense_index(
                paths, store, name, rebuild=args.rebuild, batch_size=args.batch_size,
                max_tokens=args.max_tokens,
            )  # fmt: skip
        meta = json.loads((S.index_dir(paths, store, name if name == "bm25" else f"dense-{name}")
                           / "meta.json").read_text())  # fmt: skip
        print(json.dumps({k: v for k, v in meta.items() if k != "encoder"}))


def _experiment(args) -> None:
    import importlib

    module = importlib.import_module(f"legalrag.experiments.{args.name}")
    kwargs = dict(kv.split("=", 1) for kv in args.option)
    for key, value in list(kwargs.items()):
        if value.isdigit():
            kwargs[key] = int(value)
        elif "," in value:
            kwargs[key] = value.split(",")
    result = module.run(**kwargs)
    if args.print:
        print(json.dumps({k: v for k, v in result.items() if k not in ("versions", "device")},
                         indent=2))  # fmt: skip


def _ask(args) -> None:
    from legalrag.rag.pipeline import Pipeline
    from legalrag.rag.report import to_markdown

    pipeline = Pipeline.from_config(args.config)
    answer = pipeline.answer(" ".join(args.question))
    print(to_markdown(answer))


def _serve(args) -> None:
    import uvicorn

    uvicorn.run("legalrag.api.app:app", host=args.host, port=args.port)


def main(argv: list[str] | None = None) -> int:
    net.configure()
    parser = argparse.ArgumentParser(prog="legalrag", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("fetch", help="download the benchmark file and the cited opinions")
    p.add_argument("--workers", type=int, default=8)
    p.set_defaults(fn=_fetch)

    p = sub.add_parser("build", help="parse, chunk and sample queries")
    p.set_defaults(fn=_build)

    p = sub.add_parser("index", help="build indexes: bm25 or an encoder key")
    p.add_argument("names", nargs="+")
    p.add_argument("--scheme", default="section220")
    p.add_argument("--rebuild", action="store_true")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-tokens", type=int, default=None)
    p.set_defaults(fn=_index)

    p = sub.add_parser("experiment", help="run one experiment and write results/")
    p.add_argument("name", choices=EXPERIMENTS)
    p.add_argument("option", nargs="*", help="key=value arguments for the experiment")
    p.add_argument("--print", action="store_true")
    p.set_defaults(fn=_experiment)

    p = sub.add_parser("ask", help="answer one question from the command line")
    p.add_argument("question", nargs="+")
    p.add_argument("--config", default=None)
    p.set_defaults(fn=_ask)

    p = sub.add_parser("serve", help="start the API")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(fn=_serve)

    args = parser.parse_args(argv)
    if args.command in ("index", "experiment", "build"):
        _start_memory_guard()
    args.fn(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
