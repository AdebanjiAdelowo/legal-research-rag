"""Run records. Every result file states what produced it: configuration, corpus
version, model revisions, seed, device, package versions and Git state."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path
from typing import Any

from legalrag.paths import REPO_ROOT


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10, check=False
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def git_state() -> dict[str, Any]:
    dirty = [line[3:] for line in _git("status", "--porcelain").splitlines() if line]
    return {"commit": _git("rev-parse", "HEAD") or None, "dirty_paths": dirty}


def versions(packages=("numpy", "scipy", "torch", "sentence-transformers", "transformers")):
    out: dict[str, str] = {"python": sys.version.split()[0]}
    for name in packages:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return out


def device_info() -> dict[str, Any]:
    info: dict[str, Any] = {"platform": platform.platform(), "machine": platform.machine()}
    try:
        import torch

        if torch.cuda.is_available():
            info["device"] = "cuda"
            info["gpu"] = torch.cuda.get_device_name(0)
        elif torch.backends.mps.is_available():
            info["device"] = "mps"
        else:
            info["device"] = "cpu"
    except ImportError:
        info["device"] = "cpu"
    return info


def run_record(config: dict[str, Any], corpus_version: str | None = None, **extra) -> dict:
    return {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": config,
        "corpus_version": corpus_version,
        "git": git_state(),
        "versions": versions(),
        "device": device_info(),
        **extra,
    }


def write_result(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
    return path


def read_jsonl(path: Path):
    with path.open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n
