"""The Kaggle runner's local logic: checkout verification, staging and the export filter.

Nothing here touches the network or a GPU; Git is used on temporary repositories.
"""

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "kaggle" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git(cwd, *args):
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
    ).stdout.strip()


@pytest.fixture()
def repo(tmp_path):
    path = tmp_path / "origin"
    (path / "kaggle").mkdir(parents=True)
    git(path, "init", "-q", "-b", "main")
    git(path, "config", "user.email", "t@example.com")
    git(path, "config", "user.name", "t")
    (path / "kaggle" / "run_gpu_experiments.py").write_text("print('runner v1')\n")
    (path / "code.py").write_text("x = 1\n")
    git(path, "add", "-A")
    git(path, "commit", "-q", "-m", "one")
    return path


def test_stage_writes_bundle_commit_and_private_kernel(repo, tmp_path, monkeypatch):
    prepare = load("prepare")
    monkeypatch.setattr(prepare, "ROOT", repo)
    staged = prepare.stage(tmp_path / "out", "someone")
    commit = git(repo, "rev-parse", "HEAD")
    assert staged["commit"] == commit
    assert (staged["source"] / "COMMIT").read_text().strip() == commit
    meta = json.loads((staged["kernel"] / "kernel-metadata.json").read_text())
    assert meta["is_private"] is True and meta["enable_gpu"] and meta["enable_internet"]
    assert meta["dataset_sources"] == ["someone/legal-research-rag-src"]
    # The staged runner is the committed file, byte for byte.
    assert (staged["kernel"] / "run_gpu_experiments.py").read_bytes() == (
        repo / "kaggle" / "run_gpu_experiments.py"
    ).read_bytes()
    # The bundle reproduces the commit.
    clone = tmp_path / "clone"
    git(tmp_path, "clone", "-q", str(staged["source"] / prepare.BUNDLE), str(clone))
    assert git(clone, "rev-parse", "HEAD") == commit


def test_stage_refuses_a_dirty_tree_and_an_output_inside_the_repo(repo, tmp_path, monkeypatch):
    prepare = load("prepare")
    monkeypatch.setattr(prepare, "ROOT", repo)
    with pytest.raises(SystemExit, match="outside the repository"):
        prepare.stage(repo / "staging", "someone")
    (repo / "code.py").write_text("x = 2\n")
    with pytest.raises(SystemExit, match="commit or stash"):
        prepare.stage(tmp_path / "out", "someone")


def test_checkout_from_bundle_and_verification(repo, tmp_path, monkeypatch):
    prepare, runner = load("prepare"), load("run_gpu_experiments")
    monkeypatch.setattr(prepare, "ROOT", repo)
    staged = prepare.stage(tmp_path / "out", "someone")
    monkeypatch.setattr(runner, "WORK", tmp_path / "work")
    monkeypatch.setattr(runner, "REPO", tmp_path / "work" / "repo")
    monkeypatch.setenv("LEGALRAG_SOURCE_DATASET", str(staged["source"]))
    (tmp_path / "work").mkdir()
    commit = runner.checkout_from_bundle()
    assert runner.verify_checkout(commit) == staged["commit"]
    with pytest.raises(SystemExit, match="expected"):
        runner.verify_checkout("0" * 40)
    (runner.REPO / "code.py").write_text("tampered\n")
    with pytest.raises(SystemExit, match="not clean"):
        runner.verify_checkout(commit)


def test_github_source_needs_repo_commit_and_secret(monkeypatch):
    runner = load("run_gpu_experiments")
    for name in ("LEGALRAG_GITHUB_REPO", "LEGALRAG_COMMIT", "GITHUB_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(SystemExit, match="GITHUB_TOKEN"):
        runner.checkout_from_github()


def _fake_run(runner, tmp_path, monkeypatch):
    work, export = tmp_path / "work", tmp_path / "export"
    monkeypatch.setattr(runner, "WORK", work)
    monkeypatch.setattr(runner, "EXPORT", export)
    home = work / "home"
    (home / "results" / "encoders").mkdir(parents=True)
    (home / "results" / "rag").mkdir(parents=True)
    (home / "results" / "encoders" / "encoders_a_vs_b.json").write_text('{"ok": true}')
    (home / "results" / "rag" / "answers_x.json").write_text('{"text": "opinion text"}')
    index = home / "data" / "indexes" / "v1" / "section220" / "dense-a"
    index.mkdir(parents=True)
    (index / "meta.json").write_text('{"chunks": 3}')
    (index / "embeddings.npy").write_bytes(b"\0" * 64)
    (home / "data" / "processed").mkdir(parents=True)
    (home / "data" / "processed" / "manifest.json").write_text('{"corpus_version": "v1"}')
    (home / "data" / "processed" / "corpus.jsonl").write_text('{"text": "an opinion"}\n')
    (home / "data" / "processed" / "queries_test.jsonl").write_text('{"text": "a query"}\n')
    (work / "logs").mkdir()
    (work / "logs" / "00-step.log").write_text("progress\n")
    return work, export


def test_export_holds_results_and_provenance_but_no_text_or_embeddings(tmp_path, monkeypatch):
    runner = load("run_gpu_experiments")
    _, export = _fake_run(runner, tmp_path, monkeypatch)
    archive = runner.export({"commit": "abc", "freeze": "numpy==2.0\n"})
    names = sorted(str(p.relative_to(export)) for p in export.rglob("*") if p.is_file())
    assert names == [
        "indexes/dense-a/meta.json",
        "logs/00-step.log",
        "manifest.json",
        "provenance.json",
        "requirements-frozen.txt",
        "results/encoders/encoders_a_vs_b.json",
    ]
    assert archive.exists() and archive.suffix == ".zip"
    assert json.loads((export / "provenance.json").read_text()) == {"commit": "abc"}
    blob = b"".join(p.read_bytes() for p in export.rglob("*") if p.is_file())
    assert b"opinion" not in blob and b"a query" not in blob


def test_export_aborts_on_an_unexpected_file(tmp_path, monkeypatch):
    runner = load("run_gpu_experiments")
    work, export = _fake_run(runner, tmp_path, monkeypatch)
    (work / "logs" / "corpus.jsonl").write_text('{"text": "an opinion"}\n')
    with pytest.raises(SystemExit, match="refusing to export"):
        runner.export({"freeze": ""})
    assert not export.exists()


def test_plan_names_and_expected_fingerprint_are_consistent():
    runner = load("run_gpu_experiments")
    from legalrag.index.dense import ENCODERS

    assert set(runner.PLANS) == {"encoders", "full"}
    for key in (runner.GENERAL, runner.LEGAL):
        spec = ENCODERS[key]
        assert len(spec.revision) == 40 and spec.max_tokens == 512
        assert (spec.query_prefix, spec.doc_prefix) == ("search_query: ", "search_document: ")
    assert ENCODERS[runner.GENERAL].domain == "general" and ENCODERS[runner.LEGAL].domain == "legal"
    expected = json.loads((ROOT / "kaggle" / "expected_fingerprint.json").read_text())
    assert (
        expected["queries"]["test"]["count"] == 2000 and expected["queries"]["dev"]["count"] == 800
    )
    assert len(expected["chunks"]["section220"]["sha256"]) == 64
