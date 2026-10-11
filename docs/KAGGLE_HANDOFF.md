# Kaggle handoff

Where remote GPU work stands and exactly what the next session should do. Updated
2026-10-11 12:20 UTC.

## Decision in force

Decided 2026-10-11: **wait for the weekly GPU quota to refresh on 2026-10-17 00:00 UTC,
then submit the staged follow-up plan unchanged.** Do not split it, restage it or
submit it earlier. Do not rerun the version 2 comparison. Do not cancel or interfere
with `research-innovation-monitor-gpu`. Do not poll Kaggle while waiting. Nothing is
pushed to GitHub without explicit authorisation. The staging folder and the private
dataset stay as they are.

On the next session, before submitting: check the actual quota, the active GPU
sessions and the staged commit (the commands under "Next action"). Submit only with a
free slot and enough quota. After Kaggle accepts the submission, record the version
number, commit, expected outputs and recovery commands in this file, then stop.

## Next action

**Submit the follow-up run (kernel version 4) only when both conditions hold.** It is
staged and has not been submitted.

1. **A GPU slot is free.** The account allows two batch GPU sessions. At 12:15 UTC both
   were taken: version 3 of this kernel (running) and
   `research-innovation-monitor-gpu` (queued). Do not interrupt the latter.
2. **Enough weekly GPU quota remains.** The quota is **6 hours a week** (21,600 s),
   not the 30 hours reported by third parties. At 12:16 UTC 1.93 h had been used.
   Version 3 will add about 1.75 h, leaving roughly 2.3 h before the refresh on
   2026-10-17 00:00 UTC, less whatever the other project's run consumes. The follow-up
   plan is estimated at 3 to 4 hours and has never run on a GPU. Submitting it with
   less quota than that risks a run cut off part-way.

If the quota is short, the choices are: wait for the refresh on 17 October and submit
the plan unchanged, or decide to run a subset. The plan staged is the full `followup`
plan, by decision; changing it means restaging at a new commit.

```bash
cd ~/Documents/GitHub/legal-research-rag

# 1. One check each. Proceed only with a free slot and enough quota.
kaggle kernels status adebanjiadelowo/legal-research-rag-gpu
kaggle kernels status adebanjiadelowo/research-innovation-monitor-gpu
/opt/anaconda3/bin/python kaggle/quota.py

# 2. Confirm the staged files still describe commit 18973fd and plan followup.
cat ~/kaggle-staging/legal-research-rag/source/COMMIT ~/kaggle-staging/legal-research-rag/source/PLAN
cmp ~/kaggle-staging/legal-research-rag/kernel/run_gpu_experiments.py \
    <(git show 18973fd:kaggle/run_gpu_experiments.py) && echo "runner matches the commit"

# 3. Submit. Run this alone: do not chain it after another command.
/opt/anaconda3/bin/python kaggle/kaggle_ipv4.py kernels push \
    -p ~/kaggle-staging/legal-research-rag/kernel --accelerator NvidiaTeslaT4

# 4. One status check to confirm QUEUED or RUNNING, record the version number below, stop.
kaggle kernels status adebanjiadelowo/legal-research-rag-gpu
```

The private dataset `adebanjiadelowo/legal-research-rag-src` is already at the right
version (uploaded 12:09 UTC: `COMMIT` = `18973fd0e3bd0e313650e7fc4799d70bfd144649`,
`PLAN` = `followup`). It does not need to be uploaded again unless the code changes.

If the code has changed since `18973fd`, restage first, because Kaggle must evaluate
the commit that is recorded:

```bash
python kaggle/prepare.py --username adebanjiadelowo --out ~/kaggle-staging/legal-research-rag --plan followup
/opt/anaconda3/bin/python kaggle/kaggle_ipv4.py datasets version \
    -p ~/kaggle-staging/legal-research-rag/source -m "<commit> followup"
```

## Kernel versions

Kernel `adebanjiadelowo/legal-research-rag-gpu` (private, Tesla T4):
https://www.kaggle.com/code/adebanjiadelowo/legal-research-rag-gpu

| Version | Commit | Plan | State | Outcome |
|---|---|---|---|---|
| 1 | `fa2a992` | encoders | error after 40 s | Runner did not find the bundle under `/kaggle/input`; fixed in `9ecd6f7` |
| 2 | `9ecd6f7` | encoders | **complete, validated, imported** | RQ2 results in `results/encoders/`, committed in `e6af5c6` |
| 3 | `9ecd6f7` | encoders | running when this was written (started about 12:06 UTC) | **Submitted by mistake.** A duplicate of version 2. See below |
| 4 | `18973fd` | followup | **not submitted** | See "Next action" |

### Version 3: an accidental duplicate

A chained shell command pushed the kernel after its dataset-upload step had been
killed, so version 3 ran against the previous dataset version: the same commit and
plan as version 2. It costs about 105 minutes of GPU quota and adds nothing new.

- It can be cancelled from the kernel's web page, and only there. The CLI has no
  cancel command. The SDK's `cancel_kernel_session` takes a session identifier, and no
  authenticated call returns one: not the push response (`kernel_id`, `ref`, `url`,
  `version_number`), not the kernel metadata, not the status call. Without it the
  session cannot be identified, so no cancellation was attempted. Checked again at
  12:16 UTC; it was still running.
- It uses about 1.75 h of a 6-hour weekly GPU quota.
- If it is left to finish, its output is an exact rerun and can be used as a
  replication check: download it, run the validator, and compare the metric values
  with `results/encoders/`. Do not import it over the version 2 files without
  comparing first.
- It occupies one of two GPU slots until it ends, and another project's kernel
  (`research-innovation-monitor-gpu`) was queued at the same time.

## The follow-up run (version 4)

Experiment configuration, all from `plan_followup` in `kaggle/run_gpu_experiments.py`
at `18973fd`. BM25 uses the development-tuned k1 = 1.6, b = 0.75. No dense index is
built. Each step is independent: a failed step is recorded and the next still runs.

| Step | Command arguments | Model and revision | Expected result file |
|---|---|---|---|
| rerank | `rerank reranker=bge-reranker-v2-m3` | `BAAI/bge-reranker-v2-m3` at `953dc6f6`, half precision | `rerank/rerank_bm25_bge-reranker-v2-m3.json` |
| judge | `judge judges=lexical,nli-xsmall,nli-base,nli-large` | adds `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` at `b3546ea6` | `verification/judge_validation_lexical+nli-xsmall+nli-base+nli-large.json` |
| citations | `citations judge=nli-large limit=150` | same large judge | `verification/citation_checker_nli-large.json` |
| abstention | `abstention reranker=bge-reranker-v2-m3` | same reranker | `abstention/abstention_bm25+bge-reranker-v2-m3.json` |
| rag | `rag generator=hf:microsoft/Phi-3-mini-4k-instruct@f39ac1d2 limit=200 evidence_k=4 eval_judge=nli-xsmall filter_judge=lexical tag=phi3-mini-4k` | Phi-3-mini-4k-instruct at `f39ac1d2`, half precision | `rag/rag_phi3-mini-4k.json` |

Also expected in the export: `provenance.json` (commit, plan, machine, step timings,
`failed_steps`), `fingerprint.json`, `manifest.json`, `requirements-frozen.txt`,
`indexes/bm25/meta.json` and `logs/`.

Runtime is not known. A rough estimate is three to four hours; none of these steps
has run on a GPU before, so a failed step is possible and is handled by design.

## When a run has finished

Do not resubmit. Retrieve, validate, then import.

```bash
cd ~/Documents/GitHub/legal-research-rag
OUT=~/kaggle-staging/legal-research-rag/output-v4          # one folder per version
mkdir -p $OUT && kaggle kernels output adebanjiadelowo/legal-research-rag-gpu -p $OUT

# Validation: file types and sizes, archive against files, commit, runner and uv.lock
# hashes, corpus fingerprint, step exit codes, result provenance, text-leak scan.
python kaggle/validate_export.py $OUT            # must print VALID
python kaggle/validate_export.py $OUT --import   # copies results and provenance

python scripts/make_results_tables.py
uv run pytest -q && uv run ruff check src tests app scripts kaggle
```

Then read the new numbers, update the README findings and `docs/error_analysis.md`
by hand (the tables are generated; the prose is not), update `docs/STATUS.md` and this
file, and commit locally. Report results as measured, including steps that failed and
results that go against expectation.

`kaggle kernels output` returns the latest finished version only. Version 2's output
is preserved locally in `~/kaggle-staging/legal-research-rag/output-v2/` (archive
SHA-256 `23ac7895694576d7cae611891541f23555832d1aae35864feef09f5b313f8394`).

## What has been verified

| Item | Evidence |
|---|---|
| Version 2 evaluated the intended commit | `provenance.json`: `9ecd6f70f4f1b9e9bb6b69e209e45235537acd73`; runner and `uv.lock` hashes equal the files at that commit |
| Same corpus and benchmark as the local results | `fingerprint.json` equals `kaggle/expected_fingerprint.json`: corpus `c1b900e309df`, 222,520 passages, 800 development and 2,000 test queries |
| No restricted text left Kaggle | export holds `.json`, `.log`, `.txt` only; 0 of 3,406 sampled benchmark and opinion text windows found |
| Results imported unchanged | `results/encoders/kaggle_run_v2/SHA256SUMS.json` |
| Validator agrees with the manual checks | `python kaggle/validate_export.py ~/kaggle-staging/legal-research-rag/output-v2` prints `VALID` |

## Practical notes

- **IPv6.** On this network the Kaggle CLI's uploads hang while connecting to the
  storage host. Use `kaggle/kaggle_ipv4.py` for `datasets version`, `datasets create`
  and `kernels push`. Read-only calls (`status`, `output`, `files`) work with the plain
  CLI.
- **Virtualenv.** macOS repeatedly marks `.venv/lib/python3.12/site-packages/*.pth`
  as hidden, which breaks `legalrag` and `uv run`. Fix with
  `chflags nohidden .venv/lib/python3.12/site-packages/*.pth`, or prefix commands with
  `PYTHONPATH=$PWD/src`. Tests do not depend on it.
- **Never chain a submission** after a step that can fail or be interrupted. Version 3
  is the result of doing that.
- **Quota.** 6 GPU hours a week, refreshing Saturday 00:00 UTC. Version 2 used 105
  minutes; version 3 will use about the same. Check with `kaggle/quota.py`.
- The staging folder `~/kaggle-staging/legal-research-rag/` is outside the repository
  and holds the bundle, the kernel files and the downloaded outputs.

## Remaining scientific experiments

1. The follow-up run above: a stronger reranker (RQ3), the large entailment judge,
   reranker-based abstention (RQ7), and generation with a capable model (RQ4).
2. Dense and hybrid retrieval in the chunking, decomposition, temporal and
   jurisdiction experiments, which use BM25 only.
3. Reranking over the best first stage (hybrid with the legal encoder), which needs
   the 43-minute legal index and is not in the follow-up plan.
4. A test of the legal encoder on opinions outside its training data, to separate
   exposure from domain adaptation. This needs a newer corpus than the one here.
