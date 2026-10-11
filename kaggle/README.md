# Kaggle GPU runner

`run_gpu_experiments.py` runs, on a Kaggle GPU, the comparison that is too heavy for a
laptop: a legal-domain embedding model against its general-purpose parent (RQ2).

This comparison has been run once (kernel version 2, commit `9ecd6f7`, Tesla T4). Its
results are in the main README and its measured resource use is in the section
"Measured on the completed run" below. Everything under "Resource estimates" was
written before the run and is kept for comparison.

## What is compared

| | General | Legal |
|---|---|---|
| Key | `modernbert-general` | `modernbert-legal` |
| Model | `nomic-ai/modernbert-embed-base` | `freelawproject/modernbert-embed-base_finetune_512` |
| Revision | `d556a88e332558790b210f7bdbe87da2fa94a8d8` | `04f0141fbc045122439d28d51ba670f3091e9ed8` |
| Licence | Apache-2.0 | CC0-1.0 |
| Weights | 596,070,136 bytes | 596,070,136 bytes |
| Relationship | parent | fine-tune of the parent on `Free-Law-Project/opinions-synthetic-query-512` |

The second is a fine-tune of the first, so architecture (ModernBERT-base, 149M
parameters), tokeniser, pooling (mean), output dimension (768) and normalisation (L2)
are the same. Both model cards document the same input convention, which the code
applies: `search_query: ` before a query and `search_document: ` before a passage.
Revisions, licences, sizes and conventions were read from the Hugging Face Hub on
2026-10-10.

### Conditions held identical

| Condition | Value | Enforced by |
|---|---|---|
| Passages | 222,520 section-aware passages of corpus version `c1b900e309df` | fingerprint check; index metadata check |
| Development queries | 800, citing opinions from 2000 to 2009 | fingerprint check |
| Test queries | 2,000, citing opinions from 2010 to 2018 | fingerprint check |
| Relevance labels | the case each citing court quoted, and the quoted paragraph | part of the query fingerprint |
| Query text | citations and case names masked, last 120 words | part of the query fingerprint |
| Sequence limit | 512 tokens for both (the legal model was fine-tuned at 512) | `MUST_MATCH` in `experiments/encoders.py` |
| Precision | half precision on GPU; embeddings stored as float16 | same |
| Batch size | one value for both; if either runs out of memory, both are rebuilt smaller | runner |
| Search | exact inner product over all passages, top 1,000 | experiment |
| Date filter | only cases decided before the query's date | experiment |
| Metrics | Recall@1/5/10/20/100, MRR@10, nDCG@10, Precision@10, paragraph hit@k | experiment |

The experiment raises an error, and writes no result, if the two indexes differ in
corpus version, passage count, sequence limit, precision, batch size or device.

### What is tuned, and on what

Only the BM25 weight of the hybrid, separately per encoder, on the development
queries. BM25's own parameters (k1 = 1.6, b = 0.75) were tuned earlier on the same
development queries. Nothing is chosen on the test queries, and both encoders are
reported whichever is better.

### What is reported

For each encoder, dense alone and as a hybrid with BM25: the metrics above with 95%
bootstrap intervals. For legal minus general: the paired difference on nDCG@10,
MRR@10, Recall@1, Recall@10, Recall@100 and paragraph hit@10, each with a bootstrap
interval and a randomisation-test p-value, Holm-adjusted across the six.

Systems measurements, per encoder: corpus encoding time and passages per second; peak
GPU memory during encoding and during query encoding (`torch.cuda.max_memory_allocated`
and `max_memory_reserved`); embedding storage on disk and in memory; per-query
encoding and search latency (mean, median, 95th percentile); and the number of
passages longer than the 512-token limit.

### A limit of this comparison

The legal model was fine-tuned on synthetic queries over U.S. court opinions, which
may include opinions in this corpus. A gain for it can reflect exposure to these
documents as well as domain adaptation. The experiment cannot separate the two.

## Data and licensing

| Item | Source | Licence | Handling on Kaggle |
|---|---|---|---|
| LePaRD `top_10000_data.csv.gz` | Hugging Face `rmahari/LePaRD` at revision `0194f95c3091acceab3b887c9b09ef432cf84052`, SHA-256 checked | CC BY-NC-SA 4.0 | downloaded inside the notebook; never uploaded or exported |
| 4,940 opinions | Caselaw Access Project, `static.case.law` | CC0 1.0 | fetched inside the notebook; not exported |
| Source code | this repository at one commit | MIT | private Kaggle dataset (a Git bundle), or a private GitHub repository |

Both datasets are public and need no credentials. The runner rebuilds the corpus and
benchmark and then compares their hashes with `expected_fingerprint.json`, which was
written from the build that produced the local results. If the rebuild differs in any
passage, query or label, the run stops before encoding.

If the rebuild cannot reproduce the fingerprint (for example because a source file
has changed upstream), attach the locally built `data/processed/` folder as a second
**private** Kaggle dataset. The runner detects a `processed/manifest.json` under
`/kaggle/input` and uses it in place of the fetch and build, still subject to the
fingerprint check. That folder contains text derived from LePaRD, so it must stay
private and non-commercial.

Work happens under `/tmp/legalrag`, which Kaggle does not save. The export written to
`/kaggle/working` holds only result JSON files, index metadata, the build manifest
(counts and hashes), the fingerprint, step logs, the frozen package list and a
provenance file. Before zipping, the runner aborts if the export contains any file
that is not `.json`, `.txt`, `.log` or `.md`, or any file above 5 MB. Embeddings,
opinions, queries and per-answer outputs are not exported.

## Resource estimates

None of the GPU figures below has been measured. They are derived from model size,
token counts and one local reference point (a 33M-parameter encoder at 180 passages
per second on an Apple M3 Pro). The run records the measured values.

| Item | Estimate | Basis |
|---|---|---|
| Downloads | about 2.1 GB of data and weights (LePaRD 0.54, opinions 0.31, two models 1.19), plus the locked Python packages | file sizes |
| Disk under `/tmp` | about 12 GB: packages with CUDA libraries 6 to 7, data 2.2, models 1.2, two indexes 0.7 | file sizes; package size not measured |
| Embedding storage | 341.8 MB per encoder on disk (222,520 x 768 x 2 bytes); 683.6 MB in memory as float32 for search | arithmetic |
| GPU memory | 3 to 6 GB peak at batch size 128 and 512 tokens in half precision | 0.3 GB of weights plus activations; not measured |
| Corpus encoding | 10 to 30 minutes per encoder on a T4 | about 56 million tokens per pass; not measured |
| CPU steps | fetch 5 to 15 minutes; build 8 to 15; BM25 index 2; experiment 10 to 20 | local timings on an M3 Pro, scaled loosely |
| Whole run | 1 to 2 hours | sum of the above |
| System RAM | under 10 GB | 4.9 GB peak measured locally with one dense index |

A T4 has 16 GB of memory, so the estimate leaves a wide margin. If encoding does run
out of memory, the runner deletes both dense indexes and rebuilds both at half the
batch size (128, 64, 32, 16), so the two encoders are never built under different
settings. Batch size changes padding only, and the experiment records it.

Kaggle's limits as reported by third parties (not confirmed against Kaggle's own
documentation; check the notebook settings): 12 hours per GPU session, about 30 GPU
hours per week, 20 GB of saved output, and 13 or 29 GB of RAM depending on the source.
The runner checks free disk and GPU memory before starting, and sets its memory guard
to 85% of the RAM it finds.

### Measured on the completed run

Kernel `adebanjiadelowo/legal-research-rag-gpu`, version 2, batch run on a Tesla T4
(two attached, one used), 1,149 GB free scratch disk, memory guard 27 GB.

| Item | Measured | Estimate above |
|---|---|---|
| Whole run | 105 minutes | 1 to 2 hours |
| Locked install (`uv sync`) | 72 seconds; torch 2.14.1 with CUDA 13.0 | not estimated |
| Fetch | 3.1 minutes: 4,762 opinions by identifier, 178 by page, 36 not hosted, the same counts as locally | 5 to 15 minutes |
| Build | 5.4 minutes | 8 to 15 minutes |
| Fingerprint | matched `expected_fingerprint.json` exactly | must match |
| Corpus encoding, each encoder | 42.9 minutes at 86 passages per second | 10 to 30 minutes: too optimistic |
| Peak GPU memory, corpus encoding | 1,904 MB allocated, 2,300 MB reserved, batch size 128 | 3 to 6 GB: too pessimistic |
| Embeddings | 326 MiB per encoder on disk (341.8 MB), 652 MiB in memory | as computed |
| Passages over 512 tokens | 373 of 222,520 (0.17%); longest 883 tokens | not estimated |
| Query latency | 29 to 30 ms to encode, 27 ms to search, per query | not estimated |
| Experiment step | 2.4 minutes | 10 to 20 minutes |

The batch size never had to be reduced. Kaggle's own Python start-up hook printed
`ModuleNotFoundError: No module named 'wrapt'` at the start of each step inside the
locked environment; it is unrelated to this code and every step exited 0.

### If a session is interrupted

Corpus encoding writes one file per block of 8,192 passages and resumes from the first
missing block, so re-running the script in the same session continues where it
stopped; a resumed build records how many passages came from earlier blocks, and
therefore that its timing is partial. Blocks live under `/tmp` and do not survive the
end of a session. With an expected run of one to two hours against a twelve-hour limit
this should not be needed. The whole corpus is always encoded: the index build fails
if the number of embeddings differs from the number of passages, and no option
reduces the corpus or the query set.

## Procedure

The repository has no remote. There are two ways to give Kaggle the code; both
evaluate one exact commit. Neither has been executed yet.

### Option A: private Kaggle dataset holding a Git bundle (no GitHub needed)

```bash
# 1. Stage the files locally. This uploads nothing and requires a clean working tree.
python kaggle/prepare.py --username <kaggle-username> --out ~/kaggle-staging/legal-research-rag

# 2. Create the private dataset (first time) or a new version (later).
kaggle datasets create  -p ~/kaggle-staging/legal-research-rag/source
kaggle datasets version -p ~/kaggle-staging/legal-research-rag/source -m "<commit>"

# 3. Push and run the private kernel (GPU and internet enabled in its metadata).
kaggle kernels push -p ~/kaggle-staging/legal-research-rag/kernel

# 4. Watch it, then fetch the export.
kaggle kernels status <kaggle-username>/legal-research-rag-gpu
kaggle kernels output <kaggle-username>/legal-research-rag-gpu -p ~/kaggle-staging/output

# 5. Merge the results and regenerate the tables.
unzip -o ~/kaggle-staging/output/legal-research-rag-export.zip -d ~/kaggle-staging/export
cp -R ~/kaggle-staging/export/results/encoders results/
python scripts/make_results_tables.py
```

The bundle contains exactly what Git tracks: code, documentation and aggregate result
files, under 1 MB. The runner clones it, checks out the hash in `COMMIT`, verifies the
checkout is that commit and is clean, and then hands over to the copy of itself inside
the checkout, so the code that runs is the committed code.

### Option B: private GitHub repository through Kaggle Secrets

Use this only if the repository is later pushed to GitHub, which needs an explicit
decision. Then:

1. Create a fine-grained personal access token with read-only **Contents** permission
   on that one repository.
2. In the Kaggle notebook, add it under Add-ons, Secrets, with the label
   `GITHUB_TOKEN`, and attach the secret to the notebook.
3. Set `LEGALRAG_SOURCE=github`, `LEGALRAG_GITHUB_REPO=<owner>/<name>` and
   `LEGALRAG_COMMIT=<full hash>` at the top of the kernel.

The runner fetches that single commit over HTTPS, does not print the token, and keeps
no remote configured afterwards. This path is implemented but untested.

### Options summary

| | Option A: bundle dataset | Option B: private GitHub |
|---|---|---|
| Needs a GitHub repository | no | yes |
| Secret required | none | `GITHUB_TOKEN` in Kaggle Secrets |
| What leaves the machine | a private Kaggle dataset with the tracked files | a push to a private repository |
| Tested locally | staging, cloning the bundle, commit verification and the hand-over to the committed runner, up to the machine preflight | precondition check only |

## Other plans

`LEGALRAG_PLAN=full` runs the encoder comparison and then the remaining GPU
experiments: a larger reranker, the large entailment judge, and generation with an
open instruction-tuned model. That plan downloads about 20 GB more (the 7B generator
is 15.2 GB) and needs the two-GPU option for the generator. The default plan is
`encoders`.

`python kaggle/run_gpu_experiments.py --dry-run` prints the plan without running it.
