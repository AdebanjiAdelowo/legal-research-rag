# Experiment status

Which experiments have result files, which do not, and the exact command for each
that does not. Updated 2026-10-11. Corpus version `c1b900e309df`.

## Run, with results under `results/`

| Experiment | Result file | Setting |
|---|---|---|
| Retrieval: BM25, dense, hybrid | `retrieval/retrieval_section220.json` | BM25 and `bge-small-en-v1.5`, 2,000 test queries |
| Leakage control (masked against unmasked queries) | same file, `leakage_control` | BM25 and bge-small |
| Reranking | `rerank/rerank_bm25_minilm-l6.json`, `rerank/rerank_hybrid-score-bge-small_minilm-l6.json` | `ms-marco-MiniLM-L6-v2` |
| Chunking: section-aware, fixed 220, fixed 170 | `chunking/chunking_bm25.json` | BM25 |
| Query decomposition | `decomposition/decomposition_bm25.json` | BM25, 600 multi-issue and 1,000 single-issue queries |
| Temporal leakage and outdated authority | `robustness/temporal_bm25.json` | BM25 |
| Jurisdiction and scope detection | `robustness/jurisdiction_bm25.json` | BM25 |
| Abstention | `abstention/abstention_bm25.json`, `abstention/abstention_bm25+minilm-l6.json` | BM25 score and cross-encoder score |
| Support-judge validation | `verification/judge_validation_lexical+nli-xsmall+nli-base.json` | 300 pairs of each kind, thresholds also tuned on development pairs |
| Citation-checker validation | `verification/citation_checker_lexical.json`, `verification/citation_checker_nli-xsmall.json` | Eight planted error types |
| Grounded generation, extractive | `rag/rag_extractive.json` | 300 test queries, normal and withheld evidence |
| Grounded generation, local language model | `rag/rag_tinyllama-1.1b.json` | `TinyLlama-1.1B-Chat-v1.0`, 100 test queries, three evidence passages |
| Error analysis | `analysis/error_analysis_bm25.json` | BM25 |
| Latency and resources | `systems/systems_bm25+bge-small+minilm-l6.json` | Apple M3 Pro |
| Container image | none (manual check) | Built locally with Docker 29.8: 589 MB, non-root user, health check passes; `/health`, `/answer`, `/report` and `/citations/validate` answered on the synthetic corpus |

## Not run

Each of these needs a download above 500 MB, more memory than the laptop can spare
alongside other work, a GPU, or a remote repository. None was started.

| Experiment | Why not run | Command |
|---|---|---|
| RQ2: legal against general encoder (`freelawproject/modernbert-embed-base_finetune_512` at `04f0141f` against `nomic-ai/modernbert-embed-base` at `d556a88e`) | Two 596 MB downloads and several hours of encoding per model on a laptop. Prepared for a Kaggle GPU: experiment `experiments/encoders.py`, runner and staging script under `kaggle/`, fingerprint of the local corpus and benchmark committed. Not executed; needs a Kaggle account action | `python kaggle/prepare.py --username <kaggle-username> --out <staging folder>`, then the `kaggle` commands it prints (`kaggle/README.md`) |
| Reranking with a larger or legal-domain cross-encoder | 2.3 GB download (`BAAI/bge-reranker-v2-m3`) | `legalrag experiment rerank reranker=bge-reranker-v2-m3 encoder=bge-small fusion=score alpha=0.7 k1=1.6 b=0.75` |
| Generation with a capable open model | `Phi-3-mini-4k-instruct` is cached (7.6 GB) but needs about 8 GB of memory beside a 2 GB corpus on an 18 GB shared machine; `Qwen2.5-7B-Instruct` is a 15 GB download | `legalrag experiment rag generator=hf:microsoft/Phi-3-mini-4k-instruct limit=100 eval_judge=nli-xsmall filter_judge=lexical k1=1.6 b=0.75 tag=phi3-mini` (set `LEGALRAG_MAX_GB=14`) |
| Large entailment judge | 870 MB download | `legalrag experiment judge judges=lexical,nli-xsmall,nli-base,nli-large` |
| Dense retrieval in the chunking, decomposition, temporal and jurisdiction experiments | Those experiments use BM25 only; a second dense index (171 MB, 20 minutes) is needed for the fixed-window scheme | extend `experiments/chunking.py` and `experiments/decomposition.py` to take a retriever name |
| Everything above in one job on a GPU | Needs a Kaggle account action: creating a private dataset and pushing a kernel | `kaggle/README.md` |

## Verified without being an experiment

- **Real-corpus smoke test.** `legalrag ask` with `configs/default.yaml` answered a
  question restricted to the Ninth Circuit before 2009 on the full corpus: the
  constraints were parsed and applied, evidence passages were selected from Supreme
  Court decisions, and verbatim statements were produced with their citation checks
  attached. This shows the pipeline runs end to end on real data; it is one query and
  measures nothing.
- **Kaggle preparation, local parts only.** Staging a Git bundle, cloning it and
  verifying the commit, refusing a dirty tree, the export filter, and the encoder
  experiment on a synthetic corpus with stand-in encoders are covered by tests. The
  runner has not been executed on Kaggle: the GPU steps, the locked install on Linux,
  the rebuild reproducing the fingerprint, and the GitHub source are untested.

## Known gaps in what was run

- Reranking, generation and abstention with a reranker all use a 22M-parameter
  cross-encoder trained on web search. Its negative result should not be generalised.
- The local language model is 1.1B parameters. Its results show the verifier at work
  on a weak generator; they say little about what a capable model would produce.
- Generation experiments retrieve with BM25 only.
- The scope-detection questions were written by the author of the detection rules.
- The CI workflow has never executed: the repository has no remote. Its steps were run
  by hand (lint, format check, tests with the neural packages blocked, the CLI on the
  synthetic corpus, the container build and its API checks).
