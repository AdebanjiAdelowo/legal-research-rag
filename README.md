# Legal Research RAG

Case-law retrieval, source-grounded research summaries and citation verification over
U.S. federal court opinions, with each stage measured against simpler baselines.

Given a research question, the system finds relevant opinions, selects the passages
that bear on it, writes a summary in which every statement is tied to a passage, and
then checks its own citations. It abstains when the evidence is not there.

> **Scope and limitation.** This is a research system for legal information retrieval.
> It is not a lawyer and does not give legal advice. It covers 4,940 U.S. federal
> opinions decided between 1824 and 2016, has no citator, and cannot tell whether a
> case it returns is still good law.

## Contents

- [Motivation](#motivation)
- [Research questions](#research-questions)
- [Key findings](#key-findings)
- [Architecture](#architecture)
- [Data](#data)
- [Methods](#methods)
- [Experimental protocol](#experimental-protocol)
- [Results](#results)
- [Limitations](#limitations)
- [Installation](#installation)
- [Usage](#usage)
- [Reproducing the experiments](#reproducing-the-experiments)
- [Tests](#tests)
- [Repository structure](#repository-structure)
- [References](#references)
- [License](#license)

## Motivation

Legal research tools built on language models fail in a characteristic way: they
produce fluent answers that cite cases which do not exist, or which exist and say
something else. Retrieval-augmented generation reduces the first failure and does
little about the second, because a real citation attached to an unsupported sentence
looks exactly like a correct one.

Law adds constraints that general retrieval ignores. A relevant case may not bind the
court in question. A case decided after the events at issue could not have been
relied on. A case can be on point and overruled.

This project treats each of these as something to measure: how well different
retrievers find the case a court actually relied on, what a reranker adds, whether
generated statements are supported by the passages they cite, whether the system can
tell when it should not answer, and what happens to all of this when dates and
jurisdiction are handled wrongly.

## Research questions

| | Question |
|---|---|
| RQ1 | How do BM25, dense retrieval and hybrid retrieval compare on case-law queries? |
| RQ2 | Does a legal-domain embedding model outperform its general-purpose parent? |
| RQ3 | How much does cross-encoder reranking improve relevance, and at what latency? |
| RQ4 | Can generated summaries be given citations that verifiably support them? |
| RQ5 | Does query decomposition improve retrieval for multi-issue questions? |
| RQ6 | How robust is the system to jurisdiction changes, temporal differences and misleading citations? |
| RQ7 | Can the system reliably abstain when the retrieved evidence is insufficient? |

## Key findings

<!-- findings:start -->
From the runs reported under [Results](#results): 2,000 test queries, small models on a
laptop. Several results go against the usual expectation and are reported as measured.

1. **The task is hard, and lexical retrieval is a strong baseline.** BM25 places the
   case the court quoted in the top 10 for 30.5% of queries and in the top 100 for
   69.4% (nDCG@10 0.176). A small general-purpose dense encoder alone is worse (0.152).
   Fusing the two by normalised score gives 0.186, a small but significant gain;
   reciprocal rank fusion is not distinguishable from BM25.
2. **A small web-trained reranker made ranking worse.** Rescoring with
   `ms-marco-MiniLM-L6-v2` lowered nDCG@10 from 0.186 to 0.160 over the hybrid and from
   0.176 to 0.153 over BM25.
3. **Section-aware chunking did not improve retrieval.** Fixed windows found the
   quoted paragraph more often, including in a control matched for passage length.
4. **Query decomposition had no effect with BM25**, even when the query was split at
   its true issue boundaries.
5. **Ignoring dates distorts results badly on older queries.** For queries from 1975
   to 1999, 52% of unfiltered top-10 results were cases decided after the query, and
   Recall@10 fell from 0.466 to 0.335.
6. **Jurisdiction is a strong signal.** For 94% of queries the quoted case was binding
   on the citing court. Restricting retrieval to binding authority raised Recall@10
   from 0.305 to 0.456, and made the remaining 6% of cases unreachable.
7. **Citations left in the query did not leak the answer.** They lowered BM25's score,
   because an opinion's own citation is not in its indexed text and the string matches
   other opinions that cite the same case.
8. **Existence checks are reliable; support checks are not yet.** Fabricated
   citations, wrong case names and invalid passage labels were all detected. For
   substantive support, a lexical judge detected 2% of negated claims and never flagged
   a correct one, while an entailment judge detected 94% and flagged 21% of correct
   ones.
9. **Verbatim extraction never invents a citation and never abstains.** A 1.1B local
   chat model invented citations in 31% of answers; all were caught, and under strict
   verification the system abstained on every query.
10. **The retrieval score cannot tell that the right case is missing.** With the quoted
    case withheld, a threshold on the best score separates answerable from unanswerable
    queries at chance (AUROC 0.50 for BM25, 0.51 for a cross-encoder). Questions outside
    federal law altogether are detected well (AUROC 0.99, and 100% with the scope rule).

11. **Legal-domain fine-tuning helps dense retrieval, and the gain vanishes in the
    hybrid.** Run on a Kaggle T4 under identical conditions, the legal encoder scored
    nDCG@10 0.172 against 0.155 for its general-purpose parent (+0.016, 95% interval
    0.007 to 0.025) and Recall@100 0.733 against 0.679. Fused with BM25 the two were
    indistinguishable (0.189 against 0.186). The legal encoder alone did not beat BM25
    (0.172 against 0.176), and it did not find the quoted paragraph more often than the
    general one. Its training data may include opinions in this corpus, so part of the
    dense gain may be exposure and not domain adaptation.
<!-- findings:end -->

## Architecture

```
question
   │
   ▼
query interpretation ── court and date constraints, out-of-scope detection
   │
   ▼
first-stage retrieval ── BM25 │ dense │ hybrid (rank or score fusion)
   │                      masks: decided before a date, binding on a circuit
   ▼
cross-encoder reranking (optional)
   │
   ▼
sufficiency gate ── abstain if nothing eligible or the best score is too low
   │
   ▼
evidence selection ── top passages, capped per case, labelled with court, date,
   │                  opinion type, binding or persuasive, known negative treatment
   ▼
generation ── typed statements: fact │ reasoning │ provision │ interpretation │
   │          not established; cited by passage label
   ▼
citation verification ── marker validity, existence, provenance, name agreement,
   │                     quotation fidelity, substantive support
   ▼
answer with evidence, verification results and a downloadable report
```

Every stage after BM25 is optional. The default configuration uses BM25, verbatim
extraction and a lexical support check, and runs with no model download and no
network access.

## Data

| Source | Role | Licence |
|---|---|---|
| [LePaRD](https://huggingface.co/datasets/rmahari/LePaRD) (Mahari et al., 2023), `top_10000` file at a pinned revision | Queries and relevance labels: each row records that a federal judge quoted a passage of an earlier case, with the text that precedes the quotation | CC BY-NC-SA 4.0 |
| [Caselaw Access Project](https://case.law) (Harvard Law School Library Innovation Lab) | Full text of the 4,940 quoted opinions, with paragraphs, opinion types, courts, dates and citations | CC0 1.0 |

The jurisdiction is U.S. federal courts only: 2,108 Supreme Court decisions, 2,741
court of appeals decisions and 91 from district and specialised courts, 38.5 million
words in all. No state or foreign law is included.

The retrieval task is the one LePaRD defines. The query is the passage of an opinion
that leads up to a quotation; the relevant case is the one quoted; the relevant
paragraph is the one the quotation comes from. Relevance is therefore decided by
courts, not by annotators or a model.

Neither dataset is stored in this repository. `legalrag fetch` downloads both and
verifies the benchmark file's checksum. Query files derived from LePaRD inherit its
non-commercial, share-alike terms and are not committed.

Why these two and not COLIEE, CaseHOLD or CLERC is recorded in
[docs/datasets.md](docs/datasets.md).

## Methods

Full detail is in [docs/methodology.md](docs/methodology.md).

**Document processing.** Opinions are parsed into paragraphs with stable identifiers
of the form `cap:<case>:<opinion>:<paragraph>`. Majority opinions, concurrences and
dissents are kept apart. Section headings are detected and each paragraph records its
heading path. Reporter citations are parsed into volume, reporter and page with
spelling variants normalised. Two chunking schemes are built: fixed 220-word windows,
and section-aware passages that never cross a heading or an opinion boundary and never
split a paragraph. Indexes are versioned by a hash of the corpus and refuse to load
against a different one.

**Benchmark construction.** The text before a quotation often names the answer. Rows
that repeat the quotation are dropped; reporter citations, the case names attached to
them, and short-form references to the quoted case are masked. Queries are sampled
with a cap per passage and one per citing opinion.

**Retrieval.** BM25 with parameters re-weightable after indexing; exact dense search
over normalised embeddings; reciprocal rank fusion and min-max score fusion; a
cross-encoder over the top candidates. Date and court restrictions are masks applied
before the top-k cut.

**Jurisdiction.** The federal hierarchy is modelled explicitly. A Supreme Court
decision binds every federal court; a court of appeals decision binds its own circuit.
Retrieval can be restricted to authority binding on a forum, and each result is
labelled binding or persuasive.

**Generation.** An answer is a list of typed statements. Facts, judicial reasoning and
legal provisions must cite an evidence passage. The system's own synthesis is typed as
interpretation and never presented as authority. Parts of the question the evidence
does not reach are listed as not established.

**Citation verification.** Six separate checks: that a cited passage label exists;
that a written reporter citation is in the corpus; that the cited case was among the
supplied passages; that the case name matches the citation; that quoted text occurs in
the cited passage; and that the passage supports the statement. The first five are
exact lookups. The sixth is an automatic judgment by a lexical or natural-language-
inference judge and is labelled as such wherever it is shown. A citation that exists is
not thereby treated as a citation that supports.

## Experimental protocol

| | |
|---|---|
| Splits | By the decision date of the citing opinion: development 2000 to 2009 (800 queries), test 2010 to 2018 (2,000 queries), and 1975 to 1999 (800 queries) for the temporal experiment only |
| Tuning | Every hyperparameter is chosen on development queries and then fixed |
| As-of retrieval | A query may only retrieve cases decided before its own date |
| Units | Case-level metrics rank each case by its best passage. Paragraph-level hits require a returned passage to contain the quoted paragraph |
| Uncertainty | 95% bootstrap intervals over queries; paired sign-flip randomisation tests with Holm adjustment for comparisons; Wilson intervals for small samples |
| Provenance | Every result file records its configuration, corpus version, package versions, device and Git commit |

Only the quoted case counts as relevant. Other opinions that state the same rule are
scored as misses, so all retrieval figures are lower bounds on usefulness.

## Results

All figures below are generated from the files under `results/` by
`scripts/make_results_tables.py`.

### Evaluation status

| Status | What | Where |
|---|---|---|
| Measured locally | Retrieval with BM25 and `bge-small-en-v1.5`, reranking, chunking, query decomposition, temporal and jurisdiction experiments, abstention, judge and citation-checker validation, generation with the extractive generator and a 1.1B local model, latency | Tables below; Apple M3 Pro, small models, corpus version `c1b900e309df` |
| Checked end to end, not a measurement | One question answered on the full 4,940-opinion corpus with the default configuration: constraint parsing, date and binding-authority filters, evidence selection, extraction, citation checks and the report | `legalrag ask`, see [Usage](#usage) |
| Measured on a Kaggle GPU | RQ2: the legal-domain encoder against its general-purpose parent, on the same passages, queries and labels as the local runs | Table below; Tesla T4, commit `9ecd6f7`, provenance under `results/encoders/kaggle_run_v2/` |
| Pending | A larger reranker, the large entailment judge, and generation with a capable open model | Prepared for the same Kaggle runner; [kaggle/README.md](kaggle/README.md) |

A pending row has no numbers anywhere in this repository. When a pending experiment
is run, its result file is added under `results/` and its table appears below.

<!-- results:start -->
### Retrieval: BM25, dense and hybrid (RQ1)

2,000 test queries over 222,520 passages; settings tuned on 800 development queries. Case-level metrics unless marked. Brackets are 95% bootstrap intervals.

| System | Recall@1 | Recall@10 | Recall@100 | MRR@10 | nDCG@10 | P@10 | Passage hit@10 |
|---|---|---|---|---|---|---|---|
| BM25 | 0.072 [0.061, 0.084] | 0.305 [0.285, 0.326] | 0.694 [0.674, 0.715] | 0.136 [0.125, 0.149] | 0.176 [0.163, 0.190] | 0.030 [0.029, 0.033] | 0.127 [0.112, 0.142] |
| Dense, bge-small-en-v1.5 (general) | 0.062 [0.052, 0.073] | 0.273 [0.254, 0.294] | 0.649 [0.628, 0.670] | 0.116 [0.105, 0.127] | 0.152 [0.140, 0.165] | 0.027 [0.025, 0.029] | 0.101 [0.088, 0.114] |
| Hybrid, rank fusion: BM25 + bge-small | 0.076 [0.066, 0.088] | 0.315 [0.294, 0.337] | 0.698 [0.678, 0.717] | 0.141 [0.129, 0.154] | 0.181 [0.168, 0.196] | 0.032 [0.029, 0.034] | 0.133 [0.118, 0.150] |
| Hybrid, score fusion: BM25 + bge-small | 0.083 [0.071, 0.096] | 0.315 [0.294, 0.336] | 0.703 [0.683, 0.723] | 0.146 [0.134, 0.160] | 0.186 [0.173, 0.200] | 0.032 [0.029, 0.034] | 0.139 [0.124, 0.155] |

Tuned BM25: k1 = 1.6, b = 0.75.

Paired differences in nDCG@10 unless marked (Holm-adjusted):

| Comparison | Difference |
|---|---|
| dense-bge-small vs bm25 | -0.024 [-0.035, -0.012], p < 0.001 |
| hybrid-rrf-bge-small vs bm25 | +0.005 [-0.003, +0.014], p = 0.255 |
| hybrid-score-bge-small vs bm25 | +0.010 [+0.005, +0.015], p = 0.002 |
| hybrid-score-bge-small vs hybrid-rrf-bge-small | +0.004 [-0.001, +0.010], p = 0.255 |

**Leakage control.** The same queries with citations and case names left in:

| System | nDCG@10 masked | nDCG@10 unmasked | Unmasked minus masked |
|---|---|---|---|
| BM25 | 0.176 | 0.158 | -0.018 [-0.025, -0.011], p < 0.001 |
| Dense, bge-small-en-v1.5 (general) | 0.152 | 0.154 | +0.002 [-0.004, +0.008], p = 0.571 |

### Legal against general embedding model (RQ2)

2,000 test queries over 222,520 passages (corpus version `c1b900e309df`); hybrid weight tuned on 800 development queries. Both encoders at 512 tokens, float16, batch size 128, on cuda.

| Encoder | Model | Revision | Domain |
|---|---|---|---|
| modernbert-general | `nomic-ai/modernbert-embed-base` | `d556a88e3325` | general |
| modernbert-legal | `freelawproject/modernbert-embed-base_finetune_512` | `04f0141fbc04` | legal |

| System | Recall@1 | Recall@10 | Recall@100 | MRR@10 | nDCG@10 | Passage hit@10 |
|---|---|---|---|---|---|---|
| BM25 | 0.072 [0.061, 0.084] | 0.305 [0.285, 0.326] | 0.694 [0.674, 0.715] | 0.137 [0.125, 0.149] | 0.176 [0.163, 0.190] | 0.127 [0.112, 0.142] |
| Dense, modernbert-embed-base (general) | 0.065 [0.054, 0.075] | 0.275 [0.255, 0.295] | 0.679 [0.658, 0.700] | 0.119 [0.108, 0.132] | 0.155 [0.143, 0.169] | 0.108 [0.095, 0.122] |
| Hybrid, BM25 + modernbert-embed-base (general) | 0.082 [0.070, 0.094] | 0.320 [0.299, 0.341] | 0.715 [0.696, 0.735] | 0.145 [0.132, 0.158] | 0.186 [0.172, 0.200] | 0.145 [0.129, 0.162] |
| Dense, modernbert-embed-base legal fine-tune | 0.064 [0.053, 0.074] | 0.312 [0.292, 0.333] | 0.733 [0.714, 0.753] | 0.129 [0.117, 0.140] | 0.172 [0.159, 0.185] | 0.110 [0.098, 0.124] |
| Hybrid, BM25 + modernbert-embed-base legal fine-tune | 0.077 [0.066, 0.089] | 0.333 [0.311, 0.353] | 0.729 [0.709, 0.748] | 0.145 [0.133, 0.157] | 0.189 [0.175, 0.202] | 0.140 [0.124, 0.156] |

**Dense alone: legal minus general** (paired, Holm-adjusted):

| Metric | Difference |
|---|---|
| ndcg@10 | +0.016 [+0.007, +0.025], p = 0.004 |
| mrr@10 | +0.009 [+0.001, +0.018], p = 0.145 |
| recall@1 | -0.001 [-0.011, +0.009], p = 1.000 |
| recall@10 | +0.037 [+0.021, +0.054], p < 0.001 |
| recall@100 | +0.054 [+0.037, +0.070], p < 0.001 |
| passage_hit@10 | +0.003 [-0.009, +0.015], p = 1.000 |

**Hybrid with BM25: legal minus general** (paired, Holm-adjusted):

| Metric | Difference |
|---|---|
| ndcg@10 | +0.003 [-0.003, +0.009], p = 1.000 |
| mrr@10 | -0.000 [-0.007, +0.006], p = 1.000 |
| recall@1 | -0.004 [-0.013, +0.003], p = 1.000 |
| recall@10 | +0.013 [-0.000, +0.025], p = 0.296 |
| recall@100 | +0.013 [+0.002, +0.025], p = 0.182 |
| passage_hit@10 | -0.005 [-0.013, +0.003], p = 0.934 |

**Per-query differences** (legal minus general):

| System | Metric | Legal better | Tied | General better | 5th percentile | 95th percentile |
|---|---|---|---|---|---|---|
| Dense | ndcg@10 | 344 | 1405 | 251 | -0.356 | +0.387 |
| Dense | recall@10 | 179 | 1717 | 104 | -1.000 | +1.000 |
| Hybrid | ndcg@10 | 257 | 1504 | 239 | -0.289 | +0.301 |
| Hybrid | recall@10 | 94 | 1837 | 69 | +0.000 | +0.000 |

| Encoder | Corpus encoding | Passages/s | Peak GPU, corpus | Peak GPU, queries | Embeddings on disk | Query encoding ms (p50 / p95) | Search ms (p50 / p95) | Passages over the token limit |
|---|---|---|---|---|---|---|---|---|
| modernbert-general | 42.9 min | 86 | 1904 MB | 306 MB | 326 MB | 29.1 / 31.3 | 27.0 / 28.3 | 373 |
| modernbert-legal | 43.0 min | 86 | 1904 MB | 598 MB | 326 MB | 30.0 / 33.1 | 26.6 / 27.7 | 373 |

Peak GPU memory for query encoding is measured without unloading the encoder evaluated before it, so the second figure includes the first model.

The legal model was fine-tuned on synthetic queries over U.S. court opinions that may include opinions in this corpus, so a difference in its favour can reflect exposure to the documents as well as domain adaptation.

### Cross-encoder reranking (RQ3)

| System | Recall@10 | MRR@10 | nDCG@10 | Passage hit@10 | Added latency per query |
|---|---|---|---|---|---|
| First stage: BM25 | 0.305 [0.285, 0.326] | 0.136 [0.125, 0.149] | 0.176 [0.163, 0.190] | 0.127 [0.112, 0.142] | n/a |
| + cross-encoder/ms-marco-MiniLM-L6-v2 on top 20 | 0.285 [0.266, 0.306] | 0.113 [0.101, 0.123] | 0.153 [0.141, 0.165] | 0.106 [0.092, 0.120] | 102 ms |

Device: mps. Depth chosen on 300 development queries from 20: nDCG@10 0.188, 50: nDCG@10 0.184, 100: nDCG@10 0.168. Share of queries whose case is within the reranked passages: 0.356.

| Comparison | Difference |
|---|---|
| reranked vs first stage [ndcg@10] | -0.024 [-0.032, -0.015], p < 0.001 |
| reranked vs first stage [recall@10] | -0.020 [-0.032, -0.009], p = 0.002 |
| reranked vs first stage [mrr@10] | -0.024 [-0.033, -0.015], p < 0.001 |
| reranked vs first stage [passage_hit@10] | -0.021 [-0.032, -0.010], p < 0.001 |

| System | Recall@10 | MRR@10 | nDCG@10 | Passage hit@10 | Added latency per query |
|---|---|---|---|---|---|
| First stage: Hybrid, score fusion: BM25 + bge-small | 0.315 [0.294, 0.336] | 0.146 [0.134, 0.160] | 0.186 [0.173, 0.200] | 0.139 [0.124, 0.155] | n/a |
| + cross-encoder/ms-marco-MiniLM-L6-v2 on top 20 | 0.300 [0.280, 0.321] | 0.117 [0.106, 0.128] | 0.160 [0.147, 0.172] | 0.114 [0.099, 0.128] | 103 ms |

Device: mps. Depth chosen on 300 development queries from 20: nDCG@10 0.188, 50: nDCG@10 0.184, 100: nDCG@10 0.176. Share of queries whose case is within the reranked passages: 0.369.

| Comparison | Difference |
|---|---|
| reranked vs first stage [ndcg@10] | -0.026 [-0.035, -0.017], p < 0.001 |
| reranked vs first stage [recall@10] | -0.015 [-0.028, -0.002], p = 0.025 |
| reranked vs first stage [mrr@10] | -0.029 [-0.039, -0.019], p < 0.001 |
| reranked vs first stage [passage_hit@10] | -0.025 [-0.037, -0.014], p < 0.001 |

### Fixed-size against section-aware chunking

| Scheme | Passages | Words in top 10 | nDCG@10 | Recall@10 | Passage hit@10 | Passage hits per 1,000 words read |
|---|---|---|---|---|---|---|
| section220 | 222,520 | 1662 | 0.173 [0.161, 0.186] | 0.302 [0.282, 0.323] | 0.126 [0.112, 0.141] | 0.0761 |
| fixed220 | 209,532 | 2194 | 0.185 [0.172, 0.199] | 0.314 [0.293, 0.334] | 0.160 [0.143, 0.175] | 0.0727 |
| fixed170 | 268,990 | 1697 | 0.182 [0.168, 0.196] | 0.310 [0.289, 0.331] | 0.145 [0.130, 0.161] | 0.0857 |

| Comparison | Difference |
|---|---|
| section220 vs fixed220 [ndcg@10] | -0.012 [-0.020, -0.004], p = 0.014 |
| section220 vs fixed220 [recall@10] | -0.011 [-0.025, +0.002], p = 0.213 |
| section220 vs fixed220 [passage_hit@5] | -0.024 [-0.035, -0.013], p = 0.001 |
| section220 vs fixed220 [passage_hit@10] | -0.033 [-0.046, -0.022], p = 0.001 |
| section220 vs fixed220 [passage_mrr@10] | -0.020 [-0.027, -0.013], p = 0.001 |
| section220 vs fixed170 [ndcg@10] | -0.009 [-0.017, -0.001], p = 0.084 |
| section220 vs fixed170 [recall@10] | -0.008 [-0.021, +0.005], p = 0.256 |
| section220 vs fixed170 [passage_hit@5] | -0.013 [-0.022, -0.003], p = 0.084 |
| section220 vs fixed170 [passage_hit@10] | -0.019 [-0.032, -0.006], p = 0.014 |
| section220 vs fixed170 [passage_mrr@10] | -0.016 [-0.022, -0.009], p = 0.001 |

### Query decomposition (RQ5)

**Multi-issue queries** (n = 600):

| Condition | Sub-queries | Recall@10 | nDCG@10 | All relevant cases in top 20 |
|---|---|---|---|---|
| none | 1.0 | 0.253 [0.228, 0.279] | 0.188 [0.168, 0.208] | 0.138 [0.110, 0.168] |
| segments | 4.8 | 0.247 [0.221, 0.272] | 0.178 [0.159, 0.198] | 0.132 [0.105, 0.162] |
| sentences | 4.6 | 0.244 [0.219, 0.270] | 0.174 [0.155, 0.194] | 0.138 [0.110, 0.168] |
| oracle | 3.5 | 0.254 [0.229, 0.278] | 0.180 [0.161, 0.201] | 0.147 [0.118, 0.177] |

| Comparison | Difference |
|---|---|
| segments vs none [recall@10] | -0.007 [-0.020, +0.006], p = 0.682 |
| segments vs none [ndcg@10] | -0.009 [-0.021, +0.001], p = 0.411 |
| sentences vs none [recall@10] | -0.009 [-0.023, +0.005], p = 0.672 |
| sentences vs none [ndcg@10] | -0.013 [-0.024, -0.003], p = 0.077 |
| oracle vs none [recall@10] | +0.000 [-0.011, +0.012], p = 1.000 |
| oracle vs none [ndcg@10] | -0.007 [-0.016, +0.001], p = 0.411 |

**Single-issue queries** (n = 1000):

| Condition | Sub-queries | Recall@10 | nDCG@10 | All relevant cases in top 20 |
|---|---|---|---|---|
| none | 1.0 | 0.287 [0.261, 0.316] | 0.164 [0.147, 0.183] | 0.397 [0.368, 0.426] |
| segments | 3.6 | 0.276 [0.251, 0.304] | 0.157 [0.139, 0.174] | 0.390 [0.362, 0.420] |
| sentences | 3.6 | 0.292 [0.266, 0.320] | 0.161 [0.144, 0.179] | 0.395 [0.367, 0.425] |

| Comparison | Difference |
|---|---|
| segments vs none [recall@10] | -0.011 [-0.024, +0.001], p = 0.380 |
| segments vs none [ndcg@10] | -0.008 [-0.016, -0.001], p = 0.173 |
| sentences vs none [recall@10] | +0.005 [-0.010, +0.019], p = 0.961 |
| sentences vs none [ndcg@10] | -0.003 [-0.011, +0.005], p = 0.961 |

### Time: future-case leakage and outdated authority (RQ6)

| Queries | Corpus decided before the query | Recall@10 with as-of filter | Recall@10 without | Top-10 results from the query's future | Top result from the future |
|---|---|---|---|---|---|
| history (1975 to 1999), n = 800 | 49% | 0.466 | 0.335 | 0.522 [0.506, 0.540] | 0.489 [0.454, 0.524] |
| test (2010 to 2018), n = 2000 | 99% | 0.305 | 0.303 | 0.016 [0.014, 0.019] | 0.022 [0.016, 0.029] |

**Outdated authority.** Of 2,000 test queries, 1.3% had in their top 10 a case that the treatment table records as overruled or abrogated before the query date, and 0.6% a case overruled only afterwards. The table covers 9 corpus cases, so these are lower bounds. In 12 queries the case the court itself quoted had already received negative treatment.

### Jurisdiction (RQ6)

For 94.2% of test queries the case the court quoted was binding on that court (a Supreme Court decision or one of its own circuit).

| Retrieval pool | Queries | Recall@10 |
|---|---|---|
| No court filter | all, n = 1854 | 0.305 [0.285, 0.327] |
| Binding authority only | all, n = 1854 | 0.456 [0.433, 0.478] |
| No court filter | quoted case is binding, n = 1746 | 0.298 [0.278, 0.321] |
| Binding authority only | quoted case is binding, n = 1746 | 0.484 [0.461, 0.507] |
| No court filter | quoted case is persuasive, n = 108 | 0.417 |
| Binding authority only | quoted case is persuasive, n = 108 | 0.000 |
| Another circuit's court of appeals only | all, n = 1854 | 0.002 |

Under the wrong-circuit filter the best retrieval score separates matched from mismatched searches with AUROC 0.70.

**Scope detection** on hand-written questions:

| Question group | n | Flagged as outside the corpus |
|---|---|---|
| state | 15 | 15 (100%) |
| foreign | 15 | 15 (100%) |
| non legal | 10 | 0 (0%) |
| in scope (false alarms) | 30 | 1 (3%) |

### Grounded generation and citation quality (RQ4)

**Generator `extractive`**, retrieval bm25, 300 test queries, 5 evidence passages. Support judged by `nli:cross-encoder/nli-deberta-v3-xsmall`; strict filter uses `lexical-containment`. Generation: 1 ms median.

| Measure | normal, verification off | normal, verification strict | withheld, verification off | withheld, verification strict |
|---|---|---|---|---|
| Abstained | 0.00 [0.00, 0.01] | 0.00 [0.00, 0.01] | 0.00 [0.00, 0.01] | 0.00 [0.00, 0.01] |
| Citation precision | 0.80 [0.78, 0.82] | 0.80 [0.78, 0.82] | 0.80 [0.78, 0.82] | 0.80 [0.78, 0.82] |
| Citation recall | 0.80 [0.78, 0.82] | 0.80 [0.78, 0.82] | 0.80 [0.78, 0.82] | 0.80 [0.78, 0.82] |
| Unsupported-claim rate | 0.20 [0.18, 0.22] | 0.20 [0.18, 0.22] | 0.20 [0.18, 0.22] | 0.20 [0.18, 0.22] |
| Answers with a hallucinated citation | 0.00 [0.00, 0.01] | 0.00 [0.00, 0.01] | 0.00 [0.00, 0.01] | 0.00 [0.00, 0.01] |
| Cites the quoted case | 0.19 [0.15, 0.23] | 0.19 [0.15, 0.23] | 0.00 [0.00, 0.01] | 0.00 [0.00, 0.01] |
| Cites the quoted passage | 0.09 [0.06, 0.13] | 0.09 [0.06, 0.13] | 0.00 [0.00, 0.01] | 0.00 [0.00, 0.01] |

Extractive statements are sentences copied from the passage they cite, so their true citation precision is 1. The figure shown is the share the entailment judge confirmed, which measures the judge. Strict filtering with the lexical judge removes nothing from verbatim text, so the columns match.

**Generator `llm:TinyLlama/TinyLlama-1.1B-Chat-v1.0`**, retrieval bm25, 100 test queries, 3 evidence passages. Support judged by `nli:cross-encoder/nli-deberta-v3-xsmall`; strict filter uses `lexical-containment`. Generation: 9016 ms median.

| Measure | normal, verification off | normal, verification strict | withheld, verification off | withheld, verification strict |
|---|---|---|---|---|
| Abstained | 0.00 [0.00, 0.04] | 1.00 [0.96, 1.00] | 0.00 [0.00, 0.04] | 1.00 [0.96, 1.00] |
| Citation precision | n/a | n/a | n/a | n/a |
| Citation recall | 0.00 [0.00, 0.00] | n/a | 0.00 [0.00, 0.00] | n/a |
| Unsupported-claim rate | 1.00 [1.00, 1.00] | n/a | 1.00 [1.00, 1.00] | n/a |
| Answers with a hallucinated citation | 0.31 [0.23, 0.41] | n/a | 0.31 [0.23, 0.41] | n/a |
| Cites the quoted case | 0.09 [0.05, 0.16] | n/a | 0.00 [0.00, 0.04] | n/a |
| Cites the quoted passage | 0.03 [0.01, 0.08] | n/a | 0.00 [0.00, 0.04] | n/a |

In the withheld condition the case the court quoted is removed before retrieval. "n/a" marks a measure with no statements left to score.

### Citation verification

**Support judges** (300 test pairs of each kind). A good judge accepts a true quotation and rejects the other three.

| Judge | Threshold | Accepts true quotation | Accepts: other case | Accepts: same case, wrong passage | Accepts: negated | Accuracy | Precision of 'supported' |
|---|---|---|---|---|---|---|---|
| lexical-containment | 0.60 (default) | 0.99 [0.97, 0.99] | 0.00 | 0.04 | 0.99 | 0.74 | 0.49 |
| lexical-containment | 0.78 (tuned on development pairs) | 0.96 [0.93, 0.98] | 0.00 | 0.01 | 0.96 | 0.75 | 0.50 |
| nli:cross-encoder/nli-deberta-v3-xsmall | 0.50 (default) | 0.69 [0.63, 0.74] | 0.01 | 0.04 | 0.15 | 0.87 | 0.77 |
| nli:cross-encoder/nli-deberta-v3-xsmall | 0.51 (tuned on development pairs) | 0.69 [0.63, 0.74] | 0.01 | 0.04 | 0.15 | 0.87 | 0.77 |
| nli:MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli | 0.50 (default) | 0.90 [0.86, 0.93] | 0.21 | 0.30 | 0.31 | 0.77 | 0.52 |
| nli:MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli | 0.95 (tuned on development pairs) | 0.65 [0.60, 0.70] | 0.02 | 0.10 | 0.10 | 0.86 | 0.75 |

**Planted citation errors**, judge `lexical-containment`. False alarms on 1,447 clean verbatim statements: 0.0%.

| Planted error | Check | n | Detected |
|---|---|---|---|
| fabricated citation | exact lookup | 1447 | 1.000 [0.997, 1.000] |
| real but unretrieved | exact lookup | 1447 | 0.999 [0.996, 1.000] |
| wrong name | exact lookup | 1447 | 1.000 [0.997, 1.000] |
| marker out of range | exact lookup | 1447 | 1.000 [0.997, 1.000] |
| altered quotation | exact lookup | 1432 | 0.941 [0.927, 0.952] |
| swapped marker | support judge | 1447 | 0.747 [0.724, 0.769] |
| swapped claim | support judge | 1447 | 1.000 [0.997, 1.000] |
| negated | support judge | 965 | 0.021 [0.013, 0.032] |

**Planted citation errors**, judge `nli:cross-encoder/nli-deberta-v3-xsmall`. False alarms on 733 clean verbatim statements: 20.9%.

| Planted error | Check | n | Detected |
|---|---|---|---|
| fabricated citation | exact lookup | 733 | 1.000 [0.995, 1.000] |
| real but unretrieved | exact lookup | 733 | 0.999 [0.992, 1.000] |
| wrong name | exact lookup | 733 | 1.000 [0.995, 1.000] |
| marker out of range | exact lookup | 733 | 1.000 [0.995, 1.000] |
| altered quotation | exact lookup | 724 | 0.943 [0.924, 0.958] |
| swapped marker | support judge | 733 | 0.869 [0.843, 0.891] |
| swapped claim | support judge | 733 | 0.995 [0.986, 0.998] |
| negated | support judge | 501 | 0.942 [0.918, 0.959] |

### Abstention (RQ7)

Signal: cross-encoder score of the best passage (minilm-l6). Threshold tuned on 600 development queries; 600 test queries.

| Condition in which the quoted case is unreachable | n | AUROC | Abstained (correct) | Answerable queries still answered |
|---|---|---|---|---|
| quoted case removed from the pool | 600 | 0.51 | 36% | 64% |
| as-of date moved before the quoted case | 600 | 0.59 | 48% | 64% |
| another circuit's court of appeals searched | 600 | 0.68 | 63% | 64% |
| state-law, foreign-law and non-legal questions | 40 | 0.79 | 78% | 64% |

The same score predicts whether the quoted case is in the top five with AUROC 0.58.

Signal: BM25 score of the best passage. Threshold tuned on 600 development queries; 600 test queries.

| Condition in which the quoted case is unreachable | n | AUROC | Abstained (correct) | Answerable queries still answered |
|---|---|---|---|---|
| quoted case removed from the pool | 600 | 0.50 | 68% | 33% |
| as-of date moved before the quoted case | 600 | 0.59 | 77% | 33% |
| another circuit's court of appeals searched | 600 | 0.69 | 88% | 33% |
| state-law, foreign-law and non-legal questions | 40 | 0.99 | 100% | 33% |

The same score predicts whether the quoted case is in the top five with AUROC 0.51.

On the out-of-scope questions the scope rule alone abstains on 75%; rule or score together on 100%.

### Latency and resources

Per query, one at a time, on macOS-26.6.2-arm64-arm-64bit (mps), 200 queries.

| Stage | Mean ms | Median ms | p95 ms |
|---|---|---|---|
| bm25 retrieval | 3.5 | 3.6 | 5.1 |
| dense query encoding | 10.2 | 9.4 | 13.2 |
| dense search exact | 5.1 | 5.0 | 5.7 |
| rank fusion | 0.1 | 0.1 | 0.1 |
| reranking top50 | 264.1 | 258.0 | 304.6 |
| evidence selection | 0.2 | 0.2 | 0.8 |
| generation extractive | 0.9 | 0.9 | 1.1 |
| citation verification[nli:cross-encoder/nli-deberta-v3-xsmall] | 807.4 | 756.0 | 1411.3 |

| Index | Disk MB | Build rate |
|---|---|---|
| bm25 | 38 | n/a |
| dense-bge-small | 171 | 180 passages/s on mps |

Storage: raw opinions 312 MB, processed corpus 1282 MB. Peak resident memory of the run: 2.5 GB.
<!-- results:end -->

Error patterns are analysed in [docs/error_analysis.md](docs/error_analysis.md).

## Limitations

- **Not legal advice, and no authority checking.** The system does not know whether a
  case has been overruled, reversed or superseded. A table of 18 well-known Supreme
  Court overrulings is used to attach warnings to a few cases; it is illustrative and
  far from complete. The corpus ends in 2016.
- **A small, closed corpus.** 4,940 opinions, all of them frequently quoted. Absence
  from the corpus says nothing about whether a case exists.
- **One jurisdiction.** U.S. federal courts. Questions about state or foreign law are
  detected by rule and declined; the detector was tested only on a small hand-written
  set.
- **Incomplete relevance labels.** Only the case a court quoted is counted as relevant.
- **Queries are passages of judicial opinions**, not questions typed by lawyers.
  Performance on short natural-language questions has not been measured.
- **Automatic support judgments are fallible.** The lexical judge cannot detect
  negation. Entailment models are trained on general text and were validated here only
  on near-verbatim quotations, which is easier than paraphrase.
- **No reference answers.** Generation quality is measured by citation support and by
  proximity to what the court quoted, not by legal correctness.
- **RQ2 rests on one model pair.** One legal fine-tune was compared with its parent,
  on one corpus. The fine-tune's training opinions may overlap this corpus, which the
  experiment cannot rule out, and other legal encoders may behave differently.
- **Small local models.** The results reported use a 33M-parameter encoder, a
  22M-parameter reranker trained on web search, and a 1.1B-parameter chat model. The
  negative results for reranking and for model-based generation describe those
  models, not the techniques.

## Installation

Python 3.11 or later. With [uv](https://docs.astral.sh/uv/):

```bash
uv sync --extra app --extra dev              # BM25 pipeline, API, interface, tests
uv sync --extra app --extra dev --extra ml   # adds dense encoders, rerankers, NLI judges
uv sync --extra api                          # the HTTP API alone, as in the container image
```

The core package needs NumPy, SciPy, Pydantic, PyYAML and httpx. The `ml` extra adds
PyTorch, sentence-transformers and transformers.

## Usage

Without any data the system runs on a small synthetic corpus, in which every case is
invented and labelled as such:

```bash
legalrag ask "When does a government official lose qualified immunity?"
```

To build the real corpus (about 540 MB for the benchmark file and 300 MB for the
opinions):

```bash
legalrag fetch              # download LePaRD (checksum verified) and the cited opinions
legalrag build              # parse, chunk, sample queries, write the manifest
legalrag index bm25         # about one minute
legalrag index bge-small    # optional dense index, about 20 minutes on Apple silicon
```

Court and date constraints can be written into the question:

```bash
legalrag ask "preliminary injunction standard in the Ninth Circuit before 2008"
```

### API

```bash
legalrag serve              # http://127.0.0.1:8000, interactive docs at /docs
```

| Endpoint | Purpose |
|---|---|
| `GET /health` | Corpus, configuration and the limitation notice |
| `POST /search` | Ranked cases with court, date and authority labels |
| `POST /answer` | Typed, cited statements with evidence and verification results |
| `POST /report` | The same as a Markdown report to download |
| `POST /citations/validate` | Existence check for reporter citations in any text |
| `GET /cases/{case_id}` | One opinion by paragraph |

```bash
curl -s localhost:8000/answer -H 'content-type: application/json' \
  -d '{"question": "What must a plaintiff show to have standing?", "binding_on": "9", "before": "2010-01-01"}'
```

The pipeline is selected with `LEGALRAG_CONFIG`, for example
`LEGALRAG_CONFIG=configs/local_neural.yaml legalrag serve`.

### Interface

```bash
streamlit run app/streamlit_app.py
```

Natural-language search, jurisdiction and date filters, ranked cases with binding or
persuasive labels, cited summaries grouped by statement type, evidence passages shown
in the context of the full opinion, a citation checker, and report download. The
limitation notice is displayed on every page.

### Container

```bash
docker build -t legal-research-rag .
docker run -p 8000:8000 legal-research-rag
```

The image (about 590 MB) serves the API on the synthetic corpus as a non-root user
and contains no dataset and no model weights.

## Reproducing the experiments

```bash
legalrag experiment retrieval
legalrag experiment rerank reranker=minilm-l6 encoder=bge-small fusion=score alpha=0.7 k1=1.6 b=0.75
legalrag experiment rerank reranker=minilm-l6 k1=1.6 b=0.75
legalrag experiment chunking
legalrag experiment decomposition k1=1.6 b=0.75
legalrag experiment temporal k1=1.6 b=0.75
legalrag experiment jurisdiction k1=1.6 b=0.75
legalrag experiment judge judges=lexical,nli-xsmall,nli-base
legalrag experiment citations judge=lexical k1=1.6 b=0.75
legalrag experiment citations judge=nli-xsmall limit=150 k1=1.6 b=0.75
legalrag experiment abstention k1=1.6 b=0.75
legalrag experiment abstention reranker=minilm-l6 k1=1.6 b=0.75
legalrag experiment rag generator=extractive eval_judge=nli-xsmall filter_judge=lexical k1=1.6 b=0.75
legalrag experiment rag generator=hf:TinyLlama/TinyLlama-1.1B-Chat-v1.0 limit=100 evidence_k=3 \
    eval_judge=nli-xsmall filter_judge=lexical k1=1.6 b=0.75 tag=tinyllama-1.1b
legalrag experiment analysis k1=1.6 b=0.75
legalrag experiment systems encoder=bge-small reranker=minilm-l6 judge=nli-xsmall
python scripts/make_results_tables.py
```

Each command writes a JSON file under `results/`. Experiments abort if their memory
footprint passes `LEGALRAG_MAX_GB` (default 10).

The legal-against-general encoder comparison runs on a Kaggle GPU. It checks
out one exact commit, installs from `uv.lock`, rebuilds the corpus and benchmark and
verifies them against `kaggle/expected_fingerprint.json`, encodes all 222,520 passages
with both models under identical settings, and exports results and provenance only:

```bash
python kaggle/prepare.py --username <kaggle-username> --out <staging folder outside the repo>
```

That command stages files locally and uploads nothing. The conditions held identical,
resource estimates and the remaining steps are in [kaggle/README.md](kaggle/README.md).
With both dense indexes present, the experiment itself is:

```bash
legalrag fingerprint --check kaggle/expected_fingerprint.json
legalrag experiment encoders general=modernbert-general legal=modernbert-legal k1=1.6 b=0.75
```

## Tests

```bash
uv run pytest
```

The suite runs on the synthetic corpus and needs no dataset, model, GPU or network. It
covers citation parsing, the court hierarchy, parsing and chunking, benchmark
construction and masking, retrieval and fusion, metrics and statistics, the pipeline,
abstention, citation verification, the experiment harness and the API.

## Repository structure

```
src/legalrag/
  data/          citation parsing, court hierarchy, opinion parsing, chunking,
                 duplicate detection, benchmark construction, known negative treatment
  index/         BM25, dense encoders, versioned chunk store
  retrieval/     ranking and fusion, reranking, query interpretation, decomposition
  rag/           evidence selection, generators, abstention, pipeline, report
  verify/        citation verifier and support judges
  eval/          ranking metrics and statistics
  experiments/   one module per experiment
  api/           FastAPI application and service functions
  demo.py        synthetic corpus for tests and the demonstration mode
app/             Streamlit interface
configs/         pipeline configurations
kaggle/          GPU runner, local staging script, expected corpus fingerprint
scripts/         result-table generator
docs/            dataset selection, methodology, error analysis
results/         result files (aggregate numbers only)
tests/
```

## References

- R. Mahari, D. Stammbach, E. Ash, A. Pentland. LePaRD: A Large-Scale Dataset of Judges
  Citing Precedents. arXiv:2311.09356, 2023.
- Harvard Law School Library Innovation Lab. Caselaw Access Project. https://case.law
- S. Robertson, H. Zaragoza. The Probabilistic Relevance Framework: BM25 and Beyond.
  Foundations and Trends in Information Retrieval, 2009.
- G. Cormack, C. Clarke, S. Büttcher. Reciprocal Rank Fusion Outperforms Condorcet and
  Individual Rank Learning Methods. SIGIR 2009.
- R. Nogueira, K. Cho. Passage Re-ranking with BERT. arXiv:1901.04085, 2019.
- BAAI. bge-small-en-v1.5 (model card). https://huggingface.co/BAAI/bge-small-en-v1.5
- V. Magesh et al. Hallucination-Free? Assessing the Reliability of Leading AI Legal
  Research Tools. Journal of Empirical Legal Studies, 2025.
- L. Zheng et al. When Does Pretraining Help? Assessing Self-Supervised Learning for Law
  and the CaseHOLD Dataset. ICAIL 2021.
- A. B. Hou et al. CLERC: A Dataset for Legal Case Retrieval and Retrieval-Augmented
  Analysis Generation. arXiv:2406.17186, 2024.

## License

Code: MIT. LePaRD is distributed by its authors under CC BY-NC-SA 4.0, which restricts
this benchmark to non-commercial use. Caselaw Access Project data is CC0.
