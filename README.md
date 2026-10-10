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
`scripts/make_results_tables.py`. Runs were made on an Apple M3 Pro laptop with small
models.

<!-- results:start -->
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
- **Small local models.** The results reported use a 33M-parameter encoder and a
  22M-parameter reranker.

## Installation

Python 3.11 or later. With [uv](https://docs.astral.sh/uv/):

```bash
uv sync --extra app --extra dev          # BM25 pipeline, API, interface, tests
uv sync --extra app --extra dev --extra ml   # adds dense encoders, rerankers, NLI judges
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

The image serves the API on the synthetic corpus and contains no dataset and no model
weights.

## Reproducing the experiments

```bash
legalrag experiment retrieval
legalrag experiment rerank reranker=minilm-l6 encoder=bge-small fusion=score alpha=0.7 k1=1.6 b=0.75
legalrag experiment chunking
legalrag experiment decomposition k1=1.6 b=0.75
legalrag experiment temporal k1=1.6 b=0.75
legalrag experiment jurisdiction k1=1.6 b=0.75
legalrag experiment judge judges=lexical,nli-xsmall,nli-base
legalrag experiment citations judge=nli-base limit=150 k1=1.6 b=0.75
legalrag experiment abstention reranker=minilm-l6 k1=1.6 b=0.75
legalrag experiment rag generator=extractive eval_judge=nli-base filter_judge=lexical k1=1.6 b=0.75
legalrag experiment analysis k1=1.6 b=0.75
legalrag experiment systems encoder=bge-small reranker=minilm-l6 judge=nli-base
python scripts/make_results_tables.py
```

Each command writes a JSON file under `results/`. Experiments abort if their memory
footprint passes `LEGALRAG_MAX_GB` (default 10).

Larger encoders, rerankers and open-model generation are run on a GPU with
`kaggle/run_gpu_experiments.py`; resource estimates and steps are in
[kaggle/README.md](kaggle/README.md).

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
kaggle/          GPU runner
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
