# Data

Nothing in this directory is committed except this file. Everything is rebuilt from
two pinned public sources by `legalrag fetch` and `legalrag build`.

| Path | Content | Source and licence |
|---|---|---|
| `raw/lepard/top_10000_data.csv.gz` | Citing contexts and quoted passages | LePaRD, Hugging Face `rmahari/LePaRD` at revision `0194f95c`, CC BY-NC-SA 4.0. SHA-256 checked on download |
| `raw/cap/cases/*.json` | One file per cited opinion | Caselaw Access Project, `static.case.law`, CC0 1.0 |
| `processed/corpus.jsonl` | Parsed opinions with paragraph ids | derived from the Caselaw Access Project |
| `processed/chunks_*.jsonl` | Retrieval units per chunking scheme | derived from the Caselaw Access Project |
| `processed/queries_*.jsonl` | Benchmark queries and relevance labels | derived from LePaRD: non-commercial, share-alike |
| `processed/manifest.json` | Counts, filter statistics, corpus version | generated |
| `indexes/<corpus version>/<scheme>/` | BM25 and dense indexes, cached runs | generated |

LePaRD's licence forbids commercial use and requires derivatives to carry the same
licence. Query files and per-answer experiment outputs contain LePaRD text and are
therefore kept out of version control. The result files under `results/` hold
aggregate numbers only.
