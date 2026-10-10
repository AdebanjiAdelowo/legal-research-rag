# Dataset selection

This records which legal datasets were considered, what was verified about each, and
why the project uses the pair it does. Access, licence and file facts were checked on
2026-10-09 and 2026-10-10 against the Hugging Face Hub API, the dataset cards and the
Caselaw Access Project's static site.

## Requirements

The research questions need a benchmark with five properties:

1. a retrieval task, with queries and a corpus to search;
2. relevance judgments made by lawyers or courts, not by a model;
3. full opinions, so that paragraph structure, sections and citations can be parsed;
4. decision dates and courts on both the query side and the document side, for
   temporal and jurisdictional experiments;
5. a licence that permits research use and a build that others can repeat.

## Candidates

| Dataset | Jurisdiction | Task | Relevance labels | Licence and access | Decision |
|---|---|---|---|---|---|
| LePaRD (Mahari et al., 2023) | U.S. federal courts | Given the text before a quotation in an opinion, find the quoted precedent passage | Yes: each row is a quotation a judge made | CC BY-NC-SA 4.0, public on the Hub, not gated | **Used as the benchmark** |
| Caselaw Access Project (Harvard Law School Library Innovation Lab) | All U.S. courts, 1658 to about 2020 | None: a corpus | No | CC0 1.0, static JSON at `static.case.law`. Commercial restrictions ended in March 2024 | **Used as the corpus** |
| COLIEE Task 1 | Federal Court of Canada | Find noticed cases supporting a query case | Yes | Requires a signed memorandum sent to the organisers; no redistribution | Not used: access cannot be obtained inside a reproducible build, and the data could not be shared with the code |
| CaseHOLD (Zheng et al., 2021) | U.S. | Choose the correct holding among five options | Multiple-choice key | No licence declared on the Hub card | Not used: a classification task. Recasting it as retrieval would search a pool of five candidates, which measures something different |
| CLERC (Hou et al., 2024) | U.S. federal courts | Retrieve cited cases for a passage; generate analysis | Yes, from citations | No licence declared on the Hub card; built on the CC0 Caselaw Access Project | Not used: the passage collection alone is 9.2 GB compressed, and the licence is unstated |
| MLEB (Isaacus, 2025) | Several, one per dataset | Ten small retrieval sets | Expert | CC BY 4.0 for the set inspected | Not used as primary: it mixes jurisdictions by design. A candidate for external validation |
| LeCaRD | China | Case retrieval | Expert | Public | Not used: different legal system and language |

A legal classification benchmark is not a retrieval benchmark. CaseHOLD is the clearest
case: its "holdings" are short parentheticals and its negatives were chosen to be
confusable, so accuracy on it says little about finding a case among thousands.

## Scope

One jurisdiction: **United States federal courts**. The corpus holds decisions of the
Supreme Court (2,108), the courts of appeals (2,741), and a few district and
specialised courts (91). No state decisions are included, and none are mixed in as
distractors.

Within the federal system authority differs by court. That difference is modelled
explicitly (`src/legalrag/data/courts.py`): a Supreme Court decision binds every federal
court, and a court of appeals decision binds its own circuit. Circuit is never treated
as a separate jurisdiction to be mixed or ignored.

## How the two sources fit together

LePaRD gives, for each quotation, the citing opinion's court and date, the cited
opinion's identifier and citation, the quoted text, and the text that precedes the
quotation. It does not give the cited opinion's full text. The Caselaw Access Project
does, with paragraph breaks, opinion types (majority, concurrence, dissent), authors,
parallel citations and a resolved outgoing citation list.

LePaRD's case identifiers are Caselaw Access Project identifiers, so each cited case is
fetched by its citation and checked by identifier. The `top_10000` file refers to 4,976
cited opinions:

| Outcome | Opinions |
|---|---|
| Fetched and matched by identifier | 4,762 |
| Identifier since re-issued; matched by first page and case name, alias recorded | 178 |
| Cited only to a reporter the project does not host (2010 and 2011 Supreme Court cases cited to L. Ed. 2d or S. Ct.) | 36 |
| **In the corpus** | **4,940 (99.3%)** |

The 36 missing opinions remove 17,521 of 2,076,241 benchmark rows (0.8%).

## What LePaRD's labels mean

A row says that a court quoted passage P of case S at a given point in its opinion.
That is strong evidence that S is relevant to the text around that point. It is not
evidence that other cases are irrelevant: several opinions often state the same rule in
nearly the same words, and a researcher would accept any of them. Every score in this
project therefore understates usefulness, and "the quoted case was withheld" does not
mean "the question has no answer in the corpus".

## Properties found in the data

These were measured before any model was run and shaped the benchmark construction
(`src/legalrag/data/benchmark.py`).

**Answer leakage.** In a 5% sample of rows, the text before the quotation repeated the
opening of the quotation in 15.9% of rows and contained the reporter citation of the
cited case in 8.9%. Rows that repeat the quotation are dropped. In the rest, every
reporter citation is removed together with the case name attached to it and any
trailing court-and-year parenthetical, and short-form references to the cited case by
party name ("under *Chevron*") are removed. The unmasked text is kept for a control
experiment.

**Popularity skew.** The 10,000 passages are quoted between 78 and 33,439 times.
Sampling rows uniformly would test a few hundred famous passages over and over.
Queries are sampled with at most three per passage and one per citing opinion.

**Source court.** 72% of rows quote the Supreme Court. After per-passage sampling the
share among test queries is 61%.

**OCR.** Opinion text is OCR of printed reporters. Quotations are matched to paragraphs
by word-trigram containment with a threshold of 0.5, which located the quoted paragraph
for 3,920 of 3,930 passages (99.7%).

## Licence consequences

- LePaRD is non-commercial and share-alike. The benchmark file, the query files derived
  from it, and per-answer outputs that quote it are never committed. Aggregate result
  files are.
- The Caselaw Access Project is CC0. Opinion text may be redistributed, but it is also
  rebuilt by `legalrag fetch` and is not committed, to keep the repository small.
- The code is MIT licensed. Using this project commercially would require replacing
  LePaRD with relevance data under a compatible licence.

## References

- R. Mahari, D. Stammbach, E. Ash, A. Pentland. *LePaRD: A Large-Scale Dataset of
  Judges Citing Precedents.* arXiv:2311.09356.
- Harvard Law School Library Innovation Lab. *Caselaw Access Project.*
  https://case.law
- L. Zheng, N. Guha, B. Anderson, P. Henderson, D. Ho. *When Does Pretraining Help?
  Assessing Self-Supervised Learning for Law and the CaseHOLD Dataset.* ICAIL 2021.
- A. B. Hou et al. *CLERC: A Dataset for Legal Case Retrieval and Retrieval-Augmented
  Analysis Generation.* arXiv:2406.17186.
- COLIEE: Competition on Legal Information Extraction and Entailment.
  https://coliee.org
