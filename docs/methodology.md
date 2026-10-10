# Methodology

How each component works, how it is evaluated, and what the evaluation cannot show.
Dataset selection is in [datasets.md](datasets.md); findings by failure type are in
[error_analysis.md](error_analysis.md).

## 1. Task

A query is a passage of a judicial opinion: the text that leads up to a point where the
court quotes an earlier case. The system must find that earlier case, and within it
the paragraph quoted, among 4,940 federal opinions cut into about 220,000 passages. The
query carries the date and court of the opinion it comes from.

This is the citation-recommendation form of case-law retrieval. It is harder than
keyword lookup, since the query does not name the case, and easier to label than
open-ended research questions, since the court itself chose the answer.

## 2. Documents

### Parsing

Each opinion is read from its Caselaw Access Project JSON file into one record per
paragraph. Paragraphs are never merged or re-flowed. A paragraph identifier has the
form `cap:<case id>:<opinion index>:<paragraph index>`, so it names a span that a
reader can find in the source, and identifiers are identical across rebuilds.

Each case keeps its court, court level (Supreme Court, court of appeals, district,
other), circuit, decision date, short and full name, docket number, official and
parallel citations, opinion authors, the outgoing citation list, a SHA-256 of its text
and its source URL. Majority opinions, concurrences and dissents are separate opinions
within a case and are never mixed in a passage.

Section headings are detected by rule: a short line that is a roman numeral, a capital
letter or an arabic numeral, optionally followed by a capitalised title, or one of a
fixed list of named headings ("Background", "Discussion"). Each paragraph records its
heading path, for example `II/A`. 4,140 of 4,940 opinions have at least one heading.

### Citations

Reporter citations are parsed into volume, reporter and page, with optional pin cite
and short forms ("410 U.S. at 153"). Reporter spellings are normalised to a key, so
"F. 2d" and "F.2d" are the same reporter. A citation to a reporter outside the table is
still parsed and marked as unrecognised, so that the verifier can say "cannot be
checked" and not stay silent.

### Duplicates

Exact duplicates are found by a hash of normalised text, near duplicates by a 64-bit
SimHash over word 5-grams compared within bands. Exact duplicates are dropped; near
duplicates are only reported, because an amended opinion is a different legal document
from the one it replaces. The corpus contained neither.

### Chunking

Two schemes produce the retrieval units.

*Fixed*: a sliding window of 220 words with 40 words of overlap over the opinion text,
blind to all structure.

*Section-aware*: paragraphs are packed up to 220 words; a passage never crosses a
section heading or an opinion boundary; a paragraph is split only if it alone exceeds
330 words, and then at sentence boundaries by a splitter that does not break inside
citations. Groups under 12 words (author lines, bare dispositions) are not emitted.

Both record the paragraph identifiers each passage covers. Head matter (syllabus,
counsel, the case caption) is excluded from both, which also keeps the case's own name
and citation out of its indexed text.

### Versioned indexes

The corpus version is a hash over the text hashes of all cases kept. An index lives at
`data/indexes/<corpus version>/<scheme>/<name>/` and its metadata repeats the version,
scheme and passage count. Loading an index against a corpus it was not built from
raises an error.

## 3. Benchmark construction

Built from LePaRD's 10,000 most-quoted passages (`src/legalrag/data/benchmark.py`).

| Step | Rule |
|---|---|
| Leakage: quotation | Drop a row if the query text repeats the first or last eight words of the quotation |
| Leakage: citations | Remove every reporter citation, the case name before it, and the parenthetical after it |
| Leakage: short forms | Remove the cited case's distinctive party words ("Chevron") wherever they appear capitalised |
| Length | Keep the last 120 words; require at least 25 |
| Sampling | At most three queries per passage, one per citing opinion, selected by a seeded hash |
| Paragraph labels | Locate the quotation in the cited opinion by word-trigram containment of at least 0.5 |

Splits are by the date of the citing opinion:

| Split | Citing opinions decided | Queries | Use |
|---|---|---|---|
| `dev` | 2000 to 2009 | 800 | All tuning |
| `test` | 2010 to 2018 | 2,000 | All reported results |
| `history` | 1975 to 1999 | 800 | Temporal experiment only |
| `multi_dev`, `multi_test` | as above | 300, 600 | Query decomposition |

Because test queries are later than development queries, no setting is tuned on
opinions from the future of the ones it is tested on. A citing opinion appears in one
split only.

Multi-issue queries join two or three contexts from the same citing opinion that quote
different cases. Contexts with more than 40% token overlap are treated as one issue and
not combined.

## 4. Retrieval

**BM25.** Okapi BM25 over lower-cased word tokens with stopwords removed and plural
forms conflated. A full stemmer is deliberately not used: it merges terms that are
distinct in law. The index stores raw term frequencies, so `k1` and `b` are re-weighted
without rebuilding; they are tuned on `dev` over a 20-point grid.

**Dense.** Bi-encoder embeddings, L2-normalised, stored as float16, searched exactly by
inner product. There is no approximate-nearest-neighbour index, so no approximation
error enters the results. Each encoder is pinned to a Hub revision together with the
query and document prefixes it was trained with.

**Legal against general.** The legal encoder is
`freelawproject/modernbert-embed-base_finetune_512`, a fine-tune of
`nomic-ai/modernbert-embed-base` on U.S. court opinions. Comparing a model with its own
parent holds architecture, size and tokeniser fixed, which a comparison between
unrelated models does not. One caveat applies and is not removable: the fine-tuning
data comes from U.S. case law and may include the opinions in this corpus, so an
advantage could partly reflect exposure and not only domain adaptation.

**Hybrid.** Two fusions of the BM25 and dense lists, each over the top 200 passages:
reciprocal rank fusion, with its constant tuned on `dev`; and a convex combination of
per-query min-max normalised scores, with the weight tuned on `dev`. Raw scores are
never added, because BM25 and cosine values are on different scales.

**Filters.** Every filter is a mask over passages applied before the top-k cut, so a
filtered list is still full length. The date filter keeps cases decided before a given
day. The binding-authority filter keeps Supreme Court decisions and those of one
circuit's court of appeals.

**Reranking.** A cross-encoder rescoring the top `n` passages, with `n` chosen on
`dev`. Passages below `n` keep their order beneath the reranked block.

**Query interpretation.** Court and date constraints written into a question are
extracted by rule and removed from the retrieval text. A single named circuit is read
as "the law that applies there" (Supreme Court plus that circuit) unless the question
asks for that court's own decisions. Questions that name state law or another country
are flagged as outside the corpus before any retrieval.

**Decomposition.** A query is split into sub-queries by lexical-cohesion segments or by
sentences; each is retrieved separately and the lists are fused with the full query's
list. No language model is involved.

## 5. Grounded generation

An answer is a list of typed statements, not prose:

| Type | Meaning | Must cite |
|---|---|---|
| `fact` | What happened in a cited case, as that opinion states it | Yes |
| `reasoning` | A court's rule, holding or explanation | Yes |
| `provision` | A statute, rule or constitutional text an opinion names | Yes |
| `interpretation` | The system's own synthesis; never presented as authority | No |
| `not_established` | A part of the question the evidence does not answer | No |

Evidence is the top five passages after reranking, at most two per case. Each carries
its case, court, date, opinion type, section, paragraph identifiers, whether it is
binding or persuasive for the forum in question, and any known later negative
treatment. Passages from dissents and concurrences are kept and labelled.

Two generators produce statements. The extractive generator selects sentences from the
evidence and reports them verbatim, so it cannot misstate a case; it also cannot
synthesise. The model-based generator prompts a chat model to write tagged lines and
cite passages only by label; reporter citations written by the model are treated as
suspect by the verifier. Untagged output is classed as interpretation, never as a
sourced statement.

## 6. Citation verification

Six checks, kept separate so that passing one is never read as passing another:

| Check | Question | Method |
|---|---|---|
| Marker validity | Does `[E7]` refer to a passage that was supplied? | Exact |
| Existence | Is a written reporter citation in the corpus? | Exact lookup over all official and parallel citations |
| Provenance | Was the cited case among the supplied passages? | Exact |
| Name agreement | Does the name before a citation match the case at that citation? | Word overlap of at least half |
| Quotation fidelity | Does quoted text occur in a cited passage? | Word-trigram containment of at least 0.8 |
| Substantive support | Does the cited passage support the statement? | Automatic judge |

"Not found in the corpus" is reported as exactly that. The corpus is a small part of
federal case law, so absence does not show that a citation is wrong.

The support judge is either lexical (share of the statement's content words found in
the passage) or a natural-language-inference model scoring the passage as premise and
the statement as hypothesis, over the whole passage and over three-sentence windows.
Its output is always labelled as an automatic check with the judge named. It is not a
legal determination, and its error rate is measured (section 7).

Under the strict policy, citable statements that fail are removed and the answer says
how many were withheld. If none survive, the system abstains.

## 7. Evaluation

**Retrieval.** Recall@k, MRR@10, nDCG@10 and Precision@k at the case level, where a
case's rank is that of its best passage. At the paragraph level, a hit is a returned
passage that contains the quoted paragraph. Each query has one relevant case, so
Precision@k is bounded by 1/k and is reported for completeness only.

**Uncertainty.** Means carry 95% percentile bootstrap intervals over queries (2,000
resamples). Differences between systems are paired: a bootstrap interval for the
difference and a two-sided sign-flip randomisation test (10,000 permutations), with
Holm adjustment within each family of comparisons. Proportions from small samples use
Wilson intervals.

**Generation.** There are no reference answers, so correctness is measured by proxy
against what the court did: whether the answer cites the quoted case, whether it cites
the passage holding the quoted paragraph, and how much of the quotation's wording it
reproduces. Citation precision is the share of evidence citations whose passage
supports the statement; citation recall is the share of citable statements with at
least one supporting citation; the unsupported-claim rate is its complement. The
hallucinated-citation rate is the share of answers with a marker that points at
nothing or a reporter citation that is absent from the corpus or from the evidence.

**Judge validation.** Labelled pairs are built from the benchmark with no model in the
loop: a court's quotation against the passage that holds it (supported), against a
passage of another case, against another passage of the same opinion, and a negated
version against the right passage.

**Verifier validation.** Verbatim statements are corrupted in eight known ways, and
detection is scored for each.

**Abstention.** The same queries are run with the quoted case reachable and under four
conditions in which it is not. A threshold on the retrieval score is tuned on `dev`.
Reported: AUROC of the score, coverage of answerable queries, and abstention recall.

## 8. Threats to validity

- **Labels are incomplete.** Only the quoted case counts as relevant. Other cases that
  state the same rule are scored as errors.
- **Queries are judicial prose.** A passage of an opinion is not how a lawyer phrases a
  research question. Results on short natural-language questions may differ.
- **Masking is imperfect and changes the text.** Removing citations and names also
  removes signals a real query would not have had, but can leave fragments ("the
  two-step analysis") that point to a case.
- **The corpus is closed.** Every opinion in it was quoted at least 78 times by other
  courts. There are no obscure or unquoted opinions to act as distractors at the case
  level, though at the passage level nearly all of the 220,000 passages are
  distractors.
- **The legal encoder may have seen the corpus** during fine-tuning.
- **No citator.** Whether a case is still good law is not known to the system. A table
  of 18 well-known Supreme Court overrulings is used to flag a handful of cases and to
  measure how often they are returned; it is illustrative and incomplete.
- **Automatic support judgments are fallible**, most of all on negation and on
  paraphrase, and legal text is outside the training distribution of general
  entailment models.
- **Query independence.** Queries from the same court or quoting the same passage are
  not independent, so intervals are somewhat narrow.
- **One jurisdiction.** Nothing here transfers to state law or to other legal systems
  without new data.
