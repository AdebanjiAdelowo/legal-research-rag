# Error analysis

What goes wrong, where, and why, organised by stage. Every number comes from a file
under `results/`; the file is named in each section. Unless stated otherwise the
retriever is BM25 with development-tuned parameters on the 2,000 test queries.

## 1. Where retrieval fails

`results/analysis/error_analysis_bm25.json`

Each test query falls into one outcome by where the case the court quoted first
appears:

| Outcome | Queries | Share |
|---|---|---|
| A passage holding the quoted paragraph is in the top 10 | 253 | 12.7% |
| The right case is in the top 10, through a different passage | 358 | 17.9% |
| The right case ranks 11 to 100 | 777 | 38.9% |
| The right case ranks below 100 | 373 | 18.7% |
| The right case is not among the 1,000 passages kept | 239 | 12.0% |

Three readings follow.

**Most failures are ranking failures, not recall failures.** In 39% of queries the case
is retrieved but sits between rank 11 and 100. This is the pool a reranker is meant to
fix, which makes the reranking result in section 3 the more notable.

**Finding the opinion is easier than finding the paragraph.** For 358 queries the right
opinion is in the top 10 but the passage that surfaced it is not the one the court
quoted. A long opinion discusses its rule in several places; the lexically closest
discussion is often the application to the facts and not the sentence that later courts
quote.

**One query in eight never sees the case.** These are queries whose wording shares
almost nothing with the quoted case.

### What predicts failure

Recall@10 by property of the query and of the quoted case:

| Property | Group | n | Recall@10 |
|---|---|---|---|
| Word overlap between query and quotation | under 0.1 | 672 | 0.252 |
| | 0.1 to 0.3 | 869 | 0.298 |
| | 0.3 to 0.5 | 280 | 0.354 |
| | 0.5 and above | 179 | 0.469 |
| Query length in words | under 50 | 591 | 0.362 |
| | 50 to 89 | 402 | 0.336 |
| | 90 to 119 | 190 | 0.268 |
| | 120 (truncated) | 817 | 0.258 |
| Years between the quoted case and the query | under 5 | 148 | 0.338 |
| | 5 to 14 | 648 | 0.330 |
| | 15 to 29 | 762 | 0.318 |
| | 30 and above | 442 | 0.238 |
| Length of the quoted opinion in words | under 3,000 | 221 | 0.249 |
| | 3,000 to 7,000 | 724 | 0.308 |
| | 7,000 to 15,000 | 810 | 0.310 |
| | 15,000 and above | 245 | 0.335 |
| Court of the quoted case | Supreme Court | 1,225 | 0.317 |
| | Court of appeals | 771 | 0.288 |

**Vocabulary mismatch is the dominant factor.** Recall nearly doubles from the lowest
to the highest overlap group. A third of queries share under a tenth of the
quotation's content words: the citing court describes its own facts, and the quoted
passage states an abstract rule.

**Longer queries do worse.** The query is the 120 words before the quotation. The
words that matter are the last sentence or two; everything earlier is the citing
case's own discussion, which dilutes a bag-of-words query. A learned query reduction,
or weighting the query by distance from the citation point, is the obvious remedy and
was not tried.

**Old cases are harder.** Cases quoted 30 or more years later are found less often.
Legal vocabulary drifts, and older opinions carry more OCR noise.

**Long opinions are not harder at the case level.** Recall rises slightly with opinion
length: a long opinion has more passages and so more chances for one to match. Length
hurts in a different way, covered next.

## 2. Long documents and chunking

`results/chunking/chunking_bm25.json`

Section-aware chunking did not improve retrieval. Fixed 220-word windows were better
on every metric, and the difference survived a control with fixed windows of the same
mean length as the section-aware passages:

| Scheme | Mean words | nDCG@10 | Passage hit@10 | Hits per 1,000 words read |
|---|---|---|---|---|
| Section-aware, up to 220 words | 168 | 0.173 | 0.126 | 0.076 |
| Fixed 220 words, 40 overlap | 218 | 0.185 | 0.160 | 0.073 |
| Fixed 170 words, 30 overlap | about 170 | 0.182 | 0.145 | 0.086 |

Section-aware against the length-matched control: passage hit@10 lower by 0.019 (95%
interval 0.007 to 0.032, adjusted p = 0.014); nDCG@10 lower by 0.009, not significant
after adjustment.

Two mechanisms plausibly explain it, and the experiment does not separate them.
Overlapping windows place each paragraph in two passages, so a quoted paragraph has
two chances to be retrieved with different neighbours. And a hard stop at every
heading produces short passages at section starts that hold a rule statement without
the surrounding discussion that shares vocabulary with the query.

What section-aware chunking does give is cleaner provenance: a passage never mixes a
majority opinion with a dissent and never splits a paragraph, which matters for
citation and display. That is a reason to prefer it for the generation stage, not a
retrieval gain, and the data does not support claiming one. This was measured with
BM25 only.

## 3. Reranking made results worse

`results/rerank/`

| First stage | nDCG@10 before | nDCG@10 after | Difference |
|---|---|---|---|
| BM25 | 0.176 | 0.153 | -0.024, adjusted p < 0.001 |
| Hybrid, BM25 + bge-small, score fusion | 0.186 | 0.160 | -0.026, adjusted p < 0.001 |

The reranker is `cross-encoder/ms-marco-MiniLM-L6-v2`, 22M parameters, trained on web
search queries. Rescoring the top 20 passages lowered every metric, and deeper
rescoring was worse on the development queries (nDCG@10 0.188, 0.184 and 0.176 at
depths 20, 50 and 100 over the hybrid).

The likely cause is a mismatch of query type. MS MARCO queries are short questions
with an answer span in the passage. These queries are 90-word passages of legal
argument, and the relevant passage does not answer them; it states a rule the argument
is about to invoke. A model trained to find answers scores topical restatements above
the rule's source.

This does not show that reranking fails for legal retrieval. It shows that a small
general-domain reranker, applied without adaptation, is worse than no reranker on this
task. A legal-domain or larger cross-encoder is the comparison that matters and is run
by the GPU script.

## 4. Dense retrieval and fusion

`results/retrieval/retrieval_section220.json`

`bge-small-en-v1.5` alone scored below BM25 (nDCG@10 0.153 against 0.176, adjusted
p < 0.001). Fusing the two by normalised score raised nDCG@10 to 0.186 (+0.010 over
BM25, adjusted p = 0.002). Reciprocal rank fusion was not distinguishable from BM25
(+0.005, p = 0.26).

Score fusion beat rank fusion here because the tuned weight was uneven (0.7 on BM25).
Rank fusion weights both lists equally, which hurts when one retriever is clearly
weaker.

The gain from fusion is real and small. The two retrievers fail on overlapping
queries: when the citing court's vocabulary differs from the rule's, a 33M-parameter
general encoder does not bridge the gap either.

## 5. The masking control

`results/retrieval/retrieval_section220.json`, `leakage_control`

Leaving citations and case names in the query was expected to inflate scores. It
lowered BM25's nDCG@10 from 0.176 to 0.158 (p < 0.001) and left the dense encoder
unchanged (0.153 against 0.154, p = 0.57).

The explanation is in what is indexed. Head matter is excluded, so an opinion's own
name and citation are not in its passages, and a citation string in the query cannot
match its target. What it does match is every other opinion that cites the same case.
The citation is therefore noise, and worse than noise, since it pulls in citing cases.

This means the reported masked scores are not inflated by citation leakage. It also
means a system that indexed case names and citations would behave very differently on
unmasked text, and that such a system should be evaluated on masked queries.

## 6. Query decomposition did not help

`results/decomposition/decomposition_bm25.json`

On 600 multi-issue queries, splitting into sub-queries and fusing changed nothing
measurable: Recall@10 0.253 without, 0.247 with segment splitting, 0.244 with sentence
splitting. The oracle split, which uses the true parts each query was assembled from,
reached 0.254. On single-issue queries segment splitting lowered nDCG@10 by 0.008.

That the oracle does not help is the informative part. BM25 scores a query as a sum
over its terms, so a concatenated query already retrieves for all its parts at once;
splitting only replaces one sum with a fusion of partial sums. Decomposition should
matter more for dense retrieval, where a single vector must represent every issue.
That was not tested.

A second caveat: these multi-issue queries are concatenations of passages from one
opinion. They share a case's facts and vocabulary and are easier to separate than a
question a person would write.

## 7. Time

`results/robustness/temporal_bm25.json`

For queries from 1975 to 1999, when on average 49% of the corpus had been decided,
retrieval without a date filter returned cases from the query's future in 52% of the
top-10 positions, and the top result was a future case 49% of the time. Recall@10 fell
from 0.466 to 0.335.

Later cases restate earlier rules in more modern language, closer to the query's own.
Without the filter they outrank the source. In a benchmark that pooled relevance those
later cases might be counted as hits, and the system would look better than anything a
researcher at that date could have used.

For test queries from 2010 to 2018, 99% of the corpus predates the query and the
filter changes almost nothing. The effect is a property of how much of the corpus lies
in the query's future, so an evaluation on recent queries alone would miss it.

**Outdated authority.** In 1.3% of test queries the top 10 contained a case that the
treatment table records as already overruled or abrogated at the query date; *Saucier
v. Katz* accounts for most. In 12 queries the court itself quoted such a case, which is
legitimate: overruled cases are quoted to explain what changed. A further 0.6% of
queries surfaced a case that was overruled only after the query date, which nothing in
the corpus could reveal. The table holds 18 Supreme Court overrulings, 9 of them in the
corpus, so these rates are floors.

## 8. Jurisdiction

`results/robustness/jurisdiction_bm25.json`

For 94.2% of test queries the quoted case was binding on the citing court. Restricting
retrieval to binding authority raised Recall@10 from 0.305 to 0.456. The gain comes
from shrinking the pool to cases the court was likely to cite.

The cost falls entirely on the other 5.8%: when the court quoted persuasive authority
from another circuit, the binding filter makes the case unreachable (Recall@10 from
0.417 to zero). A binding-only filter is right as a default for "what is the law here"
and wrong for "has any court addressed this".

When retrieval was restricted to the wrong circuit, the right case was reachable for
0.3% of queries, yet the system returned a full ranked list every time. The best
retrieval score fell (mean 95.3 to 66.9) but separated the two conditions only with
AUROC 0.70.

The rule-based scope detector flagged all 30 hand-written state-law and foreign-law
questions and none of the 10 non-legal ones, which it was not built for. It raised one
false alarm in 30 in-scope questions: a federal personal-jurisdiction question
mentioning a company headquartered in Germany. The detector and the test questions
were written by the same author, so the 100% figure is optimistic.

## 9. Abstention

`results/abstention/`

The retrieval score does not reveal that the right case is missing.

| Condition | AUROC, BM25 score | AUROC, cross-encoder score |
|---|---|---|
| Quoted case withheld from the pool | 0.50 | 0.51 |
| As-of date moved before the quoted case | 0.59 | 0.59 |
| Wrong circuit searched | 0.69 | 0.68 |
| Out-of-scope questions | 0.99 | 0.79 |

When the quoted case is removed, the next-best passage scores almost exactly as well,
because many opinions discuss the same rule. A threshold on the top score is therefore
at chance. This is partly correct behaviour, since an adjacent case may be a
legitimate answer, and partly the failure that matters most in legal research: the
system has no signal that the specific authority is absent.

The score does separate questions that are not about federal law at all. The scope
rule catches 75% of those alone, and the rule and the BM25 score together catch all
40.

The same score also fails as a confidence estimate: it predicts whether the quoted
case is in the top five with AUROC 0.51 (BM25) and 0.58 (cross-encoder).

Reliable abstention in the hard case needs a different signal: whether the evidence
entails an answer to the question, not how similar it is. That is an open problem here.

## 10. Citation verification

`results/verification/`

<!-- verification:start -->
<!-- verification:end -->

## 11. Generation

`results/rag/`

<!-- generation:start -->
<!-- generation:end -->

## Summary of negative results

| Expectation | Result |
|---|---|
| Section-aware chunking improves retrieval | It did not, including against a length-matched control |
| A cross-encoder reranker improves ranking | A small web-trained reranker lowered every metric |
| Dense retrieval beats BM25 | A small general encoder was worse; fusion gained 0.010 nDCG@10 |
| Citations in the query leak the answer | They lowered BM25's score |
| Query decomposition helps multi-issue queries | No effect with BM25, even with oracle splits |
| A retrieval-score threshold supports abstention | At chance when the right case is missing |
