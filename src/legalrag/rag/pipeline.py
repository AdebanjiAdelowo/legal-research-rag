"""The question-answering pipeline.

question
  -> jurisdiction and date interpretation   (retrieval/query.py)
  -> first-stage retrieval, optional fusion (index/, retrieval/core.py)
  -> optional query decomposition           (retrieval/decompose.py)
  -> optional cross-encoder reranking       (retrieval/rerank.py)
  -> sufficiency gate                       (rag/abstain.py)
  -> evidence selection                     (rag/evidence.py)
  -> typed, cited statements                (rag/generate.py)
  -> citation verification                  (verify/verifier.py)

Every stage is optional except BM25 retrieval and extraction, so the default
configuration runs with no model download and no network.
"""

from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from legalrag.data.courts import DISTRICT, Court
from legalrag.data.treatment import DISCLAIMER
from legalrag.index import store as S
from legalrag.index.bm25 import BM25Index
from legalrag.paths import Paths
from legalrag.rag.abstain import Gate
from legalrag.rag.evidence import ranked_cases, select_evidence
from legalrag.rag.generate import Generator, make_generator
from legalrag.rag.types import Answer
from legalrag.retrieval.core import Ranking, rrf, score_fusion, top_k
from legalrag.retrieval.decompose import decompose
from legalrag.retrieval.query import QueryPlan, interpret
from legalrag.verify.judges import Judge, make_judge
from legalrag.verify.verifier import CitationIndex, CitationVerifier


@dataclass
class PipelineConfig:
    corpus: str = "auto"  # auto | full | demo
    scheme: str = "section220"
    retriever: str = "bm25"  # bm25 | dense | hybrid
    encoder: str | None = None
    fusion: str = "rrf"  # rrf | score
    alpha_bm25: float = 0.5
    rrf_k: int = 60
    bm25_k1: float = 1.2
    bm25_b: float = 0.75
    depth: int = 200
    decompose: str | None = None  # None | segments | sentences | enumeration
    reranker: str | None = None
    rerank_top_n: int = 50
    evidence_k: int = 5
    per_case: int = 2
    generator: str = "extractive"
    judge: str = "lexical"
    verify: str = "flag"  # off | flag | strict
    abstain_threshold: float | None = None
    abstain_signal: str = "top1"

    @classmethod
    def load(cls, path: str | Path | None) -> PipelineConfig:
        path = path or os.environ.get("LEGALRAG_CONFIG")
        if not path:
            return cls()
        data = yaml.safe_load(Path(path).read_text()) or {}
        unknown = set(data) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown configuration keys: {sorted(unknown)}")
        return cls(**data)


@dataclass
class Pipeline:
    config: PipelineConfig
    store: S.ChunkStore
    bm25: BM25Index
    dense: object | None = None
    reranker: object | None = None
    generator: Generator = field(default_factory=lambda: make_generator("extractive"))
    judge: Judge = field(default_factory=lambda: make_judge("lexical"))
    extra_treatment: dict | None = None
    is_demo: bool = False

    def __post_init__(self):
        self.verifier = CitationVerifier(CitationIndex(self.store.cases), self.judge)
        self.gate = Gate(self.config.abstain_threshold, self.config.abstain_signal)

    # -- construction --------------------------------------------------------

    @classmethod
    def demo(cls, config: PipelineConfig | None = None, **overrides) -> Pipeline:
        """Pipeline over the synthetic corpus. No data, model or network needed."""
        from legalrag import demo

        config = config or PipelineConfig(corpus="demo")
        store = demo.store()
        return cls(
            config, store, BM25Index.build(store.texts, config.bm25_k1, config.bm25_b),
            extra_treatment=demo.DEMO_TREATMENT, is_demo=True, **overrides,
        )  # fmt: skip

    @classmethod
    def from_config(cls, path: str | Path | None = None, paths: Paths | None = None) -> Pipeline:
        config = PipelineConfig.load(path)
        paths = paths or Paths.default()
        have_data = paths.manifest.exists() and paths.chunks(config.scheme).exists()
        if config.corpus == "demo" or (config.corpus == "auto" and not have_data):
            return cls.demo(config, generator=make_generator(config.generator),
                            judge=make_judge(config.judge))  # fmt: skip
        if not have_data:
            raise FileNotFoundError(
                "corpus not built; run `legalrag fetch` and `legalrag build`, or set corpus: demo"
            )
        store = S.ChunkStore.load(paths, config.scheme)
        bm25 = S.bm25_index(paths, store)
        bm25.set_params(config.bm25_k1, config.bm25_b)
        dense = None
        if config.retriever in ("dense", "hybrid"):
            if not config.encoder:
                raise ValueError("retriever dense/hybrid needs an encoder")
            dense = S.dense_index(paths, store, config.encoder)
        reranker = None
        if config.reranker:
            from legalrag.retrieval.rerank import Reranker

            reranker = Reranker(config.reranker)
        return cls(config, store, bm25, dense, reranker, make_generator(config.generator),
                   make_judge(config.judge))  # fmt: skip

    # -- retrieval -----------------------------------------------------------

    def _first_stage(self, text: str, mask) -> Ranking:
        c = self.config
        lexical = top_k(self.bm25.score(text), c.depth, mask) if c.retriever != "dense" else None
        semantic = None
        if c.retriever != "bm25":
            semantic = top_k(self.dense.score_batch([text])[0], c.depth, mask)
        if lexical is None:
            return semantic
        if semantic is None:
            return lexical
        if c.fusion == "score":
            return score_fusion(lexical, semantic, c.alpha_bm25)
        return rrf([lexical, semantic], c.rrf_k)

    def retrieve(
        self, plan: QueryPlan, exclude_cases: tuple[str, ...] = ()
    ) -> tuple[Ranking, dict]:
        """Chunk ranking for an interpreted question, with stage timings."""
        timings: dict[str, float] = {}
        filters = plan.filters()
        if exclude_cases:
            filters["exclude_cases"] = exclude_cases
        mask = self.store.mask(**filters)
        start = time.perf_counter()
        parts = (
            decompose(plan.text, self.config.decompose) if self.config.decompose else [plan.text]
        )
        rankings = [self._first_stage(p, mask) for p in parts]
        ranking = rankings[0] if len(rankings) == 1 else rrf(rankings, self.config.rrf_k)
        timings["retrieval"] = time.perf_counter() - start
        if self.reranker is not None and len(ranking):
            start = time.perf_counter()
            ranking = self.reranker.rerank(plan.text, ranking, self.store, self.config.rerank_top_n)
            timings["reranking"] = time.perf_counter() - start
        return ranking, {"timings": timings, "sub_queries": parts if len(parts) > 1 else []}

    # -- public API ----------------------------------------------------------

    def plan(self, question: str, **overrides) -> QueryPlan:
        """Interpret a question; explicit keyword filters override what was parsed.

        Accepted overrides: ``before``, ``after``, ``circuits``, ``levels``,
        ``binding_on``, ``forum_circuit``.
        """
        plan = interpret(question)
        for key, value in overrides.items():
            if value not in (None, "", [], ()):
                setattr(plan, key, list(value) if isinstance(value, tuple) else value)
        if plan.binding_on and not plan.forum_circuit:
            plan.forum_circuit = plan.binding_on
        return plan

    def _forum(self, plan: QueryPlan) -> Court | None:
        if not plan.forum_circuit:
            return None
        return Court(f"a court in circuit {plan.forum_circuit}", DISTRICT, plan.forum_circuit)

    def search(self, question: str, k: int = 10, **overrides) -> dict:
        """Ranked cases only, no generation."""
        plan = self.plan(question, **overrides)
        ranking, info = self.retrieve(plan)
        year = int(plan.before[:4]) - 1 if plan.before else None
        return {
            "plan": asdict(plan),
            "cases": ranked_cases(ranking, self.store, k, self._forum(plan), year),
            "timings": info["timings"],
            "sub_queries": info["sub_queries"],
            "authority_notice": DISCLAIMER,
        }

    def answer(self, question: str, exclude_cases: tuple[str, ...] = (), **overrides) -> Answer:
        plan = self.plan(question, **overrides)
        answer = Answer(question=question, abstained=False, plan=asdict(plan),
                        generator=self.generator.name)  # fmt: skip
        answer.warnings.append(DISCLAIMER)
        if self.is_demo:
            answer.warnings.append(
                "Demonstration corpus: every case shown is synthetic and does not exist."
            )
        if plan.out_of_scope:
            answer.abstained = True
            answer.abstain_reason = "; ".join(plan.out_of_scope)
            return answer

        ranking, info = self.retrieve(plan, exclude_cases)
        answer.timings.update(info["timings"])
        forum = self._forum(plan)
        year = int(plan.before[:4]) - 1 if plan.before else None
        answer.ranked_cases = ranked_cases(ranking, self.store, 10, forum, year)

        abstain, reason = self.gate.should_abstain(ranking)
        if abstain:
            answer.abstained, answer.abstain_reason = True, reason
            return answer

        answer.evidence = select_evidence(
            ranking, self.store, self.config.evidence_k, self.config.per_case, forum, year,
            extra_treatment=self.extra_treatment,
        )  # fmt: skip
        start = time.perf_counter()
        abstained, statements, raw = self.generator.generate(question, answer.evidence)
        answer.timings["generation"] = time.perf_counter() - start
        answer.raw_output = raw
        answer.statements = statements
        if abstained:
            answer.abstained = True
            answer.abstain_reason = "the generator found the retrieved passages insufficient"
            return answer

        if self.config.verify != "off":
            start = time.perf_counter()
            self.verifier.verify(answer, self.config.verify)
            answer.timings["verification"] = time.perf_counter() - start
        return answer
