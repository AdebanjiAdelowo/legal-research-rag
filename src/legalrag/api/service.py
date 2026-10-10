"""Operations behind the API. Plain functions over one shared pipeline, so the web
interface can call them in process or over HTTP and get the same results."""

from __future__ import annotations

from dataclasses import asdict
from functools import lru_cache

from legalrag.data.citations import REPORTERS, find_citations
from legalrag.data.courts import CIRCUITS
from legalrag.data.treatment import DISCLAIMER, known_treatment
from legalrag.rag.pipeline import Pipeline
from legalrag.rag.report import to_markdown
from legalrag.rag.types import LIMITATION


@lru_cache(maxsize=1)
def pipeline() -> Pipeline:
    return Pipeline.from_config()


def _filters(request: dict) -> dict:
    keys = ("before", "after", "circuits", "levels", "binding_on", "forum_circuit")
    return {k: request[k] for k in keys if request.get(k)}


def health() -> dict:
    p = pipeline()
    dates = [c.decision_date for c in p.store.cases.values()]
    return {
        "status": "ok",
        "corpus_version": p.store.corpus_version,
        "demo_corpus": p.is_demo,
        "cases": len(p.store.cases),
        "passages": len(p.store),
        "date_range": [min(dates), max(dates)],
        "jurisdiction": "United States federal courts",
        "circuits": list(CIRCUITS),
        "config": asdict(p.config),
        "generator": p.generator.name,
        "support_judge": p.judge.name,
        "limitation": LIMITATION,
        "authority_notice": DISCLAIMER,
    }


def search(request: dict) -> dict:
    return pipeline().search(request["question"], int(request.get("k", 10)), **_filters(request))


def answer(request: dict):
    return pipeline().answer(request["question"], **_filters(request))


def report(request: dict) -> str:
    return to_markdown(answer(request))


def case(case_id: str) -> dict | None:
    """Full text of one case by paragraph, for inspecting evidence in context."""
    p = pipeline()
    found = p.store.cases.get(case_id)
    if found is None:
        return None
    return {
        "case_id": found.case_id,
        "name": found.name,
        "reference": found.bluebook(),
        "citations": found.citations,
        "court": found.court,
        "court_level": found.court_level,
        "circuit": found.circuit,
        "decision_date": found.decision_date,
        "docket_number": found.docket_number,
        "authors": found.authors,
        "source_url": found.source_url,
        "treatment": known_treatment(found.citations),
        "authority_notice": DISCLAIMER,
        "paragraphs": [
            {
                "para_id": para.para_id,
                "opinion_type": para.opinion_type,
                "section": para.section,
                "is_heading": para.is_heading,
                "text": para.text,
            }
            for para in found.paragraphs
        ],
    }


def validate_citations(text: str) -> dict:
    """Look up every reporter citation in ``text`` against the indexed corpus.

    This is an existence check only. It says whether the corpus holds a case at
    that citation, and which case; it says nothing about whether the case supports
    the sentence it appears in.
    """
    index = pipeline().verifier.index
    out = []
    for c in find_citations(text):
        if c.is_short_form:
            continue
        found = index.resolve(c.key) if c.known else None
        if not c.known:
            status = "reporter_not_recognised"
        elif found is None:
            status = "not_in_corpus"
        else:
            status = "in_corpus"
        out.append(
            {
                "raw": c.raw,
                "normalised": c.display,
                "reporter": REPORTERS.get(c.reporter, (c.reporter, None))[0],
                "status": status,
                "resolves_to": found.bluebook() if found else None,
                "case_id": found.case_id if found else None,
                "court": found.court if found else None,
                "treatment": known_treatment(found.citations) if found else None,
            }
        )
    return {
        "citations": out,
        "note": (
            "Existence in this corpus only. A citation marked not_in_corpus may be "
            "valid: the corpus holds a limited set of federal opinions. A citation "
            "marked in_corpus is not thereby shown to support any statement."
        ),
    }
