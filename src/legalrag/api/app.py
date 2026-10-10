"""HTTP API.

    GET  /health                 corpus, configuration and limitation notice
    POST /search                 ranked cases for a question
    POST /answer                 cited research summary with verification results
    POST /report                 the same, as a Markdown document to download
    POST /citations/validate     existence check for citations in arbitrary text
    GET  /cases/{case_id}        one opinion by paragraph

Run with ``legalrag serve``. The pipeline is chosen by ``LEGALRAG_CONFIG``; with no
built corpus it falls back to the synthetic demonstration corpus.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from legalrag.api import service
from legalrag.rag.types import LIMITATION

Circuit = Literal["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "DC", "Fed"]
Level = Literal["scotus", "appeals", "district", "other"]
DATE = r"^\d{4}-\d{2}-\d{2}$"


class Question(BaseModel):
    question: str = Field(min_length=3, max_length=4000)
    before: str | None = Field(None, pattern=DATE, description="Only cases decided before")
    after: str | None = Field(None, pattern=DATE, description="Only cases decided on or after")
    circuits: list[Circuit] = Field(default_factory=list, description="Courts of appeals")
    levels: list[Level] = Field(default_factory=list)
    binding_on: Circuit | None = Field(
        None, description="Restrict to Supreme Court and this circuit's court of appeals"
    )
    forum_circuit: Circuit | None = Field(
        None, description="Label results as binding or persuasive for a court in this circuit"
    )


class SearchRequest(Question):
    k: int = Field(10, ge=1, le=50)


class CitationText(BaseModel):
    text: str = Field(min_length=1, max_length=50000)


app = FastAPI(
    title="Legal research RAG",
    version="0.1.0",
    description=(
        "Case-law retrieval and source-grounded research summaries over U.S. federal "
        f"opinions. {LIMITATION}"
    ),
)


@app.get("/health")
def health() -> dict:
    return service.health()


@app.post("/search")
def search(request: SearchRequest) -> dict:
    return {**service.search(request.model_dump()), "limitation": LIMITATION}


@app.post("/answer")
def answer(request: Question) -> dict:
    return asdict(service.answer(request.model_dump()))


@app.post("/report", response_class=PlainTextResponse)
def report(request: Question) -> PlainTextResponse:
    return PlainTextResponse(
        service.report(request.model_dump()),
        media_type="text/markdown",
        headers={"Content-Disposition": 'attachment; filename="legal-research-report.md"'},
    )


@app.post("/citations/validate")
def validate(request: CitationText) -> dict:
    return service.validate_citations(request.text)


@app.get("/cases/{case_id}")
def case(case_id: str) -> dict:
    found = service.case(case_id)
    if found is None:
        raise HTTPException(status_code=404, detail=f"no case {case_id!r} in the corpus")
    return found
