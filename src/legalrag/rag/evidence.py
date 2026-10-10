"""Choosing which retrieved passages the generator may rely on.

The ranked list is cut to a few passages with three rules: no more than
``per_case`` passages from one opinion, so a single long case cannot fill the
context; passages from dissents and concurrences are kept but labelled, because
they are not the holding of the court; and each passage carries what is known about
the case's authority for the forum in question.
"""

from __future__ import annotations

from legalrag.data.courts import Court, authority_label
from legalrag.data.treatment import known_treatment
from legalrag.index.store import ChunkStore
from legalrag.rag.types import Evidence
from legalrag.retrieval.core import Ranking


def select_evidence(
    ranking: Ranking,
    store: ChunkStore,
    k: int = 5,
    per_case: int = 2,
    forum: Court | None = None,
    as_of_year: int | None = None,
    min_score: float | None = None,
    extra_treatment: dict | None = None,
) -> list[Evidence]:
    out: list[Evidence] = []
    used: dict[str, int] = {}
    for i, score in zip(ranking.idx, ranking.scores, strict=True):
        if len(out) >= k:
            break
        if min_score is not None and float(score) < min_score:
            break
        chunk = store.chunks[int(i)]
        if used.get(chunk.case_id, 0) >= per_case:
            continue
        used[chunk.case_id] = used.get(chunk.case_id, 0) + 1
        case = store.cases[chunk.case_id]
        court = Court(case.court, case.court_level, case.circuit)
        treatment = known_treatment(case.citations, as_of_year)
        if treatment is None and extra_treatment:
            from legalrag.data.citations import cite_key

            for cite in case.citations:
                treatment = extra_treatment.get(cite_key(cite) or "")
                if treatment:
                    break
        out.append(
            Evidence(
                label=f"E{len(out) + 1}",
                chunk_id=chunk.chunk_id,
                case_id=case.case_id,
                case_name=case.short_name,
                citation=case.official_cite,
                reference=case.bluebook(),
                court=case.court,
                court_level=case.court_level,
                circuit=case.circuit,
                decision_date=case.decision_date,
                opinion_type=chunk.opinion_type,
                section=chunk.section,
                para_ids=list(chunk.para_ids),
                text=chunk.text,
                score=float(score),
                authority=authority_label(court, forum),
                treatment=treatment,
                source_url=case.source_url,
            )
        )
    return out


def ranked_cases(ranking: Ranking, store: ChunkStore, k: int = 10, forum: Court | None = None,
                 as_of_year: int | None = None) -> list[dict]:  # fmt: skip
    """Top cases for display: best chunk per case, with authority and warnings."""
    seen: dict[str, dict] = {}
    for i, score in zip(ranking.idx, ranking.scores, strict=True):
        chunk = store.chunks[int(i)]
        if chunk.case_id in seen:
            continue
        case = store.cases[chunk.case_id]
        seen[chunk.case_id] = {
            "case_id": case.case_id,
            "name": case.short_name,
            "citation": case.official_cite,
            "reference": case.bluebook(),
            "court": case.court,
            "court_level": case.court_level,
            "circuit": case.circuit,
            "decision_date": case.decision_date,
            "score": round(float(score), 4),
            "best_chunk": chunk.chunk_id,
            "best_paragraphs": chunk.para_ids,
            "opinion_type": chunk.opinion_type,
            "snippet": chunk.text[:400],
            "authority": authority_label(Court(case.court, case.court_level, case.circuit), forum),
            "treatment": known_treatment(case.citations, as_of_year),
            "source_url": case.source_url,
        }
        if len(seen) >= k:
            break
    return list(seen.values())
