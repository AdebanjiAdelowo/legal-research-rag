"""Research interface.

    streamlit run app/streamlit_app.py

Calls the pipeline in process by default. Set ``LEGALRAG_API_URL`` to use a running
API instead (``legalrag serve``).
"""

from __future__ import annotations

import os
from dataclasses import asdict

import streamlit as st

API_URL = os.environ.get("LEGALRAG_API_URL", "").rstrip("/")
CIRCUITS = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "DC", "Fed"]
LEVELS = {"Supreme Court": "scotus", "Courts of appeals": "appeals", "District courts": "district"}
TYPE_HEADINGS = [
    ("reasoning", "Judicial reasoning in the retrieved cases"),
    ("fact", "Facts of the retrieved cases"),
    ("provision", "Legal provisions referred to"),
    ("interpretation", "System interpretation (not authority)"),
    ("not_established", "Not established by the retrieved evidence"),
]
STATUS = {
    "verified": ("supported", "green"),
    "unsupported": ("not confirmed", "red"),
    "uncited": ("no citation", "red"),
}


def call(operation: str, payload: dict | str | None = None):
    """Run one operation through the API or in process."""
    if API_URL:
        import httpx

        if operation == "health":
            return httpx.get(f"{API_URL}/health", timeout=60).json()
        if operation == "case":
            r = httpx.get(f"{API_URL}/cases/{payload}", timeout=60)
            return r.json() if r.status_code == 200 else None
        path = {"search": "/search", "answer": "/answer", "report": "/report",
                "validate": "/citations/validate"}[operation]  # fmt: skip
        r = httpx.post(f"{API_URL}{path}", json=payload, timeout=600)
        r.raise_for_status()
        return r.text if operation == "report" else r.json()
    from legalrag.api import service

    if operation == "health":
        return service.health()
    if operation == "case":
        return service.case(payload)
    if operation == "answer":
        return asdict(service.answer(payload))
    if operation == "validate":
        return service.validate_citations(payload["text"])
    return getattr(service, operation)(payload)


@st.cache_data(show_spinner=False)
def health() -> dict:
    return call("health")


def sidebar(info: dict) -> dict:
    st.sidebar.header("Scope")
    st.sidebar.caption(
        f"{info['jurisdiction']}. {info['cases']:,} opinions, "
        f"{info['date_range'][0][:4]} to {info['date_range'][1][:4]}."
    )
    filters: dict = {}
    st.sidebar.subheader("Jurisdiction")
    mode = st.sidebar.radio(
        "Courts",
        ["All federal courts", "Authority binding in a circuit", "Decisions of selected courts"],
        help="Binding authority for a circuit is the Supreme Court plus that circuit's "
        "court of appeals.",
    )
    if mode == "Authority binding in a circuit":
        filters["binding_on"] = st.sidebar.selectbox("Circuit", CIRCUITS, index=8)
    elif mode == "Decisions of selected courts":
        levels = st.sidebar.multiselect("Court level", list(LEVELS))
        filters["levels"] = [LEVELS[x] for x in levels]
        filters["circuits"] = st.sidebar.multiselect("Circuits", CIRCUITS)
        forum = st.sidebar.selectbox("Label authority for a court in", ["(none)", *CIRCUITS])
        if forum != "(none)":
            filters["forum_circuit"] = forum
    st.sidebar.subheader("Date")
    if st.sidebar.checkbox("Decided before"):
        filters["before"] = st.sidebar.date_input("Before", key="before").isoformat()
    if st.sidebar.checkbox("Decided on or after"):
        filters["after"] = st.sidebar.date_input("On or after", key="after").isoformat()
    st.sidebar.divider()
    st.sidebar.caption(
        f"Corpus version {info['corpus_version']}. Generator: {info['generator']}. "
        f"Support judge: {info['support_judge']}."
    )
    return filters


def show_plan(plan: dict) -> None:
    notes = list(plan.get("notes") or [])
    if plan.get("text") and plan["text"] != plan.get("original"):
        notes.insert(0, f"Retrieval text: “{plan['text']}”")
    if notes:
        with st.expander("How the question was read", expanded=False):
            for n in notes:
                st.write("- " + n)


def treatment_warning(treatment: dict | None) -> None:
    if treatment:
        st.warning(
            f"Later {treatment['treatment']} by {treatment['by']} ({treatment['year']}). "
            f"{treatment['note']}"
        )


def show_cases(cases: list[dict]) -> None:
    for rank, c in enumerate(cases, start=1):
        badge = {"binding": " · binding", "persuasive": " · persuasive"}.get(c["authority"], "")
        with st.expander(f"{rank}. {c['reference']}{badge}"):
            st.caption(f"{c['court']} · decided {c['decision_date']} · score {c['score']}")
            treatment_warning(c.get("treatment"))
            if c["opinion_type"] not in ("majority", "unanimous"):
                st.info(
                    f"Best passage is from a {c['opinion_type']}, not the opinion of the court."
                )
            st.write(c["snippet"] + " …")
            st.caption("Paragraphs: " + ", ".join(c["best_paragraphs"]))
            if st.button("Open full opinion", key=f"open-{rank}-{c['case_id']}"):
                st.session_state["inspect"] = (c["case_id"], set(c["best_paragraphs"]))


def show_answer(a: dict) -> None:
    for w in a["warnings"]:
        st.caption("⚠ " + w)
    show_plan(a["plan"])
    if a["abstained"]:
        st.error(f"No answer given: {a['abstain_reason']}.")
        st.caption(
            "Abstention means the indexed opinions did not provide sufficient support. It "
            "does not mean that no authority exists."
        )
    {e["label"]: e for e in a["evidence"]}
    for kind, heading in TYPE_HEADINGS:
        items = [s for s in a["statements"] if s["type"] == kind]
        if not items:
            continue
        st.subheader(heading)
        for s in items:
            cites = " ".join(f"[{c}]" for c in s["citations"])
            text = f"“{s['text']}”" if s["quoted"] else s["text"]
            check = s.get("verification") or {}
            label, colour = STATUS.get(check.get("status", ""), ("", "gray"))
            suffix = f" :{colour}[automatic check: {label}]" if label else ""
            st.markdown(f"- {text} **{cites}**{suffix}")
            for problem in check.get("problems", []):
                st.caption("  ⚠ " + problem)

    if a["evidence"]:
        st.subheader("Evidence passages")
        for e in a["evidence"]:
            with st.expander(f"[{e['label']}] {e['reference']}"):
                flags = [e["opinion_type"]]
                if e["authority"] != "unspecified":
                    flags.append(e["authority"])
                st.caption(f"{e['court']} · decided {e['decision_date']} · " + " · ".join(flags))
                treatment_warning(e.get("treatment"))
                st.write(e["text"])
                st.caption(
                    f"Section: {e['section'] or 'none'} · paragraphs: {', '.join(e['para_ids'])}"
                )
                if st.button("Show in full opinion", key=f"ev-{e['label']}"):
                    st.session_state["inspect"] = (e["case_id"], set(e["para_ids"]))
    v = a.get("verification") or {}
    if v:
        st.subheader("Citation verification")
        c1, c2, c3 = st.columns(3)
        c1.metric("Citations supported", f"{v['supported_citations']} / {v['evidence_citations']}")
        c2.metric("Not traceable to evidence", v["hallucinated_citations"])
        c3.metric("Quotations not found", v["quotations_not_found"])
        st.caption(f"Judge: {v['judge']}. {v['note']}")


def show_case(case_id: str, highlight: set[str]) -> None:
    case = call("case", case_id)
    if not case:
        st.error("Case not found.")
        return
    st.subheader(case["reference"])
    st.caption(f"{case['court']} · {case['decision_date']} · {'; '.join(case['citations'])}")
    treatment_warning(case.get("treatment"))
    st.caption(case["authority_notice"])
    if case["source_url"].startswith("http"):
        st.markdown(f"[Source file]({case['source_url']})")
    current = None
    for p in case["paragraphs"]:
        if p["opinion_type"] != current:
            current = p["opinion_type"]
            st.markdown(f"#### {current.replace('_', ' ').title()}")
        if p["is_heading"]:
            st.markdown(f"**{p['text']}**")
        elif p["para_id"] in highlight:
            st.markdown(f":orange-background[{p['text']}]")
            st.caption(p["para_id"])
        else:
            st.write(p["text"])


def main() -> None:
    st.set_page_config(page_title="Legal research", layout="wide")
    info = health()
    st.title("Case-law research")
    st.error(info["limitation"], icon="⚖️")
    if info["demo_corpus"]:
        st.warning("Demonstration corpus: every case shown is synthetic and does not exist.")
    filters = sidebar(info)

    research, validate, inspect = st.tabs(["Research", "Validate citations", "Inspect opinion"])
    with research:
        question = st.text_area(
            "Research question",
            placeholder="When does a government official lose qualified immunity?",
            height=90,
        )
        st.caption(
            "Court and date limits written into the question are recognised, for example "
            "“in the Ninth Circuit before 2009”. Sidebar settings take precedence."
        )
        col1, col2 = st.columns(2)
        if col1.button("Search cases", use_container_width=True) and question.strip():
            st.session_state["result"] = (
                "search",
                call("search", {"question": question, **filters}),
            )
        if (
            col2.button("Research summary", type="primary", use_container_width=True)
            and question.strip()
        ):
            payload = {"question": question, **filters}
            st.session_state["result"] = ("answer", call("answer", payload))
            st.session_state["report"] = call("report", payload)
        kind, result = st.session_state.get("result", (None, None))
        if kind == "search":
            show_plan(result["plan"])
            st.caption(result["authority_notice"])
            if not result["cases"]:
                st.info("No case is eligible under these filters.")
            show_cases(result["cases"])
        elif kind == "answer":
            show_answer(result)
            st.download_button(
                "Download report (Markdown)",
                st.session_state.get("report", ""),
                file_name="legal-research-report.md",
                mime="text/markdown",
            )
            if result["ranked_cases"]:
                st.subheader("Ranked cases")
                show_cases(result["ranked_cases"])
    with validate:
        st.write(
            "Paste text containing reporter citations. Each is looked up in the indexed "
            "corpus. This checks existence only, not whether the case supports the text."
        )
        text = st.text_area("Text", height=160, key="validate-text")
        if st.button("Check citations") and text.strip():
            out = call("validate", {"text": text})
            if not out["citations"]:
                st.info("No reporter citation found.")
            for c in out["citations"]:
                if c["status"] == "in_corpus":
                    st.success(f"{c['normalised']} → {c['resolves_to']}")
                    treatment_warning(c.get("treatment"))
                elif c["status"] == "not_in_corpus":
                    st.warning(f"{c['normalised']} is not in the indexed corpus.")
                else:
                    st.info(f"{c['raw']}: reporter not recognised, cannot be checked.")
            st.caption(out["note"])
    with inspect:
        target = st.session_state.get("inspect")
        if target:
            show_case(*target)
        else:
            st.info("Open an opinion from a search result or an evidence passage.")


main()
