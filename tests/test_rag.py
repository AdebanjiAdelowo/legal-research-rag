import pytest

from legalrag.rag import types as T
from legalrag.rag.abstain import Gate, gate_metrics, tune_threshold
from legalrag.rag.generate import (
    ExtractiveGenerator,
    LLMGenerator,
    ScriptedChat,
    build_prompt,
    classify_sentence,
    parse_output,
)
from legalrag.rag.pipeline import Pipeline, PipelineConfig
from legalrag.rag.report import to_markdown
from legalrag.retrieval.core import Ranking

QUESTION = "Which prong of the qualified immunity analysis should a court address first?"


def test_extractive_answer_is_verbatim_and_cited(pipeline):
    answer = pipeline.answer(QUESTION)
    assert not answer.abstained and answer.evidence
    text = {e.label: e.text.replace("\n", " ") for e in answer.evidence}
    citable = [s for s in answer.statements if s.type in T.CITABLE]
    assert citable
    for s in citable:
        assert s.quoted and len(s.citations) == 1
        assert s.text in text[s.citations[0]]
        assert s.verification["status"] == "verified"
    assert answer.verification["hallucinated_citations"] == 0
    assert answer.verification["unsupported_claim_rate"] == 0.0


def test_extractive_generator_adds_no_synthesis(pipeline):
    answer = pipeline.answer(QUESTION)
    notes = [s for s in answer.statements if s.type == T.INTERPRETATION]
    assert len(notes) == 1 and "verbatim" in notes[0].text and not notes[0].citations


def test_evidence_carries_provenance_and_authority(pipeline):
    answer = pipeline.answer(
        "irreparable harm for a preliminary injunction, binding in the Second Circuit"
    )
    assert answer.evidence
    for e in answer.evidence:
        assert e.para_ids and all(p.startswith(e.case_id) for p in e.para_ids)
        assert e.authority == "binding" and e.citation in e.reference
    assert all(e.circuit in ("2", None) for e in answer.evidence)


def test_per_case_cap(pipeline):
    answer = pipeline.answer(QUESTION)
    counts = {}
    for e in answer.evidence:
        counts[e.case_id] = counts.get(e.case_id, 0) + 1
    assert max(counts.values()) <= pipeline.config.per_case


def test_overruled_case_is_flagged_but_only_after_the_overruling(pipeline):
    answer = pipeline.answer("the order of decision for qualified immunity is mandatory")
    flagged = [e for e in answer.evidence if e.treatment]
    assert flagged and flagged[0].case_id == "cap:9003"
    assert any(
        "overruled in part" in p
        for s in answer.statements
        for p in (s.verification or {}).get("problems", [])
    )


def test_dissent_is_labelled(pipeline):
    answer = pipeline.answer("a dispute over an immaterial fact does not preclude summary judgment")
    dissent = [e for e in answer.evidence if e.opinion_type == "dissent"]
    assert dissent and not dissent[0].is_majority
    assert "DISSENT" in build_prompt("q", dissent)
    assert any(
        "dissent" in p
        for s in answer.statements
        for p in (s.verification or {}).get("problems", [])
    )


@pytest.mark.parametrize(
    "question", ["GDPR right to erasure", "Is a non-compete enforceable under Texas law?"]
)
def test_abstains_on_out_of_scope_law_without_retrieving(pipeline, question):
    answer = pipeline.answer(question)
    assert answer.abstained and not answer.evidence and not answer.statements


def test_abstains_when_filters_leave_nothing(pipeline):
    answer = pipeline.answer("standing requirements", before="1900-01-01")
    assert answer.abstained and "eligible" in answer.abstain_reason


def test_abstains_below_threshold():
    strict = Pipeline.demo(PipelineConfig(corpus="demo", abstain_threshold=1e9))
    assert strict.answer(QUESTION).abstained


def test_withholding_the_relevant_case_changes_the_evidence(pipeline):
    full = pipeline.answer(QUESTION)
    assert "cap:9004" in {e.case_id for e in full.evidence}
    withheld = pipeline.answer(QUESTION, exclude_cases=("cap:9004",))
    assert "cap:9004" not in {e.case_id for e in withheld.evidence}


def test_gate_and_threshold_tuning():
    import numpy as np

    ranking = Ranking(np.array([0, 1]), np.array([2.0, 1.0]))
    assert Gate(3.0).should_abstain(ranking)[0] and not Gate(1.5).should_abstain(ranking)[0]
    assert Gate().should_abstain(Ranking(np.array([]), np.array([])))[0]
    tuned = tune_threshold([5, 6, 7, 8], [1, 2, 3, 6.5])
    assert tuned["auroc"] > 0.8 and 3 < tuned["threshold"] <= 5
    m = gate_metrics([5, 6, 7, 8], [1, 2, 3, 6.5], tuned["threshold"])
    assert (
        m["coverage_answerable"] == 1.0
        and m["abstain_recall"] == 0.75
        and m["abstain_precision"] == 1.0
    )


def test_parse_typed_lines():
    abstained, st = parse_output(
        "[REASONING] Courts may choose the order. [E1]\n"
        "2. [FACT] The tenants sued an officer [E2, E3].\n"
        "[NOT ESTABLISHED] Whether this applies to municipalities.\n"
        "Some stray sentence without a tag."
    )
    assert not abstained
    assert [(s.type, s.citations) for s in st] == [
        (T.REASONING, ["E1"]),
        (T.FACT, ["E2", "E3"]),
        (T.NOT_ESTABLISHED, []),
        (T.INTERPRETATION, []),
    ]
    assert "[E1]" not in st[0].text


def test_parse_abstention():
    assert parse_output("INSUFFICIENT EVIDENCE") == (True, [])
    assert parse_output("  ") == (True, [])
    assert parse_output("Insufficient evidence.")[0]


def test_llm_generator_abstention_is_respected():
    p = Pipeline.demo(generator=LLMGenerator(ScriptedChat(["INSUFFICIENT EVIDENCE"])))
    answer = p.answer(QUESTION)
    assert answer.abstained and answer.raw_output == "INSUFFICIENT EVIDENCE"


def test_prompt_contains_only_supplied_evidence(pipeline):
    chat = ScriptedChat(["[REASONING] x. [E1]"])
    p = Pipeline.demo(generator=LLMGenerator(chat))
    answer = p.answer(QUESTION)
    system, user = chat.calls[0]
    assert "INSUFFICIENT EVIDENCE" in system and "legal advice" in system
    for e in answer.evidence:
        assert f"[{e.label}]" in user and e.reference in user
    assert QUESTION in user


def test_sentence_classification():
    assert classify_sentence("Under 42 U.S.C. § 1983 a person may sue.") == T.PROVISION
    assert classify_sentence("Alder sued Birch for breach of contract.") == T.FACT
    assert (
        classify_sentence("A dispute is genuine if a reasonable jury could find for the nonmovant.")
        == T.REASONING
    )
    assert classify_sentence("The company expanded.", "I/Background") == T.FACT


def test_extractive_abstains_without_overlap():
    assert ExtractiveGenerator().generate("zzz qqq", [])[0]


def test_report_states_limits_and_sources(pipeline):
    md = to_markdown(pipeline.answer(QUESTION))
    assert "not legal advice" in md and "no citator" in md and "synthetic" in md
    assert "## Evidence passages" in md and "paragraphs: cap:" in md
    assert "## Citation verification" in md and "not a legal determination" in md
    md = to_markdown(pipeline.answer("GDPR right to erasure"))
    assert "## No answer given" in md and "does not mean that no authority exists" in md


def test_config_rejects_unknown_keys(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("retriever: bm25\nnot_a_key: 1\n")
    with pytest.raises(ValueError, match="not_a_key"):
        PipelineConfig.load(path)
    path.write_text("corpus: demo\nevidence_k: 3\n")
    assert PipelineConfig.load(path).evidence_k == 3
