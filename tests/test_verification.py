from legalrag.rag import types as T
from legalrag.rag.generate import LLMGenerator, ScriptedChat
from legalrag.rag.pipeline import Pipeline, PipelineConfig
from legalrag.verify.judges import LexicalJudge, windows
from legalrag.verify.verifier import CitationIndex, names_agree, quote_found

QUESTION = "order of the qualified immunity analysis"
TRUE = "[REASONING] The sequence should no longer be regarded as mandatory and judges may exercise their sound discretion. [E1]"


def run(output, policy="flag"):
    p = Pipeline.demo(
        PipelineConfig(corpus="demo", verify=policy), generator=LLMGenerator(ScriptedChat([output]))
    )
    return p.answer(QUESTION)


def test_index_resolves_any_spelling(cases):
    index = CitationIndex(cases)
    assert index.resolve("9990 us 242").case_id == "cap:9001"
    assert index.resolve("1 us 1") is None and index.resolve(None) is None


def test_supported_statement_is_verified():
    answer = run(TRUE)
    assert answer.evidence[0].case_id == "cap:9004"
    assert answer.statements[0].verification["status"] == "verified"
    assert answer.verification["citation_precision"] == 1.0


def test_invalid_marker_is_a_hallucinated_citation():
    answer = run("[REASONING] Officials are always immune. [E9]")
    check = answer.statements[0].verification
    assert check["status"] == "uncited" and check["citations"][0]["exists"] is False
    assert answer.verification["hallucinated_citations"] == 1


def test_missing_citation_counts_as_unsupported():
    answer = run("[REASONING] Courts may choose the order of the analysis.")
    assert answer.statements[0].verification["status"] == "uncited"
    assert answer.verification["unsupported_claim_rate"] == 1.0


def test_existing_citation_is_not_automatically_supporting():
    # [E1] exists and is a real passage, but says nothing about damages caps.
    answer = run(
        "[REASONING] Punitive damages against municipalities are capped at treble the compensatory award. [E1]"
    )
    cite = answer.statements[0].verification["citations"][0]
    assert cite["exists"] is True and cite["in_evidence"] is True
    assert cite["support"] == "unsupported"
    assert answer.statements[0].verification["status"] == "unsupported"
    assert answer.verification["hallucinated_citations"] == 0  # it exists; it just does not support


def test_fabricated_reporter_citation_is_not_found():
    answer = run(TRUE.replace(" [E1]", ", see Smith v. Jones, 410 U.S. 113. [E1]"))
    reporter = next(
        c for c in answer.statements[0].verification["citations"] if c["kind"] == "reporter"
    )
    assert (
        reporter["exists"] is False
        and "may be wrong or simply outside the corpus" in reporter["problems"][0]
    )
    assert answer.statements[0].verification["status"] == "unsupported"
    assert answer.verification["hallucinated_citations"] == 1


def test_real_citation_from_memory_is_flagged_as_not_from_evidence():
    # 9996 U.S. 555 is in the corpus (the standing case) but was not retrieved for this question.
    answer = run(TRUE.replace(" [E1]", " as in 9996 U.S. 555. [E1]"))
    reporter = next(
        c for c in answer.statements[0].verification["citations"] if c["kind"] == "reporter"
    )
    assert reporter["exists"] is True and reporter["in_evidence"] is False
    assert answer.verification["hallucinated_citations"] == 1


def test_name_and_citation_mismatch():
    answer = run(TRUE.replace(" [E1]", ", Corvid v. Dunmore, 9993 U.S. 223. [E1]"))
    reporter = next(
        c for c in answer.statements[0].verification["citations"] if c["kind"] == "reporter"
    )
    assert reporter["exists"] is True and reporter["name_matches"] is False
    assert answer.verification["mismatched_names"] == 1


def test_unknown_reporter_is_unresolvable_not_false(cases):
    answer = run(TRUE.replace(" [E1]", " (cf. 12 F.4th 345). [E1]"))
    reporter = next(
        c for c in answer.statements[0].verification["citations"] if c["kind"] == "reporter"
    )
    assert reporter["exists"] is False  # known reporter, absent from corpus
    assert names_agree("Garnet v. Hollis", cases["cap:9004"]) and not names_agree(
        "Smith v. Jones", cases["cap:9004"]
    )


def test_invented_quotation_is_caught():
    answer = run(
        '[REASONING] The Court said the sequence was "an absurd relic that every sensible judge had long ignored". [E1]'
    )
    check = answer.statements[0].verification
    assert check["quotes_missing"] and check["status"] == "unsupported"
    assert answer.verification["quotations_not_found"] == 1


def test_quote_matching_tolerates_ocr_noise():
    passage = "it should no longer be regarded as mandatory. Judges of the district courts should be permitted"
    assert quote_found("should no longer be regarded as mandatory", passage)
    assert quote_found(
        "should no longer be regarded as rnandatory. Judges of the district courts should be permitted",
        passage,
        0.6,
    )
    assert not quote_found("must always be regarded as binding on every court", passage)


def test_strict_policy_removes_and_discloses():
    answer = run(
        TRUE
        + "\n[REASONING] Damages are capped at treble the award. [E1]\n[INTERPRETATION] The rule changed.",
        "strict",
    )
    kinds = [s.type for s in answer.statements]
    assert kinds == [T.REASONING, T.INTERPRETATION, T.NOT_ESTABLISHED]
    assert "1 generated statement(s) were withheld" in answer.statements[-1].text
    assert answer.verification["removed"] == 1 and not answer.abstained


def test_strict_policy_abstains_when_nothing_survives():
    answer = run("[REASONING] Damages are capped at treble the award. [E1]", "strict")
    assert answer.abstained and "could be confirmed" in answer.abstain_reason


def test_interpretation_is_never_scored_as_a_cited_claim():
    answer = run(TRUE + "\n[INTERPRETATION] This probably helps defendants.")
    assert answer.statements[1].verification["status"] == "not_applicable"
    assert answer.verification["citable_statements"] == 1


def test_lexical_judge_and_its_known_blind_spot():
    judge = LexicalJudge()
    passage = "The sequence should no longer be regarded as mandatory."
    assert judge.judge("The sequence is no longer mandatory.", passage).label == "supported"
    assert judge.judge("Punitive damages are capped.", passage).label == "unsupported"
    # Same words, opposite meaning: a lexical judge cannot tell. NLI judges exist for this.
    assert (
        judge.judge("The sequence should be regarded as mandatory.", passage).label == "supported"
    )


def test_windows_include_whole_passage_first():
    text = "One is here. Two is here. Three is here. Four is here. Five is here."
    w = windows(text)
    assert w[0] == text and len(w) > 1 and all(len(x) <= len(text) for x in w)


def test_citation_inside_the_cited_passage_is_relayed_not_hallucinated():
    from legalrag.rag.types import Statement
    from legalrag.verify.verifier import summarise

    p = Pipeline.demo()
    evidence = p.answer("certiorari to reconsider the mandatory sequence announced").evidence
    label = next(e.label for e in evidence if "9992 U.S. 194" in e.text)
    # The passage itself cites 9992 U.S. 194; the statement repeats the court's sentence.
    relayed = Statement(
        T.REASONING,
        "We granted certiorari to reconsider the mandatory sequence announced in Elm Street Tenants v. Farrow, 9992 U.S. 194.",
        [label],
    )
    checks = p.verifier.check_statements([relayed], evidence)
    reporter = next(c for c in checks[0].citations if c.kind == "reporter")
    assert reporter.in_passage and not reporter.problems
    assert checks[0].status == "verified"
    assert summarise(checks)["hallucinated_citations"] == 0


def test_same_citation_against_a_passage_that_lacks_it_is_still_flagged():
    answer = run(TRUE.replace(" [E1]", " as in 9996 U.S. 555. [E1]"))
    reporter = next(
        c for c in answer.statements[0].verification["citations"] if c["kind"] == "reporter"
    )
    assert reporter["in_passage"] is False and answer.verification["hallucinated_citations"] == 1
