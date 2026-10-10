from legalrag.data.cap import heading_level, parse_case
from legalrag.data.chunking import fixed_chunks, section_chunks, split_sentences
from legalrag.data.dedup import find_duplicates, hamming, simhash
from legalrag.demo import _CASES


def test_stable_identifiers(cases):
    case = cases["cap:9001"]
    assert case.official_cite == "9990 U.S. 242" and case.year == 1986
    assert case.court_level == "scotus"
    ids = [p.para_id for p in case.paragraphs]
    assert len(ids) == len(set(ids))
    assert parse_case(_CASES[0]).paragraphs[3].para_id == case.paragraphs[3].para_id


def test_paragraph_boundaries_and_sections(cases):
    case = cases["cap:9001"]
    held = next(p for p in case.paragraphs if p.text.startswith("Summary judgment is proper"))
    assert held.section == "II" and held.opinion_type == "majority"
    assert any(p.opinion_type == "dissent" for p in case.paragraphs)
    assert any(p.opinion_type == "head_matter" for p in case.paragraphs)


def test_nested_headings(cases):
    case = cases["cap:9005"]
    sections = {p.section for p in case.paragraphs if not p.is_heading}
    assert {"II/A", "II/B"} <= sections


def test_heading_detection():
    assert heading_level("II") == (1, "II")
    assert heading_level("A.") == (2, "A")
    assert heading_level("III. DISCUSSION") == (1, "III")
    assert heading_level("Background") == (1, "Background")
    assert heading_level("A plaintiff seeking relief must show harm.") is None
    assert heading_level("I would affirm.") is None


def test_section_chunks_respect_boundaries(cases):
    for case in cases.values():
        para = {p.para_id: p for p in case.paragraphs}
        for chunk in section_chunks(case):
            members = [para[i] for i in chunk.para_ids]
            assert len({m.section for m in members}) == 1
            assert len({m.opinion_index for m in members}) == 1
            assert all(m.opinion_type != "head_matter" for m in members)


def test_section_chunks_keep_whole_paragraphs(cases):
    case = cases["cap:9001"]
    para = {p.para_id: p.text for p in case.paragraphs}
    for chunk in section_chunks(case):
        for pid in chunk.para_ids:
            assert para[pid] in chunk.text


def test_trivial_chunks_are_dropped(cases):
    texts = [c.text for c in section_chunks(cases["cap:9001"])]
    assert not any(t == "Judge Example delivered the opinion of the Court." for t in texts)


def test_long_paragraph_is_split_at_sentences(cases):
    import copy

    case = copy.deepcopy(cases["cap:9002"])
    body = next(p for p in case.paragraphs if p.text.startswith("Government officials"))
    body.text = " ".join([body.text] * 12)
    pieces = [
        c for c in section_chunks(case, size=60, max_words=90) if c.para_ids == [body.para_id]
    ]
    assert len(pieces) > 3
    assert all(len(c.text.split()) <= 60 for c in pieces)
    assert all(c.text.rstrip().endswith(".") for c in pieces)


def test_fixed_chunks_cover_text_and_record_paragraphs(cases):
    case = cases["cap:9001"]
    chunks = list(fixed_chunks(case, size=40, overlap=10))
    assert all(len(c.text.split()) <= 40 for c in chunks)
    covered = {pid for c in chunks for pid in c.para_ids}
    body = {p.para_id for p in case.paragraphs if p.opinion_index >= 0 and not p.is_heading}
    assert covered == body


def test_sentence_splitter_keeps_citations_whole():
    s = split_sentences(
        "We follow Roe v. Wade, 410 U.S. 113 (1973). The rule is settled. See id. at 5."
    )
    assert s[0].endswith("(1973).") and len(s) == 3


def test_dedup_exact_and_near(cases):
    import copy

    twin = copy.deepcopy(cases["cap:9001"])
    twin.case_id = "cap:twin"
    exact, _ = find_duplicates([cases["cap:9001"], twin, cases["cap:9002"]])
    assert exact == [("cap:9001", "cap:twin")]
    base = "the court held that summary judgment is proper where no genuine dispute exists " * 60
    assert hamming(simhash(base), simhash(base.replace("proper", "appropriate", 1))) <= 3
    assert (
        hamming(simhash(base), simhash("bankruptcy discharge and the automatic stay apply " * 60))
        > 10
    )
