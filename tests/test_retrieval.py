import numpy as np
import pytest

from legalrag import demo
from legalrag.index.bm25 import BM25Index, stem, tokenize
from legalrag.retrieval.core import Ranking, case_ranking, chunk_hits, rrf, score_fusion, top_k
from legalrag.retrieval.decompose import by_enumeration, decompose
from legalrag.retrieval.query import interpret
from legalrag.retrieval.rerank import apply_scores


def test_tokenizer_stems_plurals_only():
    assert tokenize("The Statutes' requirements") == ["statute", "requirement"]
    assert stem("parties") == "party" and stem("probate") == "probate" and stem("class") == "class"


def test_bm25_ranks_the_right_case_first(store, bm25):
    for q in demo.queries():
        ranking = top_k(bm25.score(q.text), 10, store.mask(before=q.date))
        assert case_ranking(ranking, store)[0] == q.gold_cases[0]
        assert "hit" in chunk_hits(ranking.head(5), store, set(q.gold_paras))


def test_bm25_reweighting_matches_a_fresh_build(store):
    a = BM25Index.build(store.texts, k1=0.6, b=0.3)
    b = BM25Index.build(store.texts)
    b.set_params(0.6, 0.3)
    q = "qualified immunity clearly established"
    assert np.allclose(a.score(q), b.score(q))


def test_bm25_save_load_roundtrip(store, bm25, tmp_path):
    bm25.save(tmp_path, {"corpus_version": "x"})
    loaded = BM25Index.load(tmp_path)
    assert np.allclose(loaded.score("irreparable harm"), bm25.score("irreparable harm"))


def test_unknown_terms_score_zero(bm25):
    assert not bm25.score("zzzz qqqq").any()


def test_as_of_mask_hides_later_cases(store, bm25):
    q = "which prong of the qualified immunity analysis should be addressed first"
    late = case_ranking(top_k(bm25.score(q), 5, None), store)
    early = case_ranking(top_k(bm25.score(q), 5, store.mask(before="2005-01-01")), store)
    assert late[0] == "cap:9004" and "cap:9004" not in early


def test_binding_mask(store):
    mask = store.mask(binding_on_circuit="2")
    kept = {store.case_of(i).case_id for i in np.flatnonzero(mask)}
    assert "cap:9005" not in kept  # Ninth Circuit
    assert {"cap:9006", "cap:9008", "cap:9001"} <= kept  # Second Circuit and Supreme Court


def test_no_filters_means_no_mask(store):
    assert store.mask() is None


def test_top_k_respects_mask_and_drops_ineligible():
    scores = np.array([5.0, 9.0, 1.0, 7.0])
    r = top_k(scores, 3, np.array([True, False, True, False]))
    assert r.idx.tolist() == [0, 2]


def test_rrf_prefers_agreement():
    a = Ranking(np.array([1, 2, 3]), np.array([3.0, 2.0, 1.0]))
    b = Ranking(np.array([2, 4, 1]), np.array([9.0, 8.0, 7.0]))
    fused = rrf([a, b], k=60)
    assert fused.idx[0] == 2 and set(fused.idx.tolist()) == {1, 2, 3, 4}
    assert np.all(np.diff(fused.scores) <= 0)


def test_score_fusion_endpoints_and_scale_invariance():
    a = Ranking(np.array([1, 2, 3]), np.array([30.0, 20.0, 10.0]))
    b = Ranking(np.array([3, 2, 1]), np.array([0.9, 0.5, 0.1]))
    assert score_fusion(a, b, 1.0).idx[0] == 1
    assert score_fusion(a, b, 0.0).idx[0] == 3
    scaled = Ranking(a.idx, a.scores * 1000)
    assert score_fusion(a, b, 0.4).idx.tolist() == score_fusion(scaled, b, 0.4).idx.tolist()


def test_rerank_reorders_head_and_keeps_tail():
    r = Ranking(np.array([10, 11, 12, 13]), np.array([4.0, 3.0, 2.0, 1.0]))
    out = apply_scores(r, np.array([0.1, 0.9]))
    assert out.idx.tolist() == [11, 10, 12, 13]
    assert np.all(np.diff(out.scores) < 0)


@pytest.mark.parametrize(
    "question,expected",
    [
        ("qualified immunity before 2009", {"before": "2009-01-01"}),
        ("standing since 1992", {"after": "1992-01-01"}),
        ("injunctions between 1990 and 2000", {"after": "1990-01-01", "before": "2001-01-01"}),
        ("pleading standards in the 1990s", {"after": "1990-01-01", "before": "2000-01-01"}),
        ("the standard in the Ninth Circuit", {"binding_on_circuit": "9"}),
        ("binding authority in the 2d Cir. on injunctions", {"binding_on_circuit": "2"}),
        (
            "decisions of the D.C. Circuit on standing",
            {"circuits": ("DC",), "levels": ("appeals",)},
        ),
        ("summary judgment standard", {}),
    ],
)
def test_interpretation(question, expected):
    assert interpret(question).filters() == expected


def test_constraint_phrases_leave_the_retrieval_text():
    plan = interpret("What is the preliminary injunction test in the Ninth Circuit before 2008?")
    assert "Ninth" not in plan.text and "2008" not in plan.text
    assert "preliminary injunction test" in plan.text and plan.notes


def test_two_circuits_named_together():
    plan = interpret("Compare decisions of the Second and Ninth Circuits on injunctions")
    assert plan.circuits == ["2", "9"]


@pytest.mark.parametrize(
    "question",
    [
        "What does the California Supreme Court say about unconscionability?",
        "GDPR right to erasure",
        "Is a non-compete enforceable under Texas law?",
        "duty of care under English law",
    ],
)
def test_out_of_scope_is_flagged(question):
    assert interpret(question).out_of_scope


def test_federal_questions_are_in_scope():
    assert not interpret("Fourth Amendment search of a car in New York").out_of_scope


def test_decompose_splits_issues_and_keeps_the_original():
    text = (
        "Whether the search was lawful under the Fourth Amendment given the absence of a warrant. "
        "The officers also claim qualified immunity because the right was not clearly established. "
        "Finally the plaintiff seeks a preliminary injunction and must show irreparable harm."
    )
    parts = decompose(text)
    assert parts[0] == text and len(parts) >= 3
    assert decompose("standing requires injury in fact") == ["standing requires injury in fact"]


def test_enumeration_split():
    parts = by_enumeration(
        "(1) whether the claim is ripe for review; (2) whether the plaintiff has standing to sue"
    )
    assert len(parts) == 2 and "ripe" in parts[0] and "standing" in parts[1]
