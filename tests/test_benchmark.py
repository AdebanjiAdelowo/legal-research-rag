from legalrag.data import benchmark as B

CTX = (
    "Summary judgment is proper where no genuine issue exists. See Anderson v. Liberty Lobby, Inc., "
    "477 U.S. 242, 248, 106 S.Ct. 2505, 91 L.Ed.2d 202 (1986); Celotex Corp. v. Catrett, 477 U.S. 317 (1986). "
    "In Anderson, 477 U.S. at 250, the Court explained that the judge must ask whether a fair-minded jury "
    "could return a verdict. Under Anderson the inquiry is the same."
)


def test_mask_removes_citations_names_and_parentheticals():
    out = B.mask_citations(CTX)
    assert "477" not in out and "1986" not in out and "S.Ct." not in out
    assert "Liberty Lobby" not in out and "Celotex" not in out
    assert "fair-minded jury" in out and "Summary judgment is proper" in out


def test_short_form_party_reference_is_removed_for_the_gold_case():
    out = B.make_query_text(CTX, "Anderson v. Liberty Lobby, Inc.")
    assert "Anderson" not in out
    assert "the inquiry is the same" in out


def test_generic_party_words_are_not_stripped():
    assert B.party_words("United States v. Harris") == {"Harris"}
    text = B.remove_party_words(
        "The United States argues that Harris controls.", "United States v. Harris"
    )
    assert "United States" in text and "Harris" not in text


def test_quote_repeated_in_context_is_detected():
    quote = "otherwise involves conduct that presents a serious potential risk of physical injury"
    assert B.contains_quote(
        "it must “otherwise involves conduct that presents a serious potential risk of physical injury”",
        quote,
    )
    assert not B.contains_quote("the statute defines a violent felony broadly", quote)


def test_query_keeps_the_words_nearest_the_citation():
    text = " ".join(f"w{i}" for i in range(300))
    out = B.make_query_text(text, "A v. B", max_words=120).split()
    assert len(out) == 120 and out[-1] == "w299"


def test_splits_are_by_date():
    assert B.split_for("1999-12-31") == "history"
    assert B.split_for("2000-01-01") == "dev"
    assert B.split_for("2010-01-01") == "test"
    assert B.split_for("1960-05-05") is None and B.split_for("2010") is None


def test_locate_passage_tolerates_alterations(cases):
    case = cases["cap:9002"]
    quote = "shielded from liability for civil damages insofar as their conduct does not violate clearly established ... rights"
    (pid,) = B.locate_passage(case, [quote])
    para = next(p for p in case.paragraphs if p.para_id == pid)
    assert para.text.startswith("Government officials performing discretionary functions")
    assert (
        B.locate_passage(case, ["an entirely unrelated sentence about maritime liens and salvage"])
        == []
    )


def _row(
    dest,
    source,
    passage,
    date,
    context,
    quote="the quoted rule text that is long enough to count here",
):
    return {
        "dest_id": dest,
        "source_id": source,
        "dest_date": date,
        "dest_court": "United States District Court for the Southern District of New York",
        "source_name": "Alder v. Birch Logistics (synthetic)",
        "passage_id": passage,
        "quote": quote,
        "destination_context": context,
    }


def test_collect_filters_and_build(cases):
    long = " ".join(["The moving party contends that the record shows no dispute"] * 6)
    other = " ".join(["Officials claim immunity because the right was not clearly established"] * 6)
    rows = [
        _row("1", "9001", "9001_1", "2012-01-01", long),
        _row("2", "9001", "9001_1", "2012-02-01", "too short"),
        _row("3", "9001", "9001_1", "1980-01-01", long),  # before the source case: impossible
        _row("4", "424242", "x_1", "2012-01-01", long),  # source not in corpus
        _row(
            "5",
            "9001",
            "9001_1",
            "2012-03-01",
            long + " the quoted rule text that is long enough to count here",
        ),
        _row("6", "9002", "9002_1", "2005-01-01", other),
    ]
    candidates, quotes, multi, stats = B.collect(rows, cases, multi_fraction=10**9)
    assert stats.rows == 6 and stats.source_missing == 1 and stats.bad_date == 1
    assert stats.quote_in_context == 1 and stats.too_short == 1 and stats.eligible == 2
    queries, _ = B.build_queries(candidates, quotes, multi, cases, {"test": 10, "dev": 10}, {})
    assert [q.gold_cases for q in queries["test"]] == [["cap:9001"]]
    assert queries["dev"][0].circuit == "2" and queries["dev"][0].date == "2005-01-01"


def test_sampling_is_one_query_per_citing_opinion(cases):
    long = " ".join(["The moving party contends that the record shows no dispute"] * 6)
    rows = [_row("same", "9001", f"9001_{i}", "2012-01-01", long) for i in range(5)]
    candidates, quotes, multi, _ = B.collect(rows, cases, multi_fraction=10**9)
    queries, _ = B.build_queries(candidates, quotes, multi, cases, {"test": 10}, {})
    assert len(queries["test"]) == 1
