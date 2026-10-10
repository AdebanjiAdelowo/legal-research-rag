from legalrag.data.citations import cap_path, cite_key, find_citations, strip_citations


def test_full_citation_with_pin():
    (c,) = find_citations("See Roe v. Wade, 410 U. S. 113, 153-54 (1973).")
    assert (c.volume, c.reporter, c.page, c.pin) == (410, "us", 113, 153)
    assert c.key == "410 us 113" and c.known


def test_spacing_variants_share_a_key():
    assert cite_key("165 F.3d 1062") == cite_key("165 F. 3d 1062") == "165 f3d 1062"
    assert cite_key("550 F. Supp. 2d 100") == cite_key("550 F.Supp.2d 100")


def test_short_form_has_no_lookup_key():
    (c,) = find_citations("410 U.S. at 160")
    assert c.is_short_form and c.key is None and c.pin == 160


def test_parallel_citations_are_separate_and_pin_does_not_eat_next_volume():
    found = find_citations("477 U.S. 242, 248, 106 S.Ct. 2505, 91 L.Ed.2d 202 (1986)")
    assert [c.key for c in found] == ["477 us 242", "106 sct 2505", "91 led2d 202"]
    assert found[0].pin == 248 and found[1].pin is None


def test_longest_reporter_wins():
    (c,) = find_citations("12 F. Supp. 2d 345")
    assert c.reporter == "fsupp2d"


def test_strip_removes_citation_and_parenthetical():
    out = strip_citations(
        "held so. United States v. Harris, 165 F.3d 1062, 1067 (6th Cir.1999), and more"
    )
    assert "165" not in out and "1999" not in out and "and more" in out


def test_cap_path_only_for_hosted_reporters():
    assert cap_path("Roe v. Wade, 410 U.S. 113 (1973)") == ("us", 410, 113)
    assert cap_path("Padilla v. Kentucky, 176 L. Ed. 2d 284 (2010)") is None


def test_ordinary_numbers_are_not_citations():
    assert find_citations("In 1999 the F. Scott company paid 300 dollars.") == []
