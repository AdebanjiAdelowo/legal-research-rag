from legalrag.data.courts import (
    APPEALS,
    DISTRICT,
    OTHER,
    SCOTUS,
    authority_label,
    classify_court,
    is_binding,
)

SC = classify_court("Supreme Court of the United States")
CA6 = classify_court("United States Court of Appeals for the Sixth Circuit")
CA9 = classify_court("United States Court of Appeals for the Ninth Circuit")
EDMI = classify_court("United States District Court for the Eastern District of Michigan")


def test_levels_and_circuits():
    assert (SC.level, CA6.level, EDMI.level) == (SCOTUS, APPEALS, DISTRICT)
    assert (CA6.circuit, EDMI.circuit) == ("6", "6")
    assert (
        classify_court(
            "United States Court of Appeals for the District of Columbia Circuit"
        ).circuit
        == "DC"
    )
    assert classify_court("United States Court of Appeals for the Federal Circuit").circuit == "Fed"


def test_west_virginia_is_not_virginia_but_same_circuit_anyway():
    wv = classify_court("United States District Court for the Southern District of West Virginia")
    assert wv.circuit == "4"
    assert (
        classify_court("United States District Court for the District of Columbia").circuit == "DC"
    )


def test_specialised_courts_are_not_placed():
    assert classify_court("United States Bankruptcy Court for the District of Maine").level == OTHER
    assert classify_court("United States Court of Federal Claims").circuit is None


def test_supreme_court_binds_everyone():
    assert is_binding(SC, EDMI) and is_binding(SC, CA9)


def test_circuit_binds_itself_and_its_districts_only():
    assert is_binding(CA6, EDMI) and is_binding(CA6, CA6)
    assert not is_binding(CA9, EDMI) and not is_binding(CA6, CA9)
    assert not is_binding(CA6, SC)


def test_district_court_binds_no_one():
    assert not is_binding(EDMI, EDMI)


def test_label():
    assert authority_label(CA9, EDMI) == "persuasive"
    assert authority_label(CA6, EDMI) == "binding"
    assert authority_label(CA6, None) == "unspecified"
