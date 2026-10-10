"""What the system knows about later negative treatment of a case: very little.

A retrieved case can be highly relevant and no longer good law. Establishing that
needs a citator (Shepard's, KeyCite), which is proprietary and not part of any open
dataset used here. The corpus also stops in the late 2010s, so it cannot contain the
decisions that overruled several of its most-cited cases.

The table below is a short hand-curated list of well-known Supreme Court overrulings
and abrogations. It exists for two purposes only:

* to let the interface attach a visible warning to a handful of famous cases, and
* to measure how often such cases are returned by retrieval that has no notion of
  authority (see ``experiments/robustness.py``).

It is not a citator, it is not complete, and the absence of a case from it says
nothing about whether that case is still good law. Every answer the system produces
carries that statement.

Each entry: official citation of the affected case -> (treatment, later authority,
year of the later authority, one-line scope note).
"""

from __future__ import annotations

from legalrag.data.citations import cite_key

_TABLE: dict[str, tuple[str, str, int, str]] = {
    "163 U.S. 537": (
        "overruled",
        "Brown v. Board of Education, 347 U.S. 483",
        1954,
        "Plessy v. Ferguson: separate-but-equal doctrine rejected in public education and thereafter generally.",
    ),
    "198 U.S. 45": (
        "abrogated",
        "West Coast Hotel Co. v. Parrish, 300 U.S. 379",
        1937,
        "Lochner v. New York: liberty-of-contract review of economic regulation abandoned.",
    ),
    "261 U.S. 525": (
        "overruled",
        "West Coast Hotel Co. v. Parrish, 300 U.S. 379",
        1937,
        "Adkins v. Children's Hospital.",
    ),
    "41 U.S. 1": (
        "overruled",
        "Erie Railroad Co. v. Tompkins, 304 U.S. 64",
        1938,
        "Swift v. Tyson: general federal common law in diversity cases.",
    ),
    "316 U.S. 455": (
        "overruled",
        "Gideon v. Wainwright, 372 U.S. 335",
        1963,
        "Betts v. Brady: right to appointed counsel in state felony trials.",
    ),
    "338 U.S. 25": (
        "overruled in part",
        "Mapp v. Ohio, 367 U.S. 643",
        1961,
        "Wolf v. Colorado: exclusionary rule not applicable to the states.",
    ),
    "277 U.S. 438": (
        "overruled",
        "Katz v. United States, 389 U.S. 347",
        1967,
        "Olmstead v. United States: trespass-based limit on Fourth Amendment searches.",
    ),
    "355 U.S. 41": (
        "abrogated in part",
        "Bell Atlantic Corp. v. Twombly, 550 U.S. 544",
        2007,
        "Conley v. Gibson: the 'no set of facts' pleading standard was retired.",
    ),
    "478 U.S. 186": ("overruled", "Lawrence v. Texas, 539 U.S. 558", 2003, "Bowers v. Hardwick."),
    "482 U.S. 496": (
        "overruled",
        "Payne v. Tennessee, 501 U.S. 808",
        1991,
        "Booth v. Maryland: victim-impact evidence at capital sentencing.",
    ),
    "494 U.S. 652": (
        "overruled",
        "Citizens United v. FEC, 558 U.S. 310",
        2010,
        "Austin v. Michigan Chamber of Commerce.",
    ),
    "448 U.S. 56": (
        "abrogated",
        "Crawford v. Washington, 541 U.S. 36",
        2004,
        "Ohio v. Roberts: reliability test for testimonial hearsay under the Confrontation Clause.",
    ),
    "533 U.S. 194": (
        "overruled in part",
        "Pearson v. Callahan, 555 U.S. 223",
        2009,
        "Saucier v. Katz: mandatory two-step sequence for qualified immunity made discretionary.",
    ),
    "431 U.S. 209": (
        "overruled",
        "Janus v. AFSCME, 585 U.S. 878",
        2018,
        "Abood v. Detroit Board of Education: public-sector agency fees.",
    ),
    "410 U.S. 113": (
        "overruled",
        "Dobbs v. Jackson Women's Health Organization, 597 U.S. 215",
        2022,
        "Roe v. Wade.",
    ),
    "505 U.S. 833": (
        "overruled",
        "Dobbs v. Jackson Women's Health Organization, 597 U.S. 215",
        2022,
        "Planned Parenthood of Southeastern Pa. v. Casey.",
    ),
    "467 U.S. 837": (
        "overruled",
        "Loper Bright Enterprises v. Raimondo, 603 U.S. 369",
        2024,
        "Chevron U.S.A. Inc. v. NRDC: deference to agency interpretations of ambiguous statutes.",
    ),
    "403 U.S. 602": (
        "abandoned",
        "Kennedy v. Bremerton School District, 597 U.S. 507",
        2022,
        "Lemon v. Kurtzman: the Lemon test for Establishment Clause claims.",
    ),
    "530 U.S. 466": (
        "limited",
        "none",
        0,
        "Apprendi v. New Jersey remains good law; listed as a control and never flagged.",
    ),
}

NEGATIVE_TREATMENT: dict[str, dict[str, str | int]] = {
    cite_key(cite): {  # type: ignore[misc]
        "cite": cite,
        "treatment": treatment,
        "by": by,
        "year": year,
        "note": note,
    }
    for cite, (treatment, by, year, note) in _TABLE.items()
    if year
}

DISCLAIMER = (
    "Authority status is not verified. This system has no citator: a case shown here "
    "may have been overruled, reversed, superseded by statute or limited by later "
    "decisions. The corpus ends in the late 2010s."
)


def known_treatment(citations: list[str], as_of_year: int | None = None) -> dict | None:
    """Negative treatment recorded for a case, looked up by any of its citations.

    With ``as_of_year``, treatment that happened later than that year is not
    returned: a question framed as of 2005 should not be told about a 2022 overruling
    as though it were already law.
    """
    for cite in citations:
        entry = NEGATIVE_TREATMENT.get(cite_key(cite) or "")
        if entry and (as_of_year is None or int(entry["year"]) <= as_of_year):
            return dict(entry)
    return None
