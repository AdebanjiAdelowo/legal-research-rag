"""A small synthetic corpus for tests, continuous integration and the demo mode of
the application.

Every case here is invented. Names end in "(synthetic)", courts are real court names
so that the hierarchy logic can be exercised, and reporter volumes are 9990 and
above, which no federal reporter has reached. Nothing in this file states the law;
the text only imitates the shape of an opinion closely enough to test parsing,
chunking, retrieval, generation and citation checking without any download.
"""

from __future__ import annotations

from legalrag.data.cap import parse_case
from legalrag.data.chunking import chunk_cases
from legalrag.data.schema import Case, Chunk, Query
from legalrag.index.store import ChunkStore

VERSION = "demo-0001"

_SCOTUS = "Supreme Court of the United States"
_CA2 = "United States Court of Appeals for the Second Circuit"
_CA9 = "United States Court of Appeals for the Ninth Circuit"


def _raw(cid, name, date, court, cite, majority, cites=(), dissent=None, head=""):
    opinions = [{"type": "majority", "author": "Judge Example", "text": majority}]
    if dissent:
        opinions.append({"type": "dissent", "author": "Judge Other", "text": dissent})
    return {
        "id": cid,
        "name": name,
        "name_abbreviation": name,
        "decision_date": date,
        "docket_number": f"No. {cid}",
        "citations": [{"type": "official", "cite": cite}],
        "court": {"name": court},
        "jurisdiction": {"name_long": "United States"},
        "cites_to": [{"cite": c, "case_ids": []} for c in cites],
        "casebody": {"head_matter": head, "opinions": opinions},
    }


_CASES = [
    _raw(
        9001,
        "Alder v. Birch Logistics (synthetic)",
        "1986-06-25",
        _SCOTUS,
        "9990 U.S. 242",
        "Judge Example delivered the opinion of the Court.\n"
        "I\n"
        "Alder sued Birch Logistics for breach of a haulage contract. After discovery, Birch moved for summary judgment, and the district court granted the motion.\n"
        "II\n"
        "Summary judgment is proper only where there is no genuine dispute as to any material fact. A dispute is genuine if the evidence is such that a reasonable jury could return a verdict for the nonmoving party.\n"
        "The judge's function at this stage is not to weigh the evidence and determine the truth of the matter but to determine whether there is a genuine issue for trial. Credibility determinations are for the jury.\n"
        "III\n"
        "Because the record contains conflicting testimony about delivery dates, a reasonable jury could find for Alder. The judgment is reversed.",
        dissent="Judge Other, dissenting.\nI would affirm. The conflicting testimony concerns a fact that is not material to the contract claim, and a dispute over an immaterial fact does not preclude summary judgment.",
        head="Alder v. Birch Logistics (synthetic)\nSyllabus. Held: summary judgment requires the absence of a genuine dispute of material fact.",
    ),
    _raw(
        9002,
        "Corvid v. Dunmore (synthetic)",
        "1982-06-24",
        _SCOTUS,
        "9991 U.S. 800",
        "Judge Example delivered the opinion of the Court.\n"
        "I\n"
        "Corvid alleges that Dunmore, a state official, ordered his dismissal in retaliation for testimony before a legislative committee.\n"
        "II\n"
        "Government officials performing discretionary functions are shielded from liability for civil damages insofar as their conduct does not violate clearly established statutory or constitutional rights of which a reasonable person would have known.\n"
        "Reliance on the objective reasonableness of an official's conduct, as measured by reference to clearly established law, should avoid excessive disruption of government and permit the resolution of many insubstantial claims on summary judgment.\n"
        "III\n"
        "The case is remanded for application of this standard.",
    ),
    _raw(
        9003,
        "Elm Street Tenants v. Farrow (synthetic)",
        "2001-06-18",
        _SCOTUS,
        "9992 U.S. 194",
        "Judge Example delivered the opinion of the Court.\n"
        "I\n"
        "Tenants sued Officer Farrow for the use of excessive force during an eviction.\n"
        "II\n"
        "A court required to rule upon the qualified immunity issue must consider first whether the facts alleged show the officer's conduct violated a constitutional right. Only if a violation could be made out does the court ask whether the right was clearly established.\n"
        "This order of decision is mandatory. It serves to elaborate the law from case to case.\n"
        "III\n"
        "Reversed.",
        cites=["9991 U.S. 800"],
    ),
    _raw(
        9004,
        "Garnet v. Hollis (synthetic)",
        "2009-01-21",
        _SCOTUS,
        "9993 U.S. 223",
        "Judge Example delivered the opinion of the Court.\n"
        "I\n"
        "We granted certiorari to reconsider the mandatory sequence announced in Elm Street Tenants v. Farrow, 9992 U.S. 194.\n"
        "II\n"
        "On reconsidering the procedure required in Elm Street Tenants, we conclude that, while the sequence set forth there often is appropriate, it should no longer be regarded as mandatory. Judges of the district courts and the courts of appeals should be permitted to exercise their sound discretion in deciding which of the two prongs of the qualified immunity analysis should be addressed first.\n"
        "The rigid order of decision sometimes results in a substantial expenditure of scarce judicial resources on difficult questions that have no effect on the outcome of the case.\n"
        "III\n"
        "The judgment is reversed.",
        cites=["9992 U.S. 194", "9991 U.S. 800"],
    ),
    _raw(
        9005,
        "Ironwood Fisheries v. Juniper (synthetic)",
        "1994-03-02",
        _CA9,
        "9994 F.3d 1101",
        "EXAMPLE, Circuit Judge:\n"
        "I. BACKGROUND\n"
        "Ironwood Fisheries sought to enjoin enforcement of a seasonal closure order.\n"
        "II. DISCUSSION\n"
        "A\n"
        "A plaintiff seeking a preliminary injunction must establish that he is likely to succeed on the merits, that he is likely to suffer irreparable harm in the absence of preliminary relief, that the balance of equities tips in his favor, and that an injunction is in the public interest.\n"
        "B\n"
        "In this circuit, serious questions going to the merits and a balance of hardships that tips sharply toward the plaintiff can support issuance of an injunction, so long as the other two elements are also met.\n"
        "III. CONCLUSION\n"
        "The order denying the injunction is vacated.",
    ),
    _raw(
        9006,
        "Kestrel Shipping v. Larch (synthetic)",
        "1997-09-10",
        _CA2,
        "9995 F.3d 77",
        "EXAMPLE, Circuit Judge:\n"
        "I\n"
        "Kestrel Shipping appeals the denial of a preliminary injunction against a former employee.\n"
        "II\n"
        "In this circuit a party seeking a preliminary injunction must show irreparable harm and either a likelihood of success on the merits or sufficiently serious questions going to the merits to make them a fair ground for litigation, with the balance of hardships tipping decidedly in the movant's favor.\n"
        "Irreparable harm is the single most important prerequisite for the issuance of a preliminary injunction. A monetary loss that can be compensated by damages is not irreparable.\n"
        "III\n"
        "Affirmed.",
    ),
    _raw(
        9007,
        "Maple v. Northgate Hospital (synthetic)",
        "1992-06-12",
        _SCOTUS,
        "9996 U.S. 555",
        "Judge Example delivered the opinion of the Court.\n"
        "I\n"
        "Maple challenges a regulation governing hospital reimbursement but does not allege that she has been or will be denied treatment.\n"
        "II\n"
        "The irreducible constitutional minimum of standing contains three elements. The plaintiff must have suffered an injury in fact that is concrete and particularized and actual or imminent; there must be a causal connection between the injury and the conduct complained of; and it must be likely that the injury will be redressed by a favorable decision.\n"
        "The party invoking federal jurisdiction bears the burden of establishing these elements.\n"
        "III\n"
        "Because Maple alleges no concrete injury, the complaint must be dismissed.",
    ),
    _raw(
        9008,
        "Osprey Mutual v. Pell (synthetic)",
        "2003-11-05",
        _CA2,
        "9997 F.3d 310",
        "EXAMPLE, Circuit Judge:\n"
        "I\n"
        "Osprey Mutual denied coverage under a marine insurance policy, and Pell sued.\n"
        "II\n"
        "Under the law of this circuit, ambiguous terms in an insurance contract are construed against the insurer who drafted them. A term is ambiguous when it is reasonably susceptible to more than one interpretation.\n"
        "Whether a contract term is ambiguous is a question of law for the court.\n"
        "III\n"
        "The exclusion clause is ambiguous and is construed in favor of coverage. Affirmed.",
        cites=["9990 U.S. 242"],
    ),
]

# Negative treatment inside the synthetic corpus, mirroring treatment.NEGATIVE_TREATMENT.
DEMO_TREATMENT = {
    "9992 us 194": {
        "cite": "9992 U.S. 194",
        "treatment": "overruled in part",
        "by": "Garnet v. Hollis (synthetic), 9993 U.S. 223",
        "year": 2009,
        "note": "Mandatory two-step sequence made discretionary (synthetic example).",
    }
}


def cases() -> dict[str, Case]:
    parsed = [parse_case(raw, source_url="demo://synthetic") for raw in _CASES]
    return {c.case_id: c for c in parsed}


def chunks(scheme: str = "section", size: int = 220) -> list[Chunk]:
    return list(chunk_cases(cases().values(), scheme, size=size))


def store(scheme: str = "section", size: int = 220) -> ChunkStore:
    return ChunkStore.from_parts(f"{scheme}{size}", VERSION, chunks(scheme, size), cases())


def queries() -> list[Query]:
    def q(i, text, date, court, level, circuit, gold, para_text):
        corpus = cases()
        paras = [p.para_id for p in corpus[gold].paragraphs if para_text in p.text]
        return Query(f"demo-{i}", text, text, date, court, level, circuit, [gold], paras,
                     [para_text], "demo")  # fmt: skip

    district = "United States District Court for the Southern District of New York"
    return [
        q(0, "The moving party says no reasonable jury could find for the plaintiff on this record, so there is no genuine dispute of material fact.", "2005-01-10", district, "district", "2", "cap:9001", "genuine dispute as to any material fact"),
        q(1, "The official argues she is shielded from damages because the right was not clearly established at the time.", "2005-03-01", district, "district", "2", "cap:9002", "clearly established statutory or constitutional rights"),
        q(2, "Plaintiff has not alleged a concrete and particularized injury in fact that a favorable decision would redress.", "2006-05-05", _CA9, "appeals", "9", "cap:9007", "irreducible constitutional minimum of standing"),
        q(3, "The movant must show irreparable harm; lost profits compensable in damages do not qualify.", "2010-02-02", district, "district", "2", "cap:9006", "Irreparable harm is the single most important prerequisite"),
        q(4, "Courts may decide which prong of the immunity analysis to address first.", "2012-07-07", _CA9, "appeals", "9", "cap:9004", "should no longer be regarded as mandatory"),
    ]  # fmt: skip
