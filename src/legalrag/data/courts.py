"""Court hierarchy for the U.S. federal system.

The scope of this project is a single jurisdiction: U.S. federal courts. Within it,
authority is not uniform. A Supreme Court decision binds every federal court; a
court of appeals decision binds that circuit's own later panels and the district
courts inside the circuit, and is only persuasive elsewhere. This module encodes
that structure so that retrieval can be filtered to authority that is binding on a
given forum, and so that evaluation can report when a relevant case is merely
persuasive.

It models vertical and horizontal stare decisis only at the level of court and
circuit. It does not know whether a decision has since been overruled, reversed or
superseded by statute (see ``treatment.py`` for what little the system knows there).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

SCOTUS = "scotus"
APPEALS = "appeals"
DISTRICT = "district"
OTHER = "other"

CIRCUITS = ("1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "DC", "Fed")

_ORDINALS = {
    "first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5",
    "sixth": "6", "seventh": "7", "eighth": "8", "ninth": "9", "tenth": "10",
    "eleventh": "11",
}  # fmt: skip

# State or territory -> circuit containing its district courts.
_STATE_CIRCUIT = {
    "maine": "1", "massachusetts": "1", "new hampshire": "1", "rhode island": "1",
    "puerto rico": "1",
    "connecticut": "2", "new york": "2", "vermont": "2",
    "delaware": "3", "new jersey": "3", "pennsylvania": "3", "virgin islands": "3",
    "maryland": "4", "north carolina": "4", "south carolina": "4", "virginia": "4",
    "west virginia": "4",
    "louisiana": "5", "mississippi": "5", "texas": "5", "canal zone": "5",
    "kentucky": "6", "michigan": "6", "ohio": "6", "tennessee": "6",
    "illinois": "7", "indiana": "7", "wisconsin": "7",
    "arkansas": "8", "iowa": "8", "minnesota": "8", "missouri": "8", "nebraska": "8",
    "north dakota": "8", "south dakota": "8",
    "alaska": "9", "arizona": "9", "california": "9", "hawaii": "9", "idaho": "9",
    "montana": "9", "nevada": "9", "oregon": "9", "washington": "9", "guam": "9",
    "northern mariana islands": "9",
    "colorado": "10", "kansas": "10", "new mexico": "10", "oklahoma": "10", "utah": "10",
    "wyoming": "10",
    "alabama": "11", "florida": "11", "georgia": "11",
    "district of columbia": "DC",
}  # fmt: skip
# Longest names first so that "west virginia" is tested before "virginia".
_STATES = sorted(_STATE_CIRCUIT, key=len, reverse=True)


@dataclass(frozen=True)
class Court:
    name: str
    level: str  # SCOTUS | APPEALS | DISTRICT | OTHER
    circuit: str | None  # one of CIRCUITS, or None when not attributable


def classify_court(name: str) -> Court:
    """Place a court name in the federal hierarchy.

    Specialised and historical courts (bankruptcy panels, Court of Claims, Tax Court,
    the pre-1912 circuit courts) are returned as OTHER with no circuit: their
    precedential position is not modelled.
    """
    n = re.sub(r"\s+", " ", name.strip().lower())
    if "supreme court of the united states" in n:
        return Court(name, SCOTUS, None)
    if "court of appeals" in n:
        if "district of columbia" in n:
            return Court(name, APPEALS, "DC")
        if "federal circuit" in n:
            return Court(name, APPEALS, "Fed")
        for word, circuit in _ORDINALS.items():
            if f"{word} circuit" in n:
                return Court(name, APPEALS, circuit)
        return Court(name, OTHER, None)
    if "district court" in n and "bankruptcy" not in n:
        for state in _STATES:
            if re.search(rf"\b{state}\b", n):
                return Court(name, DISTRICT, _STATE_CIRCUIT[state])
        return Court(name, DISTRICT, None)
    return Court(name, OTHER, None)


def is_binding(authority: Court, forum: Court) -> bool:
    """Whether a decision of ``authority`` binds ``forum`` under the court hierarchy.

    Supreme Court decisions bind every federal forum. A court of appeals decision
    binds its own circuit: later panels of that court and district courts within
    it. Everything else is treated as persuasive.

    Known simplifications: the Eleventh Circuit's adoption of pre-1981 Fifth Circuit
    precedent (*Bonner v. City of Prichard*) and the Federal Circuit's adoption of
    its predecessor courts' decisions are not modelled, and nothing here reflects
    whether the authority is still good law.
    """
    if authority.level == SCOTUS:
        return True
    if authority.level == APPEALS and authority.circuit is not None:
        return forum.level in (APPEALS, DISTRICT) and forum.circuit == authority.circuit
    return False


def authority_label(authority: Court, forum: Court | None) -> str:
    """``"binding"``, ``"persuasive"`` or ``"unspecified"`` when no forum is given."""
    if forum is None:
        return "unspecified"
    return "binding" if is_binding(authority, forum) else "persuasive"
