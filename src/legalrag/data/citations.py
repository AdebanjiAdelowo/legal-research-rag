"""Parsing and normalising U.S. reporter citations.

A citation such as ``410 U.S. 113`` is treated as a (volume, reporter, page) triple.
Reporter spellings vary across opinions and OCR ("U. S.", "F. 2d", "F.3d"), so every
reporter string is reduced to a canonical key before comparison. The table below
covers the federal reporters that occur in the corpus; a citation to any other
reporter is still parsed, with ``known=False``, so that the verifier can report it
as "not resolvable" rather than silently ignoring it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# canonical key -> (display form, Caselaw Access Project slug or None)
REPORTERS: dict[str, tuple[str, str | None]] = {
    "us": ("U.S.", "us"),
    "sct": ("S. Ct.", None),
    "led": ("L. Ed.", None),
    "led2d": ("L. Ed. 2d", None),
    "f": ("F.", "f"),
    "f2d": ("F.2d", "f2d"),
    "f3d": ("F.3d", "f3d"),
    "f4th": ("F.4th", None),
    "fsupp": ("F. Supp.", "f-supp"),
    "fsupp2d": ("F. Supp. 2d", "f-supp-2d"),
    "fsupp3d": ("F. Supp. 3d", "f-supp-3d"),
    "fappx": ("F. App'x", "f-appx"),
    "frd": ("F.R.D.", "frd"),
    "br": ("B.R.", "br"),
    "usappdc": ("U.S. App. D.C.", "us-app-dc"),
    "ctcl": ("Ct. Cl.", "us-ct-cl"),
    "ccpa": ("C.C.P.A.", "ccpa"),
    "mj": ("M.J.", "mj"),
    "tc": ("T.C.", "tc"),
    "fedcl": ("Fed. Cl.", None),
    "fcas": ("F. Cas.", None),
}

# Reporter alternatives, longest first so that "F. Supp. 2d" wins over "F.".
_REPORTER_PATTERNS = [
    r"U\.\s?S\.\s?App\.\s?D\.\s?C\.",
    r"F\.\s?Supp\.\s?(?:2d|3d)",
    r"F\.\s?Supp\.",
    r"F\.\s?App(?:'|’)?x\.?",
    r"Fed\.\s?Appx\.",
    r"Fed\.\s?Cl\.",
    r"F\.\s?Cas\.",
    r"F\.\s?R\.\s?D\.",
    r"F\.\s?(?:2d|3d|4th)",
    r"L\.\s?Ed\.\s?2d",
    r"L\.\s?Ed\.",
    r"S\.\s?Ct\.",
    r"U\.\s?S\.",
    r"B\.\s?R\.",
    r"C\.\s?C\.\s?P\.\s?A\.",
    r"Ct\.\s?Cl\.",
    r"M\.\s?J\.",
    r"T\.\s?C\.",
    r"F\.",
]
_REPORTER = "(?:" + "|".join(_REPORTER_PATTERNS) + ")"

# Full form: "410 U.S. 113" optionally followed by a pin cite ", 153" or ", 153-54".
# Short form: "410 U.S. at 153".
_CITE_RE = re.compile(
    rf"(?<![\w.])(?P<volume>\d{{1,4}})\s+(?P<reporter>{_REPORTER})\s+"
    rf"(?:(?P<at>at)\s+(?P<pin_only>\d{{1,5}})|(?P<page>\d{{1,5}})"
    rf"(?:\s*,\s*(?P<pin>\d{{1,5}})(?!\d)(?:\s?[-–]\s?\d{{1,5}}(?!\d))?(?!\s+{_REPORTER}))?)"
)

_ALIASES = {"fedappx": "fappx", "fappx": "fappx"}


def reporter_key(reporter: str) -> str:
    """Canonical key for a reporter string: lower-case letters and digits only."""
    key = re.sub(r"[^a-z0-9]", "", reporter.lower())
    return _ALIASES.get(key, key)


@dataclass(frozen=True)
class Citation:
    """One reporter citation found in text."""

    volume: int
    reporter: str  # canonical key, see REPORTERS
    page: int | None  # first page of the cited case; None for a short-form citation
    pin: int | None  # pin-cited page, if any
    start: int
    end: int
    raw: str

    @property
    def known(self) -> bool:
        return self.reporter in REPORTERS

    @property
    def is_short_form(self) -> bool:
        return self.page is None

    @property
    def key(self) -> str | None:
        """Lookup key ``"410 us 113"``; None for short-form citations."""
        if self.page is None:
            return None
        return f"{self.volume} {self.reporter} {self.page}"

    @property
    def display(self) -> str:
        name = REPORTERS.get(self.reporter, (self.reporter, None))[0]
        if self.page is None:
            return f"{self.volume} {name} at {self.pin}"
        return f"{self.volume} {name} {self.page}"


def find_citations(text: str) -> list[Citation]:
    """Every reporter citation in ``text``, in order of appearance."""
    found: list[Citation] = []
    for m in _CITE_RE.finditer(text):
        short = m.group("at") is not None
        found.append(
            Citation(
                volume=int(m.group("volume")),
                reporter=reporter_key(m.group("reporter")),
                page=None if short else int(m.group("page")),
                pin=int(m.group("pin_only")) if short else _int(m.group("pin")),
                start=m.start(),
                end=m.end(),
                raw=m.group(0),
            )
        )
    return found


def _int(value: str | None) -> int | None:
    return int(value) if value else None


def cite_key(cite: str) -> str | None:
    """Lookup key for a citation string such as ``"165 F.3d 1062"``.

    Returns None when the string holds no full-form citation.
    """
    for c in find_citations(cite):
        if c.key:
            return c.key
    return None


def strip_citations(text: str) -> str:
    """Remove every reporter citation from ``text``, with any trailing court-and-year
    parenthetical such as ``(6th Cir. 1999)``."""
    out: list[str] = []
    pos = 0
    for c in find_citations(text):
        out.append(text[pos : c.start])
        pos = c.end
        tail = re.match(r"\s*\([^()]{0,40}?\d{4}\)", text[pos:])
        if tail:
            pos += tail.end()
    out.append(text[pos:])
    return re.sub(r"[ \t]{2,}", " ", "".join(out))


def cap_path(cite: str) -> tuple[str, int, int] | None:
    """``(slug, volume, first_page)`` for fetching a citation from the Caselaw Access
    Project, or None when the reporter is not hosted there."""
    for c in find_citations(cite):
        if c.page is None:
            continue
        slug = REPORTERS.get(c.reporter, (None, None))[1]
        if slug:
            return slug, c.volume, c.page
    return None
