"""Building retrieval queries and relevance judgments from LePaRD.

Each LePaRD row records that a federal judge, writing opinion D, quoted a passage
from earlier opinion S, together with the text of D that precedes the quotation.
That gives a retrieval task with relevance decided by the court itself: the context
is the query, S is the relevant case and the quoted passage is the relevant
paragraph.

Three properties of the raw rows would make scores uninterpretable if ignored.

Answer leakage
    The preceding context often ends with the citation to S, and sometimes repeats
    the quotation. A lexical retriever can then match the case by its reporter
    citation or party names. Rows whose context contains the quotation are dropped,
    and in the remaining rows every reporter citation is removed together with the
    case name attached to it, and short-form references by party name ("under
    Chevron") are removed for the cited case. The unmasked text is kept as
    ``raw_text`` so that the size of the leak can be measured.

Popularity skew
    A few passages are quoted tens of thousands of times. Queries are sampled with a
    cap per passage and one query per citing opinion, so the benchmark is close to a
    macro-average over passages rather than a frequency-weighted one.

Time
    Splits are by the decision date of the citing opinion. Development queries come
    from 2000 to 2009 and test queries from 2010 onwards, so tuning never sees
    opinions later than those it is tested on. A third split of older queries
    (before 2000) exists for the temporal experiment, where most of the corpus
    postdates the query.

Relevance here means "the court quoted this case at this point". Other cases in the
corpus may state the same rule and would also be useful to a researcher; they count
as non-relevant. Scores are therefore lower bounds on usefulness.
"""

from __future__ import annotations

import hashlib
import heapq
import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from legalrag.data.citations import find_citations
from legalrag.data.courts import classify_court
from legalrag.data.schema import Case, Query

_TOKEN = re.compile(r"[a-z0-9]+")
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# A capitalised run directly before a citation: "United States v. Harris," or
# "Anderson," (short form). Connectives allowed inside a case name are listed.
_SUFFIX = r"(?:,\s+(?:Inc|Co|Corp|Ltd|LLC|L\.L\.C|N\.A|L\.P|S\.A|et al)\.?)?"
_NAME_BEFORE = re.compile(
    rf"(?:(?:[A-Z][\w.'’&-]*{_SUFFIX}|v\.|vs\.|of|the|and|for|ex|rel\.|in|re|&)\s+){{0,12}}"
    rf"[A-Z][\w.'’&-]*{_SUFFIX},?\s*$"
)
_PAREN_AFTER = re.compile(r"\s*\([^()]{0,60}?\d{4}\)")
_SIGNALS = re.compile(
    r"\b(?:See also|See, e\.g\.,|See generally|See|Cf\.|Accord|But see|Compare|"
    r"citing|quoting)\s*$"
)

SPLITS = ("history", "dev", "test")


def split_for(date: str) -> str | None:
    """Split of a citing opinion decided on ``date`` (ISO)."""
    if not _ISO_DATE.match(date):
        return None
    year = int(date[:4])
    if year < 1975:
        return None
    if year < 2000:
        return "history"
    if year < 2010:
        return "dev"
    return "test"


def tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def mask_citations(text: str) -> str:
    """Remove reporter citations, attached case names and trailing parentheticals.

    ``... held that escape qualifies. United States v. Harris, 165 F.3d 1062, 1067
    (6th Cir.1999), `` becomes ``... held that escape qualifies. ``.
    """
    out: list[str] = []
    pos = 0
    for c in find_citations(text):
        if c.start < pos:
            continue
        head = text[pos : c.start]
        m = _NAME_BEFORE.search(head)
        if m:
            head = head[: m.start()]
        head = _SIGNALS.sub("", head)
        out.append(head)
        pos = c.end
        tail = _PAREN_AFTER.match(text, pos)
        if tail:
            pos = tail.end()
    out.append(text[pos:])
    masked = "".join(out)
    masked = re.sub(r"\s+([,;.])", r"\1", masked)
    masked = re.sub(r"([.,;])(?:\s*[.,;])+", r"\1", masked)
    return re.sub(r"\s{2,}", " ", masked).strip(" ,;")


def remove_name(text: str, name: str) -> str:
    """Remove occurrences of a case name such as ``Roe v. Wade`` from ``text``."""
    parts = [re.escape(p) for p in name.split()]
    if len(parts) < 3:
        return text
    return re.sub(r"\s+".join(parts) + r",?", " ", text, flags=re.IGNORECASE)


# Words too common in case names to identify a case.
_GENERIC_PARTY = frozenset(
    [
        "united",
        "states",
        "state",
        "people",
        "commonwealth",
        "city",
        "county",
        "town",
        "board",
        "department",
        "commission",
        "company",
        "corporation",
        "corp",
        "inc",
        "incorporated",
        "co",
        "ltd",
        "llc",
        "bank",
        "national",
        "american",
        "america",
        "international",
        "association",
        "union",
        "district",
        "school",
        "university",
        "insurance",
        "group",
        "service",
        "services",
        "systems",
        "industries",
        "federal",
        "general",
        "secretary",
        "commissioner",
        "director",
        "attorney",
        "warden",
        "estate",
        "trust",
        "fund",
        "authority",
        "agency",
        "north",
        "south",
        "east",
        "west",
        "new",
        "first",
        "second",
        "third",
        "fourth",
        "fifth",
        "sixth",
        "seventh",
        "eighth",
        "ninth",
    ]
)


def party_words(name: str) -> set[str]:
    """Distinctive capitalised words of a case name: ``{"Chevron", "Natural", ...}``."""
    return {w for w in re.findall(r"[A-Z][A-Za-z'’-]{3,}", name) if w.lower() not in _GENERIC_PARTY}


def remove_party_words(text: str, name: str) -> str:
    """Remove short-form references to a case ("under Chevron, ..."), matching its
    distinctive party words as whole capitalised words, with any possessive."""
    words = party_words(name)
    if not words:
        return text
    pattern = r"\b(?:" + "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True))
    return re.sub(pattern + r")(?:['’]s)?\b,?", " ", text)


def contains_quote(context: str, quote: str, n: int = 8) -> bool:
    """Whether the context repeats the opening or closing ``n`` tokens of the quote."""
    q = tokens(quote)
    if len(q) < 4:
        return False
    c = " " + " ".join(tokens(context)) + " "
    head = " " + " ".join(q[:n]) + " "
    tail = " " + " ".join(q[-n:]) + " "
    return head in c or tail in c


def make_query_text(context: str, source_name: str, max_words: int = 120) -> str:
    """Masked query: citations and the cited case's name removed, last ``max_words``
    words kept (the text nearest the point of citation)."""
    masked = remove_name(mask_citations(context), source_name)
    masked = remove_party_words(masked, source_name)
    words = masked.split()
    return " ".join(words[-max_words:])


def _ngrams(toks: list[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)}


def locate_passage(case: Case, quotes: Iterable[str], threshold: float = 0.5) -> list[str]:
    """Ids of the paragraphs of ``case`` that hold a quoted passage.

    A paragraph matches when it contains at least ``threshold`` of the word
    trigrams of some quote (bigrams for very short quotes). OCR noise and the
    court's own alterations to a quotation ("[t]he", ellipses) mean exact matching
    would miss many passages. Only the best paragraph is returned, plus any other
    that matches equally well, since a rule is sometimes restated verbatim.
    """
    paras = [(p.para_id, tokens(p.text)) for p in case.paragraphs if not p.is_heading and p.text]
    best: dict[str, float] = {}
    for quote in quotes:
        q = tokens(quote)
        n = 3 if len(q) >= 6 else 2
        grams = _ngrams(q, n)
        if not grams:
            continue
        for para_id, toks in paras:
            if len(toks) < n:
                continue
            score = len(grams & _ngrams(toks, n)) / len(grams)
            if score > best.get(para_id, 0.0):
                best[para_id] = score
    if not best:
        return []
    top = max(best.values())
    if top < threshold:
        return []
    return [pid for pid, s in best.items() if s >= max(threshold, top - 0.02)]


def _hash(*parts: str) -> int:
    return int.from_bytes(hashlib.blake2b("|".join(parts).encode(), digest_size=8).digest(), "big")


@dataclass
class BuildStats:
    """Filter counts. Rows that could not enter the sample (their hash is larger
    than those already kept for the passage) are skipped before the text filters,
    so ``quote_in_context``, ``too_short`` and ``eligible`` count examined rows only
    and are not rates over the whole file."""

    rows: int = 0
    source_missing: int = 0
    bad_date: int = 0
    quote_in_context: int = 0
    too_short: int = 0
    eligible: int = 0


@dataclass
class _Row:
    key: int
    dest_id: str
    passage_id: str
    source_id: str
    date: str
    court: str
    text: str
    raw: str


def collect(
    rows: Iterable[dict[str, str]],
    cases: dict[str, Case],
    aliases: dict[str, str] | None = None,
    per_passage: int = 3,
    min_words: int = 25,
    multi_fraction: int = 12,
    seed: str = "lepard-v1",
):
    """One pass over the benchmark rows.

    Returns ``(candidates, quotes, multi, stats)``: per-split candidate rows capped
    per passage, up to five distinct quotations per passage, rows of a hash-selected
    subset of citing opinions (for multi-issue queries), and filter counts.
    """
    stats = BuildStats()
    heaps: dict[tuple[str, str], list[tuple[int, int, _Row]]] = defaultdict(list)
    quotes: dict[str, list[str]] = defaultdict(list)
    multi: dict[str, list[_Row]] = defaultdict(list)
    for n, row in enumerate(rows):
        stats.rows += 1
        case_id = f"cap:{row['source_id']}"
        case_id = (aliases or {}).get(case_id, case_id)
        source = cases.get(case_id)
        if source is None:
            stats.source_missing += 1
            continue
        pid = row["passage_id"]
        if len(quotes[pid]) < 5 and row["quote"] not in quotes[pid]:
            quotes[pid].append(row["quote"])
        split = split_for(row["dest_date"])
        if split is None or row["dest_date"] <= source.decision_date:
            stats.bad_date += 1
            continue
        key = _hash(seed, row["dest_id"], pid)
        heap = heaps[(split, pid)]
        in_multi = _hash(seed, "multi", row["dest_id"]) % multi_fraction == 0
        if len(heap) >= per_passage and -heap[0][0] < key and not in_multi:
            continue  # would not be kept; skip the expensive text work
        context = row["destination_context"]
        if contains_quote(context, row["quote"]):
            stats.quote_in_context += 1
            continue
        text = make_query_text(context, row["source_name"])
        if len(text.split()) < min_words:
            stats.too_short += 1
            continue
        stats.eligible += 1
        item = _Row(
            key, row["dest_id"], pid, case_id, row["dest_date"], row["dest_court"],
            text, " ".join(context.split()[-120:]),
        )  # fmt: skip
        if in_multi and len(multi[row["dest_id"]]) < 8:
            multi[row["dest_id"]].append(item)
        if len(heap) < per_passage:
            heapq.heappush(heap, (-key, n, item))
        elif -heap[0][0] > key:
            heapq.heapreplace(heap, (-key, n, item))
    candidates: dict[str, list[_Row]] = defaultdict(list)
    for (split, _pid), heap in heaps.items():
        candidates[split].extend(item for _, _, item in heap)
    return candidates, quotes, multi, stats


def _to_query(qid: str, rows: list[_Row], split: str, gold_paras, quotes, kind: str) -> Query:
    first = rows[0]
    court = classify_court(first.court)
    seen_cases: dict[str, None] = {}
    paras: dict[str, None] = {}
    qs: list[str] = []
    for r in rows:
        seen_cases.setdefault(r.source_id)
        for p in gold_paras.get(r.passage_id, []):
            paras.setdefault(p)
        qs.append((quotes.get(r.passage_id) or [""])[0])
    return Query(
        query_id=qid,
        text=" ".join(r.text for r in rows)
        if kind == "single"
        else " ".join(" ".join(r.text.split()[-60:]) for r in rows),
        raw_text=" ".join(r.raw for r in rows)
        if kind == "single"
        else " ".join(" ".join(r.raw.split()[-60:]) for r in rows),
        date=first.date,
        court=first.court,
        court_level=court.level,
        circuit=court.circuit,
        gold_cases=list(seen_cases),
        gold_paras=list(paras),
        gold_quotes=qs,
        split=split,
        kind=kind,
        parts=[" ".join(r.text.split()[-60:]) for r in rows] if kind == "multi" else [],
    )


def build_queries(
    candidates: dict[str, list[_Row]],
    quotes: dict[str, list[str]],
    multi: dict[str, list[_Row]],
    cases: dict[str, Case],
    sizes: dict[str, int],
    multi_sizes: dict[str, int],
) -> tuple[dict[str, list[Query]], dict[str, list[str]]]:
    """Sample queries per split and attach paragraph-level gold.

    Returns ``(queries by split name, gold paragraph ids by passage id)``. Split
    names are ``dev``, ``test``, ``history``, ``multi_dev`` and ``multi_test``.
    """
    gold_paras: dict[str, list[str]] = {}

    def paras_for(row: _Row) -> None:
        if row.passage_id not in gold_paras:
            gold_paras[row.passage_id] = locate_passage(
                cases[row.source_id], quotes.get(row.passage_id, [])
            )

    out: dict[str, list[Query]] = {}
    for split, rows in candidates.items():
        chosen: list[_Row] = []
        used_dest: set[str] = set()
        used_passage: dict[str, int] = defaultdict(int)
        # Round-robin over passages in hash order: first one query for as many
        # passages as possible, then a second, so no passage dominates.
        for round_ in range(3):
            for row in sorted(rows, key=lambda r: r.key):
                if len(chosen) >= sizes.get(split, 0):
                    break
                if row.dest_id in used_dest or used_passage[row.passage_id] != round_:
                    continue
                used_dest.add(row.dest_id)
                used_passage[row.passage_id] += 1
                chosen.append(row)
        for row in chosen:
            paras_for(row)
        out[split] = [
            _to_query(f"{split}-{i:05d}", [r], split, gold_paras, quotes, "single")
            for i, r in enumerate(sorted(chosen, key=lambda r: r.key))
        ]

    by_split: dict[str, list[list[_Row]]] = defaultdict(list)
    for dest_id in sorted(multi, key=lambda d: _hash("order", d)):
        rows = sorted(multi[dest_id], key=lambda r: r.key)
        # One part per cited case, and only parts whose text differs: a sentence that
        # cites two cases for the same proposition is one issue, not two.
        distinct: dict[str, _Row] = {}
        bags: list[set[str]] = []
        for r in rows:
            if r.source_id in distinct:
                continue
            bag = set(tokens(" ".join(r.text.split()[-60:])))
            if any(len(bag & other) / max(1, len(bag | other)) > 0.4 for other in bags):
                continue
            distinct[r.source_id] = r
            bags.append(bag)
        if len(distinct) < 2:
            continue
        split = split_for(rows[0].date)
        if split in ("dev", "test"):
            by_split[split].append(list(distinct.values())[:3])
    for split, groups in by_split.items():
        name = f"multi_{split}"
        groups = groups[: multi_sizes.get(split, 0)]
        for g in groups:
            for r in g:
                paras_for(r)
        out[name] = [
            _to_query(f"{name}-{i:05d}", g, name, gold_paras, quotes, "multi")
            for i, g in enumerate(groups)
        ]
    return out, gold_paras
