"""Chunk-local target candidates — the open-vocabulary grammar constraint.

The pilot constrained extraction targets to a GLOBAL name list exported with the
training data — a closed-world assumption that silently caps recall on entities
outside the training corpus and requires trie rebuilds as the universe grows.
The deployment-honest formulation: the candidate set for a chunk is the set of
capitalized spans found IN that chunk. Grounding is then a *construction*
property (everything emittable is verbatim-present, so the grounding gate and
provenance span always succeed), no corpus-level vocabulary exists at all, and a
brand-new company is extractable the day it first appears in a filing.

The extractor here is deliberately a heuristic (proper-noun runs by regex), not
an NER model: candidate PRECISION barely matters — junk candidates are only
*options*, and the trained model must actively choose them — while candidate
RECALL bounds extraction recall (a gold target missing from the candidates is
unemittable). Measured on the round-3/4 export, verbatim gold coverage is the
binding ceiling either way (~89% of teacher targets appear in their chunk).

Everything is pure string processing; no model, no network.
"""

from __future__ import annotations

import re
from collections.abc import Collection

# A capitalized token: starts uppercase, may contain internal &'-digits ("S&P",
# "O'Reilly"). No dot — periods glue spans across sentence boundaries; the cost
# is initials ("John A. Smith" splits) and "3M"-style leading digits, both
# acceptable as junk-candidate noise. No "and" connector — it merges
# coordinated entities ("NVIDIA and S&P Global") into one bogus span.
_CAP_TOKEN = r"(?:[A-Z][A-Za-z0-9&'\-]*)"
# Lowercase connectors allowed INSIDE a span ("Bank of America").
_CONNECT = r"(?:of|the|de|da|van|von|for|&)"
_SPAN_RE = re.compile(rf"\b{_CAP_TOKEN}(?:\s+(?:{_CAP_TOKEN}|{_CONNECT}\s+{_CAP_TOKEN}))*")

# Single capitalized tokens that are almost always sentence furniture in filings,
# never a target on their own. Multi-token spans are exempt (the span pattern
# already required capitalized continuation).
_STOP_WORDS = (
    "the we our us you they it its he she a an and or but if in on at as of for "
    "from to with under over during however company corporation registrant item "
    "part note see form table exhibit section management board directors officers "
    "january february march april may june july august september october november "
    "december quarter annual report business overview risk factors states"
)
_SINGLE_STOP = frozenset(_STOP_WORDS.split())

_CONNECTOR_EDGE = frozenset(_CONNECT.strip("(?:)").split("|"))


def _normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", name.lower()).strip()


def chunk_target_candidates(
    text: str,
    *,
    filer: str | None = None,
    known_people: Collection[str] = (),
    max_candidates: int = 64,
) -> list[str]:
    """Capitalized spans of ``text`` that could serve as extraction targets.

    Filters: single-token sentence furniture (``_SINGLE_STOP``), spans reducing
    to connectors, the filer itself (a filer-centric edge cannot point at the
    filer), and ``known_people`` (board-bio names — same set the judge's person
    gate uses). Order follows first appearance; capped at ``max_candidates``
    (grammar size control). Deduplicated case-sensitively by surface form.
    """
    people = {_normalize(p) for p in known_people}
    filer_norm = _normalize(filer) if filer else None

    out: list[str] = []
    seen: set[str] = set()
    for match in _SPAN_RE.finditer(text):
        span = match.group().strip()
        # Trim trailing sentence punctuation the token pattern may swallow.
        span = span.rstrip(".',&-")
        words = span.split()
        while words and words[-1].lower() in _CONNECTOR_EDGE:
            words.pop()
        if not words:
            continue
        span = " ".join(words)
        if len(words) == 1:
            w = words[0]
            # Keep single tokens only when they look entity-ish: an acronym /
            # ticker (>=2 uppercase) or a long capitalized word not in the stop set.
            if w.lower() in _SINGLE_STOP:
                continue
            if not (w.isupper() and len(w) >= 2) and len(w) < 4:
                continue
        elif all(w.lower() in _SINGLE_STOP or w.lower() in _CONNECTOR_EDGE for w in words):
            continue  # multi-token pure furniture ("The Company", "Risk Factors")
        norm = _normalize(span)
        if not norm or norm in people or (filer_norm and norm == filer_norm):
            continue
        # Also drop spans that are a strict prefix of the filer name ("Advanced
        # Micro Devices" inside "Advanced Micro Devices, Inc.").
        if filer_norm and norm and filer_norm.startswith(norm):
            continue
        if span not in seen:
            seen.add(span)
            out.append(span)
        if len(out) >= max_candidates:
            break
    return out


_SUFFIX_WORDS = (
    "incorporated corporation company holdings limited technologies "
    "inc corp co ltd llc plc lp sa ag nv group"
)
_SUFFIXES = frozenset(_SUFFIX_WORDS.split())


def _core(name: str) -> str:
    words = _normalize(name).split()
    while len(words) > 1 and words[-1] in _SUFFIXES:
        words.pop()
    return " ".join(words)


def normalize_name(name: str) -> str:
    """Resolver-style comparison key: lowercased, punctuation-free, suffix-stripped."""
    return _core(name)


def match_candidate(target: str, candidates: Collection[str]) -> str | None:
    """Map a (teacher-label) target name onto a candidate surface form.

    Exact normalized match first, then corporate-suffix-stripped match either
    way ("NVIDIA Corp" label ↔ "NVIDIA Corporation" span). Returns the matched
    CANDIDATE string (training supervision must use the emittable surface), or
    ``None`` — an unmatchable label is unlearnable under chunk-local candidates
    and is dropped by callers with a count.
    """
    norm = _normalize(target)
    core = _core(target)
    by_norm = {_normalize(c): c for c in candidates}
    if norm in by_norm:
        return by_norm[norm]
    by_core = {_core(c): c for c in candidates}
    return by_core.get(core)


__all__ = ["chunk_target_candidates", "match_candidate", "normalize_name"]
