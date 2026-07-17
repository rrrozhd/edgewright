"""The controller's prompt formats — shared by mining, SFT, and inference.

One function, one format. ``mine_trajectories`` builds SFT prompts with it,
``train.sft`` tokenizes them, and ``DecoderPolicyController`` reproduces the exact
same prompt at inference — any drift between the three silently degrades the
controller, so they all import from here. The write-path extractor prompt
(``format_extraction_prompt``) follows the same rule: ``train.sft_extractor`` and
``eval.extractor_cascade`` both import it from here.

The target completion is ``target_text(relation_or_stop)`` (leading space, defined
next to the trie in ``constrained_decoding`` so the constraint and the SFT labels
tokenize identically); the extractor's completion is the triple-grammar
serialization (``encode_triples_target``).
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from kgat.controller.constrained_decoding import NONE_LABEL, STOP_TOKEN

# Cap the candidates listed in the prompt (the trie still constrains over ALL of
# them; the listing is a hint for the model, not the action space).
MAX_LISTED_CANDIDATES = 64

# Typed entity markers wrapping the FILER's mentions in the chunk (entity-marker RE
# — Zhou & Chen 2021; "Matching the Blanks"). Anchoring who the filer is is what
# relation DIRECTION hinges on ("[F]we[/F] sell to X" = supplier; "we buy from X" =
# customer), the largest genuine-error bucket in the 2026-07 human audit. Plain-text
# markers (no tokenizer surgery): the model learns their meaning from SFT. Applied
# to the model INPUT only — grounding/provenance stay on the original chunk text.
FILER_OPEN, FILER_CLOSE = "[F]", "[/F]"

# First-person references to the filer in SEC prose. IGNORECASE catches sentence-
# initial "We"/"Our". "us" is deliberately EXCLUDED here and matched separately
# case-sensitively (``_US_PRONOUN``) so the country "US"/"U.S." — extremely common in
# filings ("US GAAP", "our US operations") — is never mistaken for the filer.
_FIRST_PERSON = re.compile(
    r"\b(we|our|ourselves|the Company|the Corporation|the Registrant|the Partnership)\b",
    re.IGNORECASE,
)
_US_PRONOUN = re.compile(r"\bus\b")  # lowercase only: the pronoun, not the country "US"
_CORP_SUFFIX = re.compile(
    r"[\s,]+(inc|inc\.|incorporated|corp|corp\.|corporation|co|co\.|company|ltd|ltd\.|"
    r"llc|l\.l\.c\.|plc|lp|l\.p\.|holdings|group)\b\.?",
    re.IGNORECASE,
)


def _filer_name_pattern(filer: str) -> re.Pattern | None:
    """Regex matching the filer's name and its suffix-stripped core in the chunk.

    Case-SENSITIVE by design: proper-noun company mentions in filing prose are
    capitalized, and several filer cores are common English words ("Apple", "Gap",
    "Block", "Match", "Visa"). Case folding would mark every lowercase occurrence of
    the noun ("the gap between", "a block of shares") as the filer, injecting a
    confound into the direction signal these markers exist to sharpen.
    """
    core = _CORP_SUFFIX.sub("", filer).strip().rstrip(".,")
    variants = {v for v in (filer.strip(), core) if len(v) >= 3}
    if not variants:
        return None
    # Longest first so the full name wins over its core; match across punctuation.
    pats = [
        r"[\s]+".join(re.escape(w) for w in v.split())
        for v in sorted(variants, key=len, reverse=True)
    ]
    return re.compile(r"\b(" + "|".join(pats) + r")\b")


def mark_filer_mentions(text: str, filer: str) -> str:
    """Wrap the filer's name and first-person mentions with ``[F]…[/F]`` markers.

    Deterministic and idempotent-safe (skips spans already inside a marker is not
    needed since we mark in one pass over disjoint matches). Overlapping name /
    first-person matches are handled by marking names first, then first-person on
    the remaining text outside existing markers.
    """
    spans: list[tuple[int, int]] = []
    name_pat = _filer_name_pattern(filer)
    if name_pat is not None:
        spans.extend(m.span() for m in name_pat.finditer(text))
    spans.extend(m.span() for m in _FIRST_PERSON.finditer(text))
    spans.extend(m.span() for m in _US_PRONOUN.finditer(text))
    if not spans:
        return text
    # Merge/dedup overlapping spans, left to right.
    spans.sort()
    merged: list[tuple[int, int]] = []
    for s, e in spans:
        if merged and s < merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    out, last = [], 0
    for s, e in merged:
        out.append(text[last:s])
        out.append(f"{FILER_OPEN}{text[s:e]}{FILER_CLOSE}")
        last = e
    out.append(text[last:])
    return "".join(out)


def format_prompt(state_repr: str, candidates: Sequence[str]) -> str:
    """Render the controller input for one decision.

    ``state_repr`` comes from ``kgat.traversal.engine.serialize_state`` (question +
    step + frontier chains). The completion the model is trained/decoded to produce
    is `` <relation>`` or `` [STOP]`` immediately after the trailing ``next:``.
    """
    listed = list(candidates)[:MAX_LISTED_CANDIDATES]
    lines = [f"- {c}" for c in listed]
    hidden = len(candidates) - len(listed)
    if hidden > 0:
        lines.append(f"- ... (+{hidden} more)")
    lines.append(f"- {STOP_TOKEN}")
    return f"{state_repr}\ncandidates:\n" + "\n".join(lines) + "\nnext:"


def format_extraction_prompt(
    filer: str, text: str, relations: Sequence[str], *, mark_filer: bool = False
) -> str:
    """Render the backfill extractor's input for one filing chunk.

    The completion the model is trained/decoded to produce immediately after the
    trailing ``extraction:`` is the triple-grammar serialization — `` NONE`` or
    `` <relation> :: <target>`` items joined by `` ;`` (constrained decoding makes
    anything else impossible; the relation listing here is a hint, not the action
    space).

    ``mark_filer`` wraps the filer's mentions in the chunk with ``[F]…[/F]`` (see
    ``mark_filer_mentions``). This MUST match between SFT and inference — it is
    single-sourced here so ``train.sft_extractor`` / ``eval.extractor_cascade`` /
    ``train.grpo_routing`` cannot drift.
    """
    if mark_filer:
        text = mark_filer_mentions(text, filer)
        hint = (
            f" ({FILER_OPEN}…{FILER_CLOSE} marks the filer; "
            "judge each relation from the filer's side)"
        )
    else:
        hint = ""
    return (
        "extract the filer's company relationships from this SEC filing text.\n"
        f"filer: {filer}\n"
        f"text: {text}\n"
        f"relations: {' | '.join(relations)} (or {NONE_LABEL} if the text states none){hint}\n"
        "extraction:"
    )


__all__ = [
    "format_prompt",
    "format_extraction_prompt",
    "mark_filer_mentions",
    "FILER_OPEN",
    "FILER_CLOSE",
    "MAX_LISTED_CANDIDATES",
]
