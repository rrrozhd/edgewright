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

from collections.abc import Sequence

from kgat.controller.constrained_decoding import NONE_LABEL, STOP_TOKEN

# Cap the candidates listed in the prompt (the trie still constrains over ALL of
# them; the listing is a hint for the model, not the action space).
MAX_LISTED_CANDIDATES = 64


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


def format_extraction_prompt(filer: str, text: str, relations: Sequence[str]) -> str:
    """Render the backfill extractor's input for one filing chunk.

    The completion the model is trained/decoded to produce immediately after the
    trailing ``extraction:`` is the triple-grammar serialization — `` NONE`` or
    `` <relation> :: <target>`` items joined by `` ;`` (constrained decoding makes
    anything else impossible; the relation listing here is a hint, not the action
    space).
    """
    return (
        "extract the filer's company relationships from this SEC filing text.\n"
        f"filer: {filer}\n"
        f"text: {text}\n"
        f"relations: {' | '.join(relations)} (or {NONE_LABEL} if the text states none)\n"
        "extraction:"
    )


__all__ = ["format_prompt", "format_extraction_prompt", "MAX_LISTED_CANDIDATES"]
