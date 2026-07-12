"""The controller's prompt format — shared by mining, SFT, and inference.

One function, one format. ``mine_trajectories`` builds SFT prompts with it,
``train.sft`` tokenizes them, and ``DecoderPolicyController`` reproduces the exact
same prompt at inference — any drift between the three silently degrades the
controller, so they all import from here.

The target completion is ``target_text(relation_or_stop)`` (leading space, defined
next to the trie in ``constrained_decoding`` so the constraint and the SFT labels
tokenize identically).
"""

from __future__ import annotations

from collections.abc import Sequence

from kgat.controller.constrained_decoding import STOP_TOKEN

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


__all__ = ["format_prompt", "MAX_LISTED_CANDIDATES"]
