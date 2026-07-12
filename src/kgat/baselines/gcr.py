"""GCR (Graph-Constrained Reasoning) baseline wrapper (STUB — M2).

GCR constrains LLM decoding with a KG-Trie so generated reasoning stays on real KG
paths. Official repo (user-provided, verify before use):
``RManLuo/graph-constrained-reasoning``. This wrapper adapts the released GCR pipeline
to score through our harness.

Inputs:  a dataset split + the released GCR checkpoint/config.
Outputs: per-question predictions + a ``CostRecord``, scored by ``kgat.eval.harness``.

Milestone: M2. TODO before implementing: confirm the repo URL above resolves and pull
the published WebQSP/CWQ Hit/F1 numbers from the paper/repo (do not invent them).
"""

from __future__ import annotations

from kgat.data.schemas import Entity, Question

# User-provided; confirm it resolves before relying on it (do not treat as verified).
OFFICIAL_REPO = "RManLuo/graph-constrained-reasoning"


class GCRBaseline:
    """Wrapper around the official GCR (KG-Trie constrained decoding) pipeline. Stub (M2)."""

    def __init__(self, checkpoint: str, **kwargs: object) -> None:
        self.checkpoint = checkpoint
        self._extra = kwargs

    def predict(self, question: Question) -> tuple[Entity, ...]:
        raise NotImplementedError(
            f"GCRBaseline is implemented in M2. Verify {OFFICIAL_REPO} and its published "
            "numbers before wiring this up."
        )


__all__ = ["GCRBaseline", "OFFICIAL_REPO"]
