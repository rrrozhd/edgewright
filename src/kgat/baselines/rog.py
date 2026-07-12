"""RoG (Reasoning on Graphs) baseline wrapper (STUB — M2).

RoG plans relation paths, retrieves them from the KG, and reasons over them with an
LLM. This wrapper adapts the official released RoG checkpoint/pipeline so it can be
scored through *our* harness (same metrics + cost axis), for a like-for-like frontier
comparison.

Inputs:  a dataset split + the released RoG checkpoint/config.
Outputs: per-question predictions + a ``CostRecord``, scored by ``kgat.eval.harness``.

Milestone: M2. TODO before implementing: locate and VERIFY the official RoG repo and
its published WebQSP/CWQ Hit/F1 numbers (do NOT hardcode unverified URLs or invent
target numbers — pull them from the paper/official repo and record the tolerance).
"""

from __future__ import annotations

from kgat.data.schemas import Entity, Question


class RoGBaseline:
    """Wrapper around the official RoG pipeline. Not implemented (M2)."""

    def __init__(self, checkpoint: str, **kwargs: object) -> None:
        self.checkpoint = checkpoint
        self._extra = kwargs

    def predict(self, question: Question) -> tuple[Entity, ...]:
        raise NotImplementedError(
            "RoGBaseline is implemented in M2. Locate/verify the official RoG repo and "
            "its published WebQSP/CWQ numbers before wiring this up."
        )


__all__ = ["RoGBaseline"]
