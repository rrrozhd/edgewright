"""Cross-encoder traversal controller (STUB — cheapest-controller floor, ~150M).

A ModernBERT cross-encoder that scores each ``(question, candidate relation)`` pair
and picks the argmax, with a learned/threshold ``STOP``. This is the cheap end of
the model-size sweep — the floor the decoder controllers are compared against on the
cost/quality frontier.

Inputs:  ``select(state, candidates)`` — question text + each candidate relation.
Outputs: an ``Action``; ``Action.score`` is the winning pair's relevance score.

Milestone: M6 (size sweep + cross-encoder floor). Model load is lazy.
"""

from __future__ import annotations

from kgat.controller.base import TraversalController
from kgat.data.schemas import Action, Relation, TraversalState


class CrossEncoderPolicyController(TraversalController):
    """Question x relation cross-encoder scorer. Not implemented (M6)."""

    def __init__(self, model_name: str, stop_threshold: float = 0.5, **kwargs: object) -> None:
        self.model_name = model_name
        self.stop_threshold = stop_threshold
        self._extra = kwargs

    def select(self, state: TraversalState, candidates: list[Relation]) -> Action:
        raise NotImplementedError(
            "CrossEncoderPolicyController.select is implemented in M6 (cross-encoder "
            "floor). Use controller=dummy until then."
        )


__all__ = ["CrossEncoderPolicyController"]
