"""GNN relation proposer (STUB — Arch B).

A GNN over the question subgraph proposes/prunes a small set of promising relations
at each frontier, which a lightweight controller then chooses among. The idea mirrors
GNN-RAG (a GNN retriever feeding a reasoner), but here the GNN feeds the *traversal
policy* rather than a full LLM reasoner.

Inputs:  ``select(state, candidates)`` — plus the subgraph structure it was built on.
Outputs: an ``Action`` over the GNN-ranked candidate relations.

Milestone: M7 (Arch B / Arch C arms). Requires the ``.[gnn]`` extra (torch-geometric).
"""

from __future__ import annotations

from kgat.controller.base import TraversalController
from kgat.data.schemas import Action, Relation, TraversalState


class GNNProposerController(TraversalController):
    """GNN-guided relation proposer + selector. Not implemented (M7)."""

    def __init__(self, model_name: str, top_k: int = 8, **kwargs: object) -> None:
        self.model_name = model_name
        self.top_k = top_k
        self._extra = kwargs

    def select(self, state: TraversalState, candidates: list[Relation]) -> Action:
        raise NotImplementedError(
            "GNNProposerController.select is implemented in M7 (Arch B). "
            "Use controller=dummy until then."
        )


__all__ = ["GNNProposerController"]
