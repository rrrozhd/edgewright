"""GNN-RAG baseline wrapper (STUB — M2).

GNN-RAG uses a GNN retriever over the KG to surface answer-bearing paths, which an
LLM reasoner then reads. This wrapper adapts the official released pipeline to score
through our harness.

Inputs:  a dataset split + the released GNN-RAG checkpoint/config.
Outputs: per-question predictions + a ``CostRecord``, scored by ``kgat.eval.harness``.

Milestone: M2. TODO before implementing: locate and VERIFY the official GNN-RAG repo
and its published WebQSP/CWQ Hit/F1 numbers (do NOT hardcode unverified URLs or
invent target numbers). Requires the ``.[gnn]`` + ``.[ml]`` extras.
"""

from __future__ import annotations

from kgat.data.schemas import Entity, Question


class GNNRAGBaseline:
    """Wrapper around the official GNN-RAG pipeline. Not implemented (M2)."""

    def __init__(self, checkpoint: str, **kwargs: object) -> None:
        self.checkpoint = checkpoint
        self._extra = kwargs

    def predict(self, question: Question) -> tuple[Entity, ...]:
        raise NotImplementedError(
            "GNNRAGBaseline is implemented in M2. Locate/verify the official GNN-RAG repo "
            "and its published numbers before wiring this up."
        )


__all__ = ["GNNRAGBaseline"]
