"""``KGStore`` — the graph backend interface.

Everything the traversal engine and controllers need from the knowledge graph goes
through this ABC: what relations leave a node, what nodes a relation reaches, and
scoping the store to a single question's subgraph. The default backend is
``InMemoryKGStore`` over preprocessed subgraphs; a Neo4j adapter lives behind the
same interface (stubbed this pass).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from kgat.data.schemas import Entity, Relation


class KGStore(ABC):
    """Abstract knowledge-graph store, scoped one question at a time."""

    @abstractmethod
    def relations_of(self, node: Entity) -> list[Relation]:
        """Return the outgoing relation labels available at ``node``.

        This is the *valid action space* offered to the controller at a frontier
        node — the constraint that makes a sub-1B policy viable.
        """
        raise NotImplementedError

    @abstractmethod
    def neighbors(self, node: Entity, relation: Relation) -> list[Entity]:
        """Return the tail nodes reachable from ``node`` via ``relation``."""
        raise NotImplementedError

    @abstractmethod
    def load_question_subgraph(self, qid: str) -> None:
        """Scope the store to a single question's subgraph.

        After this call, ``relations_of`` / ``neighbors`` answer against that
        question's subgraph only. Implementations should raise ``KeyError`` for an
        unknown ``qid``.
        """
        raise NotImplementedError


__all__ = ["KGStore"]
