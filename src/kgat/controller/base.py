"""``TraversalController`` — the traversal policy interface, plus a dummy impl.

A controller picks the next relation to ``EXPAND`` or emits ``STOP``, given the
current state and the *valid* candidate relations at the frontier. The candidate
constraint is load-bearing: it is what makes a sub-1B policy viable, so the engine
ALWAYS passes real candidates and controllers must choose from them.

``DummyController`` is a GPU-free reference policy (highest-degree relation, stop
after N hops) that lets the engine, the eval harness, and the tests run end-to-end
before any model exists.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from kgat.data.schemas import Action, Relation, TraversalState
from kgat.graph.store import KGStore


class TraversalController(ABC):
    """Abstract traversal policy."""

    @abstractmethod
    def select(self, state: TraversalState, candidates: list[Relation]) -> Action:
        """Pick the next relation to ``EXPAND``, or ``STOP``.

        ``candidates`` is the valid action space at the current frontier (from the
        ``KGStore``). It is always non-empty-checked by the engine; when it *is*
        empty the engine stops, so a controller may assume any returned EXPAND
        relation is drawn from ``candidates``.
        """
        raise NotImplementedError

    def bind_store(self, store: KGStore) -> None:
        """Optional hook: the engine calls this once per question so a controller
        may consult the (question-scoped) store. Default is a no-op — neural
        controllers work purely off ``candidates`` and ignore it."""
        return None


class DummyController(TraversalController):
    """GPU-free reference policy: expand the highest-degree relation, stop after N hops.

    "Highest-degree" = the candidate relation that reaches the most distinct tails
    from the current frontier nodes (needs the store, obtained via ``bind_store``).
    Ties break lexicographically for determinism. Without a bound store it falls back
    to the lexicographically-first candidate.
    """

    def __init__(self, max_hops: int = 2) -> None:
        self.max_hops = max_hops
        self._store: KGStore | None = None

    def bind_store(self, store: KGStore) -> None:
        self._store = store

    def select(self, state: TraversalState, candidates: list[Relation]) -> Action:
        if state.step >= self.max_hops or not candidates:
            return Action.stop(score=1.0)

        if self._store is None:
            relation = sorted(candidates)[0]
            return Action.expand(relation, score=0.0)

        # Degree of a candidate = distinct tails reachable via it from the frontier.
        best_relation: Relation | None = None
        best_degree = -1
        for relation in sorted(candidates):  # sorted => deterministic tie-break
            tails: set[str] = set()
            for node in state.frontier_nodes:
                tails.update(self._store.neighbors(node, relation))
            degree = len(tails)
            if degree > best_degree:
                best_degree = degree
                best_relation = relation

        if best_relation is None or best_degree <= 0:
            return Action.stop(score=1.0)
        return Action.expand(best_relation, score=float(best_degree))


__all__ = ["TraversalController", "DummyController"]
