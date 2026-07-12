"""In-memory ``KGStore`` over preprocessed per-question subgraphs.

Holds every question's subgraph in memory and, when scoped via
``load_question_subgraph``, exposes a directed adjacency index
``node -> relation -> [tails]`` for O(1) relation and neighbor lookups.

This is the default backend for the whole foundation — no external services, no
model deps.
"""

from __future__ import annotations

from collections.abc import Iterable

from kgat.data.schemas import Entity, Relation, Triple
from kgat.data.subgraph import SubgraphRecord
from kgat.graph.store import KGStore

# adjacency: node -> relation -> ordered unique list of tails
_Adjacency = dict[Entity, dict[Relation, list[Entity]]]


def _build_adjacency(triples: Iterable[Triple]) -> _Adjacency:
    adj: _Adjacency = {}
    for t in triples:
        by_rel = adj.setdefault(t.head, {})
        tails = by_rel.setdefault(t.relation, [])
        if t.tail not in tails:  # keep deterministic order, drop dup tails
            tails.append(t.tail)
    return adj


class InMemoryKGStore(KGStore):
    """A ``KGStore`` backed by an in-memory dict of subgraphs."""

    def __init__(self, subgraphs: dict[str, SubgraphRecord] | None = None) -> None:
        self._subgraphs: dict[str, SubgraphRecord] = dict(subgraphs or {})
        self._adjacency_cache: dict[str, _Adjacency] = {}
        self._current_qid: str | None = None
        self._current_adj: _Adjacency = {}

    # -- construction helpers -------------------------------------------------
    @classmethod
    def from_records(cls, records: Iterable[SubgraphRecord]) -> InMemoryKGStore:
        """Build a store from an iterable of ``SubgraphRecord``s (keyed by qid)."""
        return cls({rec.qid: rec for rec in records})

    def add_record(self, record: SubgraphRecord) -> None:
        self._subgraphs[record.qid] = record
        self._adjacency_cache.pop(record.qid, None)

    @property
    def qids(self) -> list[str]:
        return list(self._subgraphs)

    # -- KGStore interface ----------------------------------------------------
    def load_question_subgraph(self, qid: str) -> None:
        if qid not in self._subgraphs:
            raise KeyError(f"no subgraph loaded for qid {qid!r}")
        if qid not in self._adjacency_cache:
            self._adjacency_cache[qid] = _build_adjacency(self._subgraphs[qid].triples)
        self._current_qid = qid
        self._current_adj = self._adjacency_cache[qid]

    def relations_of(self, node: Entity) -> list[Relation]:
        self._require_scoped()
        return list(self._current_adj.get(node, {}).keys())

    def neighbors(self, node: Entity, relation: Relation) -> list[Entity]:
        self._require_scoped()
        return list(self._current_adj.get(node, {}).get(relation, []))

    # -- internals ------------------------------------------------------------
    def _require_scoped(self) -> None:
        if self._current_qid is None:
            raise RuntimeError("call load_question_subgraph(qid) before querying the store")


__all__ = ["InMemoryKGStore"]
