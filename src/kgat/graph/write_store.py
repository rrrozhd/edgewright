"""``KGWriteStore`` — the write-side graph interface (DESIGN-KG-FILLING.md).

The read path scopes a ``KGStore`` per question; the write path commits proposed
edges THROUGH the governance chain (``kgat.governance.write_policy``) into a
``KGWriteStore``. Every committed edge carries an ``EdgeProvenance`` — the
chunk id, the located grounding span, which system produced it (small extractor
vs teacher escalation), its confidence, and the extractor identity. Multiple
provenances may accumulate on one edge (re-extraction from another chunk is
corroboration, not duplication); entity resolution/merging stays alphina-side.

``InMemoryKGWriteStore`` is the reference/testing backend; a PG/Neo4j adapter
implements the same ABC behind alphina's existing dedup/supersede machinery.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from kgat.data.schemas import Triple


@dataclass(frozen=True)
class EdgeProvenance:
    """Why one committed edge exists — the per-edge certificate payload."""

    chunk: str  # source chunk id
    span: tuple[int, int] | None  # grounding char offsets within the chunk
    route: str  # "extract" (small model) | "escalate" (teacher)
    confidence: float  # decode confidence (extract) or teacher confidence
    extractor: str = ""  # adapter/model identity (hash or version tag)


class KGWriteStore(ABC):
    """Abstract write-side store. Commits happen ONLY via the governance chain."""

    @abstractmethod
    def add_triple(self, triple: Triple, *, provenance: EdgeProvenance) -> None:
        """Commit one edge with its provenance record."""
        raise NotImplementedError

    @abstractmethod
    def has_triple(self, triple: Triple) -> bool:
        raise NotImplementedError

    @abstractmethod
    def provenance_of(self, triple: Triple) -> tuple[EdgeProvenance, ...]:
        """All provenance records accumulated on ``triple`` (empty if absent)."""
        raise NotImplementedError


@dataclass
class InMemoryKGWriteStore(KGWriteStore):
    """Dict-backed reference backend (tests, offline pilots)."""

    _edges: dict[Triple, list[EdgeProvenance]] = field(default_factory=dict)

    def add_triple(self, triple: Triple, *, provenance: EdgeProvenance) -> None:
        self._edges.setdefault(triple, []).append(provenance)

    def has_triple(self, triple: Triple) -> bool:
        return triple in self._edges

    def provenance_of(self, triple: Triple) -> tuple[EdgeProvenance, ...]:
        return tuple(self._edges.get(triple, ()))

    def __len__(self) -> int:
        return len(self._edges)


__all__ = ["EdgeProvenance", "KGWriteStore", "InMemoryKGWriteStore"]
