"""Neo4j-backed ``KGStore`` adapter (STUB).

Optional graph backend for traversal over a live Neo4j instance instead of
preprocessed in-memory subgraphs. Behind the same ``KGStore`` interface so it is a
config choice, not a code fork.

Milestone: not scheduled for M0-M9's core path; implement when a persistent/large
KG backend is needed (e.g. the transfer-KG experiments in M9). Connection details
come from ``NEO4J_URI`` / ``NEO4J_USER`` / ``NEO4J_PASSWORD`` (see ``.env.example``)
via Hydra config — never hardcoded.
"""

from __future__ import annotations

from kgat.data.schemas import Entity, Relation
from kgat.graph.store import KGStore


class Neo4jKGStore(KGStore):
    """``KGStore`` over a Neo4j database. Not implemented this pass.

    Inputs (at construction, via config): bolt URI, credentials, and the Cypher
    templates for relation/neighbor lookups. ``load_question_subgraph`` would scope
    queries to a question's entity neighborhood (e.g. by a subgraph tag or an
    n-hop expansion around the topic entities).
    """

    def __init__(self, uri: str, user: str, password: str, **kwargs: object) -> None:
        raise NotImplementedError(
            "Neo4jKGStore is a stub. Implement the neo4j driver + Cypher lookups when "
            "a live KG backend is required (see module docstring)."
        )

    def relations_of(self, node: Entity) -> list[Relation]:
        raise NotImplementedError

    def neighbors(self, node: Entity, relation: Relation) -> list[Entity]:
        raise NotImplementedError

    def load_question_subgraph(self, qid: str) -> None:
        raise NotImplementedError


__all__ = ["Neo4jKGStore"]
