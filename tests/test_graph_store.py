"""Tests for the graph store (kgat.graph)."""

from __future__ import annotations

import pytest

from kgat.data.schemas import Question, Triple
from kgat.data.subgraph import SubgraphRecord, parse_graph_field, record_from_raw
from kgat.graph.inmemory import InMemoryKGStore
from kgat.graph.store import KGStore


def _record(qid: str, triples: list[Triple], topic=("a",), gold=("c",)) -> SubgraphRecord:
    q = Question(
        qid=qid, text=f"q {qid}", topic_entities=topic, gold_answers=gold, dataset="sample"
    )
    return SubgraphRecord(question=q, triples=tuple(triples))


def test_abc_cannot_be_instantiated():
    with pytest.raises(TypeError):
        KGStore()  # type: ignore[abstract]


def test_relations_and_neighbors():
    rec = _record(
        "q1",
        [
            Triple("a", "r1", "b"),
            Triple("a", "r1", "b2"),
            Triple("a", "r2", "c"),
            Triple("b", "r3", "d"),
        ],
    )
    store = InMemoryKGStore.from_records([rec])
    store.load_question_subgraph("q1")

    assert set(store.relations_of("a")) == {"r1", "r2"}
    assert store.neighbors("a", "r1") == ["b", "b2"]  # order preserved
    assert store.neighbors("a", "r2") == ["c"]
    assert store.neighbors("b", "r3") == ["d"]
    # Unknown node / relation -> empty, not error.
    assert store.relations_of("zzz") == []
    assert store.neighbors("a", "no_such_rel") == []


def test_requires_scoping_first():
    store = InMemoryKGStore.from_records([_record("q1", [Triple("a", "r", "b")])])
    with pytest.raises(RuntimeError):
        store.relations_of("a")  # must load_question_subgraph first


def test_unknown_qid_raises():
    store = InMemoryKGStore.from_records([_record("q1", [Triple("a", "r", "b")])])
    with pytest.raises(KeyError):
        store.load_question_subgraph("does-not-exist")


def test_scoping_isolates_subgraphs():
    r1 = _record("q1", [Triple("a", "r1", "b")])
    r2 = _record("q2", [Triple("x", "r2", "y")])
    store = InMemoryKGStore.from_records([r1, r2])

    store.load_question_subgraph("q1")
    assert store.relations_of("a") == ["r1"]
    assert store.relations_of("x") == []  # q2's node invisible while scoped to q1

    store.load_question_subgraph("q2")
    assert store.relations_of("x") == ["r2"]
    assert store.relations_of("a") == []


def test_parse_graph_field_normalizes_and_dedupes():
    raw = [
        ["a", "r", "b"],
        ["a", "r", "b"],  # dup -> dropped
        [" a ", " r ", " c "],  # stripped
        ["bad", "row"],  # wrong arity -> dropped
        ["a", "", "b"],  # empty component -> dropped
    ]
    triples = parse_graph_field(raw)
    assert Triple("a", "r", "b") in triples
    assert Triple("a", "r", "c") in triples
    assert len(triples) == 2


def test_record_from_raw_field_aliases():
    raw = {
        "id": "s1",
        "question": "who?",
        "q_entity": ["e0"],
        "a_entity": ["g0"],
        "graph": [["e0", "rel", "g0"]],
    }
    rec = record_from_raw(raw, dataset="webqsp")
    assert rec.qid == "s1"
    assert rec.question.topic_entities == ("e0",)
    assert rec.question.gold_answers == ("g0",)
    assert rec.question.dataset == "webqsp"
    assert rec.triples == (Triple("e0", "rel", "g0"),)
