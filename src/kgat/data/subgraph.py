"""Per-question subgraph loading and normalization.

We reuse the preprocessed subgraph format the KGQA baselines release rather than
rebuilding Freebase. Each dataset record carries the question, its linked topic
entities, gold answers, and a per-question subgraph as a list of ``[head, relation,
tail]`` triples.

Expected on-disk schema (one JSON object per line, ``*.jsonl``)::

    {
      "id": "WebQTrn-0",
      "question": "what is the name of justin bieber brother",
      "q_entity": ["m.06w2sn5"],
      "a_entity": ["m.0gxnnwc"],
      "graph": [["m.06w2sn5", "people.person.sibling_s", "m.0gxnnwc"], ...]
    }

Assumed to match the ``rmanluo/RoG-webqsp`` / ``rmanluo/RoG-cwq`` HuggingFace
releases — **verify the field names against the actual release at M2** (see
``kgat.data.loaders``). Alternate field names are accepted (``q_entity`` /
``topic_entities`` / ``question_entities`` etc.) so a minor schema drift does not
break loading.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from kgat.data.schemas import Entity, Question, Triple

# Field-name aliases tolerated in raw records, most-canonical first.
_QID_KEYS = ("id", "qid", "ID", "QuestionId")
_TEXT_KEYS = ("question", "text", "ProcessedQuestion", "RawQuestion")
_TOPIC_KEYS = ("q_entity", "topic_entities", "question_entities", "q_entities", "entities")
_ANSWER_KEYS = ("a_entity", "gold_answers", "answers", "answer", "a_entities")
_GRAPH_KEYS = ("graph", "subgraph", "triples", "kg")


@dataclass(frozen=True)
class SubgraphRecord:
    """A question paired with its normalized per-question subgraph."""

    question: Question
    triples: tuple[Triple, ...]

    @property
    def qid(self) -> str:
        return self.question.qid


def _first_present(raw: Mapping[str, object], keys: Iterable[str]) -> object | None:
    for key in keys:
        if key in raw and raw[key] is not None:
            return raw[key]
    return None


def _as_str_tuple(value: object) -> tuple[str, ...]:
    """Coerce a scalar-or-list field into a tuple of cleaned strings."""
    if value is None:
        return ()
    if isinstance(value, str):
        stripped = value.strip()
        return (stripped,) if stripped else ()
    if isinstance(value, Iterable):
        out: list[str] = []
        for item in value:
            if item is None:
                continue
            s = str(item).strip()
            if s:
                out.append(s)
        return tuple(out)
    return (str(value).strip(),)


def parse_graph_field(raw_graph: object) -> tuple[Triple, ...]:
    """Parse the ``graph`` field into normalized, deduped ``Triple``s.

    Accepts each edge as a ``[head, relation, tail]`` list/tuple. Rows that are not
    length-3 or that have an empty component after stripping are dropped. Order is
    preserved; exact duplicates are removed.
    """
    if not raw_graph:
        return ()
    if not isinstance(raw_graph, Iterable):
        raise TypeError(f"graph field must be iterable, got {type(raw_graph).__name__}")

    seen: set[tuple[str, str, str]] = set()
    triples: list[Triple] = []
    for edge in raw_graph:
        if not isinstance(edge, (list, tuple)) or len(edge) != 3:
            continue
        head, relation, tail = (str(x).strip() for x in edge)
        if not head or not relation or not tail:
            continue
        key = (head, relation, tail)
        if key in seen:
            continue
        seen.add(key)
        triples.append(Triple(head=head, relation=relation, tail=tail))
    return tuple(triples)


def record_from_raw(raw: Mapping[str, object], dataset: str) -> SubgraphRecord:
    """Build a ``SubgraphRecord`` from one raw dataset dict.

    ``dataset`` is the split's dataset name ("webqsp" | "cwq" | "metaqa" | "sample").
    Raises ``ValueError`` if the record has no usable id or question text.
    """
    qid_val = _first_present(raw, _QID_KEYS)
    text_val = _first_present(raw, _TEXT_KEYS)
    if qid_val is None or text_val is None:
        raise ValueError(f"record missing id/question (keys present: {sorted(raw.keys())})")

    topic: tuple[Entity, ...] = _as_str_tuple(_first_present(raw, _TOPIC_KEYS))
    gold: tuple[Entity, ...] = _as_str_tuple(_first_present(raw, _ANSWER_KEYS))
    triples = parse_graph_field(_first_present(raw, _GRAPH_KEYS))

    question = Question(
        qid=str(qid_val),
        text=str(text_val),
        topic_entities=topic,
        gold_answers=gold,
        dataset=dataset,
    )
    return SubgraphRecord(question=question, triples=triples)


__all__ = ["SubgraphRecord", "parse_graph_field", "record_from_raw"]
