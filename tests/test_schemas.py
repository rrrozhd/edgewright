"""Tests for the core data contracts (kgat.data.schemas)."""

from __future__ import annotations

import pytest

from kgat.data.schemas import (
    Action,
    ActionType,
    Path,
    Question,
    Trajectory,
    TrajectoryStep,
    Triple,
)
from kgat.eval.cost import CostRecord


def test_action_expand_requires_relation():
    ok = Action(ActionType.EXPAND, relation="r")
    assert ok.type is ActionType.EXPAND and ok.relation == "r"
    with pytest.raises(ValueError):
        Action(ActionType.EXPAND, relation=None)


def test_action_stop_forbids_relation():
    ok = Action(ActionType.STOP)
    assert ok.type is ActionType.STOP and ok.relation is None
    with pytest.raises(ValueError):
        Action(ActionType.STOP, relation="r")


def test_action_helpers():
    assert Action.expand("r", score=0.5) == Action(ActionType.EXPAND, "r", 0.5)
    assert Action.stop(score=0.9).type is ActionType.STOP


def test_path_current_node_empty_raises():
    # Truly empty: no triples AND no root anchor -> raises (the literal contract).
    with pytest.raises(ValueError):
        _ = Path().current_node


def test_path_current_node_root_fallback():
    # Unexpanded but anchored: current_node returns the root topic entity.
    assert Path(root="e0").current_node == "e0"
    assert len(Path(root="e0")) == 0


def test_path_current_node_nonempty_returns_tail():
    p = Path(triples=(Triple("a", "r1", "b"), Triple("b", "r2", "c")), root="a")
    assert p.current_node == "c"
    assert p.nodes == ("a", "b", "c")
    assert p.relations == ("r1", "r2")
    assert len(p) == 2


def test_triple_and_question_roundtrip():
    t = Triple("h", "r", "t")
    assert (t.head, t.relation, t.tail) == ("h", "r", "t")
    q = Question(
        qid="q1",
        text="who?",
        topic_entities=("e0",),
        gold_answers=("g0", "g1"),
        dataset="sample",
    )
    assert q.qid == "q1"
    assert q.topic_entities == ("e0",)
    assert q.gold_answers == ("g0", "g1")


def test_trajectory_construction():
    step = TrajectoryStep(
        state_repr="Q: who? || step=0",
        candidates=("r1", "r2"),
        action=Action.expand("r1"),
    )
    traj = Trajectory(
        qid="q1",
        steps=[step],
        predicted_answers=("a1",),
        hit=True,
        cost=CostRecord(hops=1, llm_calls=2),
    )
    assert traj.qid == "q1"
    assert traj.steps[0].action.relation == "r1"
    assert traj.hit is True
    assert traj.cost.hops == 1
