"""End-to-end traversal engine tests with the DummyController + DummySynthesizer.

These run with zero model deps and are the foundation's proof that the whole loop
works before any neural controller exists.
"""

from __future__ import annotations

from kgat.controller.base import DummyController
from kgat.data.schemas import Action, ActionType, Question, TraversalState, Triple
from kgat.data.subgraph import SubgraphRecord
from kgat.governance.policy import AllowAllPolicy, HopPolicy
from kgat.graph.inmemory import InMemoryKGStore
from kgat.synthesis.base import DummySynthesizer
from kgat.traversal.budget import BudgetCaps
from kgat.traversal.engine import TraversalEngine


class _AlwaysFailPolicy(HopPolicy):
    """Test-local mandatory policy that fails every hop (drives the hard-block path)."""

    name = "always_fail"
    mandatory = True

    def check(self, state: TraversalState, action: Action) -> tuple[bool, dict]:
        return False, {"reason": "test"}


def _fixture_store() -> InMemoryKGStore:
    # From topic "a": r_hi reaches {b1, b2} (degree 2, highest), r_lo reaches {c} (degree 1).
    # From b1: r_next -> d.
    triples = (
        Triple("a", "r_hi", "b1"),
        Triple("a", "r_hi", "b2"),
        Triple("a", "r_lo", "c"),
        Triple("b1", "r_next", "d"),
    )
    q = Question(
        qid="q1", text="test?", topic_entities=("a",), gold_answers=("b1",), dataset="sample"
    )
    return InMemoryKGStore.from_records([SubgraphRecord(question=q, triples=triples)])


def _question() -> Question:
    return Question(
        qid="q1", text="test?", topic_entities=("a",), gold_answers=("b1",), dataset="sample"
    )


def test_engine_runs_end_to_end():
    store = _fixture_store()
    engine = TraversalEngine(store, DummyController(max_hops=1), DummySynthesizer())
    result = engine.run(_question())

    traj = result.trajectory
    # DummyController expands the highest-degree relation (r_hi) first.
    assert traj.steps[0].action.type is ActionType.EXPAND
    assert traj.steps[0].action.relation == "r_hi"
    # After one hop the frontier tails are the answers.
    assert set(traj.predicted_answers) == {"b1", "b2"}
    assert traj.hit is True  # gold "b1" is reached


def test_engine_returns_audit_certificate():
    store = _fixture_store()
    engine = TraversalEngine(store, DummyController(max_hops=1), DummySynthesizer())
    result = engine.run(_question())

    cert = result.certificate
    assert cert.qid == "q1"
    assert cert.final_verdict is True  # no policies -> vacuously verified
    assert len(cert.hops) >= 1
    assert cert.cost is result.trajectory.cost


def test_controller_hop_limit_respected():
    store = _fixture_store()
    engine = TraversalEngine(store, DummyController(max_hops=0), DummySynthesizer())
    result = engine.run(_question())
    # max_hops=0 -> immediate STOP, no expansion; frontier stays at the topic entity.
    assert result.trajectory.predicted_answers == ("a",)
    assert result.trajectory.cost.hops == 0
    assert result.trajectory.steps[0].action.type is ActionType.STOP


def test_budget_cap_stops_traversal():
    store = _fixture_store()
    # Controller would go deep, but the budget caps hops at 1.
    engine = TraversalEngine(
        store,
        DummyController(max_hops=10),
        DummySynthesizer(),
        budget_caps=BudgetCaps(max_hops=1),
    )
    result = engine.run(_question())
    assert result.trajectory.cost.hops == 1


def test_llm_call_cost_tracks_decisions():
    store = _fixture_store()
    engine = TraversalEngine(store, DummyController(max_hops=1), DummySynthesizer())
    result = engine.run(_question())
    # One EXPAND decision + one STOP decision == 2 controller calls.
    assert result.trajectory.cost.llm_calls == 2
    assert result.trajectory.cost.wall_ms >= 0.0


def test_governance_policy_records_checks():
    store = _fixture_store()
    engine = TraversalEngine(
        store,
        DummyController(max_hops=1),
        DummySynthesizer(),
        policies=[AllowAllPolicy()],
    )
    result = engine.run(_question())
    for hop in result.certificate.hops:
        assert hop.checks_passed.get("allow_all") is True
    assert result.certificate.final_verdict is True


def test_mandatory_policy_hard_block():
    # A failed mandatory policy must hard-block the hop, fail the verdict, and prevent
    # any expansion — the governance property the whole audit story rests on.
    store = _fixture_store()
    engine = TraversalEngine(
        store,
        DummyController(max_hops=5),  # would happily expand if not blocked
        DummySynthesizer(),
        policies=[_AlwaysFailPolicy()],
    )
    result = engine.run(_question())

    cert = result.certificate
    assert cert.final_verdict is False  # mandatory policy failed
    assert cert.hops[0].checks_passed == {"always_fail": False}
    # Blocked before expanding: the decision cost 1 llm_call but no hops were taken.
    assert result.trajectory.cost.hops == 0
    assert result.trajectory.cost.llm_calls == 1
    # Frontier never advanced past the topic entity.
    assert result.trajectory.predicted_answers == ("a",)
