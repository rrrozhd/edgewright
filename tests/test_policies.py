"""Tests for the concrete governance policies."""

from __future__ import annotations

import pytest

from kgat.data.schemas import Action, Path, Question, TraversalState
from kgat.governance.policy import (
    AllowAllPolicy,
    ConfidenceFloorPolicy,
    ProvenanceRequiredPolicy,
    RelationAllowlistPolicy,
    build_policies,
)
from kgat.traversal.budget import BudgetLedger


def _state() -> TraversalState:
    q = Question(qid="q", text="?", topic_entities=("a",), gold_answers=("b",), dataset="sample")
    return TraversalState(question=q, frontier=[Path(root="a")], step=0, budget=BudgetLedger())


def test_relation_allowlist_patterns():
    policy = RelationAllowlistPolicy(("people.person.*", "location.capital"))
    ok, detail = policy.check(_state(), Action.expand("people.person.sibling"))
    assert ok and detail["matched_pattern"] == "people.person.*"
    ok, _ = policy.check(_state(), Action.expand("location.capital"))
    assert ok
    ok, detail = policy.check(_state(), Action.expand("finance.transfer"))
    assert not ok and detail["matched_pattern"] is None


def test_stop_always_passes_every_policy():
    stop = Action.stop()
    for policy in (
        RelationAllowlistPolicy(("nothing-matches",)),
        ConfidenceFloorPolicy(0.99),
        ProvenanceRequiredPolicy(),
    ):
        ok, _ = policy.check(_state(), stop)
        assert ok, f"{policy.name} must not veto STOP"


def test_confidence_floor():
    policy = ConfidenceFloorPolicy(0.5)
    assert policy.check(_state(), Action.expand("r", score=0.7))[0]
    assert not policy.check(_state(), Action.expand("r", score=0.3))[0]


def test_provenance_required():
    # No lookup configured -> every expansion fails (fail-closed).
    ok, detail = ProvenanceRequiredPolicy().check(_state(), Action.expand("r"))
    assert not ok and detail["reason"] == "no_provenance_source_configured"
    # With a lookup returning sources for the frontier's edge -> passes.
    sourced = ProvenanceRequiredPolicy(lookup=lambda node, rel: ("doc-1",))
    ok, detail = sourced.check(_state(), Action.expand("r"))
    assert ok and detail["n_sources"] == 1


def test_build_policies_from_config():
    policies = build_policies(
        {"policies": ["relation_allowlist", "confidence_floor"], "confidence_floor": 0.2}
    )
    assert [p.name for p in policies] == ["relation_allowlist", "confidence_floor"]
    assert policies[1].floor == 0.2
    # Enabled-but-empty keeps the audit path exercised.
    assert [p.name for p in build_policies({})] == ["allow_all"]
    with pytest.raises(KeyError):
        build_policies({"policies": ["bogus_policy"]})


def test_allow_all():
    ok, _ = AllowAllPolicy().check(_state(), Action.expand("anything"))
    assert ok
