"""Tests for cost accounting + aggregation (kgat.eval.cost)."""

from __future__ import annotations

import pytest

from kgat.eval.cost import CostRecord, aggregate, mean_cost
from kgat.traversal.budget import BudgetCaps, BudgetLedger


def _records() -> list[CostRecord]:
    return [
        CostRecord(llm_calls=1, hops=0),
        CostRecord(llm_calls=3, hops=2),
        CostRecord(llm_calls=2, hops=1),
    ]


def test_mean_cost():
    assert mean_cost(_records(), "llm_calls") == 2.0
    assert mean_cost(_records(), "hops") == 1.0
    assert mean_cost([], "llm_calls") == 0.0


def test_aggregate_summaries():
    agg = aggregate(_records())
    assert agg["llm_calls"]["mean"] == 2.0
    assert agg["llm_calls"]["median"] == 2.0
    assert agg["llm_calls"]["max"] == 3.0
    # p90/p99 are between median and max for this small sample.
    assert 2.0 <= agg["llm_calls"]["p90"] <= 3.0


def test_aggregate_empty_is_all_zero():
    agg = aggregate([])
    assert agg["llm_calls"] == {"mean": 0.0, "median": 0.0, "p90": 0.0, "p99": 0.0, "max": 0.0}


def test_scalar_and_unknown_axis():
    rec = CostRecord(llm_calls=5)
    assert rec.scalar("llm_calls") == 5.0
    with pytest.raises(KeyError):
        rec.scalar("bogus_axis")
    with pytest.raises(KeyError):
        mean_cost(_records(), "bogus_axis")


def test_from_ledger_roundtrip():
    ledger = BudgetLedger(caps=BudgetCaps(max_hops=3))
    ledger.charge(hops=2, llm_calls=4, prompt_tokens=10, gen_tokens=5, wall_ms=1.5)
    rec = CostRecord.from_ledger(ledger)
    assert rec.hops == 2
    assert rec.llm_calls == 4
    assert rec.prompt_tokens == 10
    assert rec.gen_tokens == 5
    assert rec.wall_ms == 1.5


def test_ledger_exhaustion():
    ledger = BudgetLedger(caps=BudgetCaps(max_hops=2))
    assert not ledger.exhausted()
    ledger.charge(hops=2)
    assert ledger.exhausted()  # inclusive cap
