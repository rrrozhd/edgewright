"""Tests for the cost-penalized reward (kgat.train.reward).

The invariants here are load-bearing for every downstream RL experiment: correctness
must dominate cost at the default lambda, and premature stopping must never out-reward
a correct deeper answer.
"""

from __future__ import annotations

import math

import pytest

from kgat.eval.cost import CostRecord
from kgat.train.reward import DEFAULT_LAMBDA, compute_reward, normalized_cost

GOLD = ("g",)


def _cost(llm_calls: int) -> CostRecord:
    return CostRecord(llm_calls=llm_calls)


def test_correct_cheap_beats_correct_expensive():
    cheap = compute_reward(("g",), GOLD, _cost(2), lam=DEFAULT_LAMBDA)
    expensive = compute_reward(("g",), GOLD, _cost(18), lam=DEFAULT_LAMBDA)
    assert cheap > expensive


def test_correct_expensive_beats_wrong_cheap():
    correct_expensive = compute_reward(("g",), GOLD, _cost(18), lam=DEFAULT_LAMBDA)
    wrong_cheap = compute_reward(("x",), GOLD, _cost(1), lam=DEFAULT_LAMBDA)
    assert correct_expensive > wrong_cheap


def test_lambda_zero_ignores_cost():
    cheap = compute_reward(("g",), GOLD, _cost(2), lam=0.0)
    expensive = compute_reward(("g",), GOLD, _cost(18), lam=0.0)
    assert cheap == expensive == 1.0


def test_premature_stop_never_beats_correct_deeper():
    # Premature STOP: answered nothing (or wrong) but cheaply.
    premature = compute_reward((), GOLD, _cost(1), lam=DEFAULT_LAMBDA)
    correct_deeper = compute_reward(("g",), GOLD, _cost(18), lam=DEFAULT_LAMBDA)
    assert correct_deeper > premature


def test_hit_correctness_mode():
    r = compute_reward(("g", "other"), GOLD, _cost(0), lam=DEFAULT_LAMBDA, correctness="hit")
    assert r == 1.0  # hit=1, zero cost
    r2 = compute_reward(("other",), GOLD, _cost(0), lam=DEFAULT_LAMBDA, correctness="hit")
    assert r2 == 0.0


def test_normalized_cost_clamps():
    assert normalized_cost(_cost(5), cost_cap=20.0, axis="llm_calls") == 0.25
    assert normalized_cost(_cost(100), cost_cap=20.0, axis="llm_calls") == 1.0  # clamped
    assert normalized_cost(0, cost_cap=20.0, axis="llm_calls") == 0.0


def test_reward_reduces_to_correctness_minus_penalty():
    # f1=1, cost 10/20=0.5, lam=0.2 -> 1 - 0.2*0.5 = 0.9
    r = compute_reward(("g",), GOLD, _cost(10), lam=0.2, cost_cap=20.0)
    assert math.isclose(r, 0.9)


def test_invalid_args_raise():
    with pytest.raises(ValueError):
        compute_reward(("g",), GOLD, _cost(1), lam=-0.1)
    with pytest.raises(ValueError):
        compute_reward(("g",), GOLD, _cost(1), correctness="bogus")
    with pytest.raises(ValueError):
        normalized_cost(_cost(1), cost_cap=0.0, axis="llm_calls")
