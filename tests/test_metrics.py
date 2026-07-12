"""Tests for KGQA metrics (kgat.eval.metrics) against a hand-built oracle."""

from __future__ import annotations

import math

from kgat.eval.metrics import f1, hit, hits_at_1, precision, recall


def test_hit():
    assert hit(["b1", "b2"], ["b1"]) is True
    assert hit(["x"], ["b1"]) is False
    assert hit([], ["b1"]) is False
    assert hit(["x"], []) is False  # no gold -> nothing to hit


def test_hits_at_1_uses_top_prediction():
    assert hits_at_1(["b1", "b2"], ["b1"]) == 1.0
    assert hits_at_1(["b2", "b1"], ["b1"]) == 0.0  # top-1 is b2, not gold
    assert hits_at_1(["b1"], ["b1", "b2"]) == 1.0
    assert hits_at_1([], ["b1"]) == 0.0
    assert hits_at_1(["b1"], []) == 0.0


def test_precision_recall():
    # preds {b1, b2}, gold {b1}
    assert precision(["b1", "b2"], ["b1"]) == 0.5
    assert recall(["b1", "b2"], ["b1"]) == 1.0
    # dedup: repeated predictions don't inflate the denominator
    assert precision(["b1", "b1", "x"], ["b1"]) == 0.5


def test_f1_partial_credit():
    assert math.isclose(f1(["b1", "b2"], ["b1"]), 2 / 3)  # p=.5, r=1 -> 0.6667


def test_f1_perfect_and_edge_cases():
    assert f1(["b1"], ["b1"]) == 1.0
    assert f1([], ["b1"]) == 0.0  # gold present, nothing predicted
    assert f1([], []) == 1.0  # correctly predicts "no answer"
    assert f1(["x"], []) == 0.0  # gold empty but over-predicted
    assert f1(["x"], ["b1"]) == 0.0  # disjoint
