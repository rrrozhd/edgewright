"""KGQA metrics: Hit, Hits@1, and F1.

Definitions (per the brief; note these are conflated across papers, so M2 must
compare each baseline against the *matching* definition):

* **Hit**    — 1 if ANY gold answer appears anywhere in the predictions, else 0.
* **Hits@1** — 1 if the TOP-1 (first) prediction is a gold answer, else 0. Requires
               predictions to be ordered by confidence.
* **F1**     — harmonic mean of precision (|pred ∩ gold| / |pred|) and recall
               (|pred ∩ gold| / |gold|), balancing coverage of *all* golds against
               over-prediction. Predictions and golds are treated as sets.

Pure functions over sequences of entity strings — no model deps.
"""

from __future__ import annotations

from collections.abc import Sequence

from kgat.data.schemas import Entity


def _dedup(seq: Sequence[Entity]) -> list[Entity]:
    seen: dict[Entity, None] = {}
    for x in seq:
        seen.setdefault(x, None)
    return list(seen)


def hit(predictions: Sequence[Entity], gold: Sequence[Entity]) -> bool:
    """True if any gold answer is among the predictions."""
    gold_set = set(gold)
    return any(p in gold_set for p in predictions)


def hits_at_1(predictions: Sequence[Entity], gold: Sequence[Entity]) -> float:
    """1.0 if the top-ranked prediction is a gold answer, else 0.0.

    ``predictions`` must be ordered by confidence (best first). Empty predictions or
    empty gold yield 0.0.
    """
    if not predictions or not gold:
        return 0.0
    return 1.0 if predictions[0] in set(gold) else 0.0


def precision(predictions: Sequence[Entity], gold: Sequence[Entity]) -> float:
    preds = _dedup(predictions)
    if not preds:
        return 0.0
    gold_set = set(gold)
    correct = sum(1 for p in preds if p in gold_set)
    return correct / len(preds)


def recall(predictions: Sequence[Entity], gold: Sequence[Entity]) -> float:
    golds = _dedup(gold)
    if not golds:
        return 0.0
    pred_set = set(predictions)
    covered = sum(1 for g in golds if g in pred_set)
    return covered / len(golds)


def f1(predictions: Sequence[Entity], gold: Sequence[Entity]) -> float:
    """Set-based F1 of predictions against gold.

    Edge cases: if there are no gold answers, F1 is 1.0 when predictions are also
    empty (correctly predicting "no answer") and 0.0 otherwise. If gold is non-empty
    but predictions are empty, F1 is 0.0.
    """
    if not gold:
        return 1.0 if len(_dedup(predictions)) == 0 else 0.0
    p = precision(predictions, gold)
    r = recall(predictions, gold)
    if p + r == 0.0:
        return 0.0
    return 2 * p * r / (p + r)


__all__ = ["hit", "hits_at_1", "precision", "recall", "f1"]
