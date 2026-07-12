"""Per-question cost accounting and split-level aggregation.

``CostRecord`` is the frozen snapshot of what one traversal cost across every axis
the project measures. Aggregation helpers reduce a list of records to mean / median /
percentile summaries for a whole split — the raw material for the cost/quality
frontier (see ``kgat.eval.frontier``).

Pure stdlib (``statistics``) — no numpy/pandas here so the cost axis stays importable
in the zero-model-deps foundation.
"""

from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass

from kgat.traversal.budget import BudgetLedger

# The cost axes we track everywhere. Keep this list authoritative — metrics,
# frontier, and logging all iterate over it.
COST_FIELDS: tuple[str, ...] = ("hops", "llm_calls", "prompt_tokens", "gen_tokens", "wall_ms")


@dataclass(frozen=True)
class CostRecord:
    """Immutable per-question cost across all measured axes."""

    hops: int = 0
    llm_calls: int = 0
    prompt_tokens: int = 0
    gen_tokens: int = 0
    wall_ms: float = 0.0

    @classmethod
    def from_ledger(cls, ledger: BudgetLedger) -> CostRecord:
        """Freeze a live ``BudgetLedger`` into a ``CostRecord``."""
        return cls(
            hops=ledger.hops,
            llm_calls=ledger.llm_calls,
            prompt_tokens=ledger.prompt_tokens,
            gen_tokens=ledger.gen_tokens,
            wall_ms=ledger.wall_ms,
        )

    def as_dict(self) -> dict[str, float]:
        return dict(asdict(self))

    def scalar(self, axis: str) -> float:
        """Return a single cost axis by name (used by the reward's cost term)."""
        if axis not in COST_FIELDS:
            raise KeyError(f"unknown cost axis {axis!r}; expected one of {COST_FIELDS}")
        return float(getattr(self, axis))


def _percentile(values: list[float], q: float) -> float:
    """Linear-interpolated percentile, ``q`` in [0, 100]. Empty -> 0.0."""
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(values)
    rank = (q / 100.0) * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    frac = rank - low
    return float(ordered[low] * (1 - frac) + ordered[high] * frac)


def aggregate(records: list[CostRecord]) -> dict[str, dict[str, float]]:
    """Aggregate a split's cost records.

    Returns ``{axis: {"mean", "median", "p90", "p99", "max"}}`` for every cost axis.
    An empty input yields all-zero summaries so callers never special-case it.
    """
    summary: dict[str, dict[str, float]] = {}
    for axis in COST_FIELDS:
        values = [float(getattr(r, axis)) for r in records]
        if values:
            summary[axis] = {
                "mean": float(statistics.fmean(values)),
                "median": float(statistics.median(values)),
                "p90": _percentile(values, 90),
                "p99": _percentile(values, 99),
                "max": float(max(values)),
            }
        else:
            summary[axis] = {"mean": 0.0, "median": 0.0, "p90": 0.0, "p99": 0.0, "max": 0.0}
    return summary


def mean_cost(records: list[CostRecord], axis: str = "llm_calls") -> float:
    """Mean of a single cost axis across a split (the frontier's x-coordinate)."""
    if axis not in COST_FIELDS:
        raise KeyError(f"unknown cost axis {axis!r}; expected one of {COST_FIELDS}")
    values = [float(getattr(r, axis)) for r in records]
    return float(statistics.fmean(values)) if values else 0.0


__all__ = ["COST_FIELDS", "CostRecord", "aggregate", "mean_cost"]
