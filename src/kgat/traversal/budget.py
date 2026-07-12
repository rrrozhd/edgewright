"""Budget accounting for a single traversal.

The ``BudgetLedger`` tracks the cost dimensions the project cares about — hops, LLM
calls, prompt/generation tokens, wall-clock — and enforces configurable caps. The
engine charges the ledger as it expands the frontier and stops when any cap is hit.

This module has no model dependencies and is pure/testable.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BudgetCaps:
    """Upper bounds on each cost dimension. ``None`` means "no cap on this axis"."""

    max_hops: int | None = None
    max_llm_calls: int | None = None
    max_prompt_tokens: int | None = None
    max_gen_tokens: int | None = None
    max_wall_ms: float | None = None

    @classmethod
    def from_config(cls, cfg: object) -> BudgetCaps:
        """Build caps from a mapping-like or attribute-like Hydra config node."""

        def _get(key: str) -> object | None:
            if cfg is None:
                return None
            if isinstance(cfg, dict):
                return cfg.get(key)
            return getattr(cfg, key, None)

        return cls(
            max_hops=_get("max_hops"),
            max_llm_calls=_get("max_llm_calls"),
            max_prompt_tokens=_get("max_prompt_tokens"),
            max_gen_tokens=_get("max_gen_tokens"),
            max_wall_ms=_get("max_wall_ms"),
        )


@dataclass
class BudgetLedger:
    """Running tally of spend against ``caps``. Mutated in place by ``charge``."""

    caps: BudgetCaps = BudgetCaps()
    hops: int = 0
    llm_calls: int = 0
    prompt_tokens: int = 0
    gen_tokens: int = 0
    wall_ms: float = 0.0

    def charge(
        self,
        *,
        hops: int = 0,
        llm_calls: int = 0,
        prompt_tokens: int = 0,
        gen_tokens: int = 0,
        wall_ms: float = 0.0,
    ) -> None:
        """Add spend to the ledger. All arguments default to zero."""
        self.hops += hops
        self.llm_calls += llm_calls
        self.prompt_tokens += prompt_tokens
        self.gen_tokens += gen_tokens
        self.wall_ms += wall_ms

    def exhausted(self) -> bool:
        """True if any capped dimension has reached or exceeded its cap.

        Caps are inclusive: a ``max_hops=3`` budget is exhausted once ``hops == 3``,
        so the engine will not start a 4th hop.
        """
        c = self.caps
        checks = (
            (c.max_hops, self.hops),
            (c.max_llm_calls, self.llm_calls),
            (c.max_prompt_tokens, self.prompt_tokens),
            (c.max_gen_tokens, self.gen_tokens),
            (c.max_wall_ms, self.wall_ms),
        )
        return any(cap is not None and spent >= cap for cap, spent in checks)

    def snapshot(self) -> dict[str, float]:
        """Plain dict of the current spend (handy for logging)."""
        return {
            "hops": self.hops,
            "llm_calls": self.llm_calls,
            "prompt_tokens": self.prompt_tokens,
            "gen_tokens": self.gen_tokens,
            "wall_ms": self.wall_ms,
        }


__all__ = ["BudgetCaps", "BudgetLedger"]
