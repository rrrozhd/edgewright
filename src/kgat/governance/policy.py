"""``HopPolicy`` — per-hop governance checks.

Governance wraps traversal: before an expansion is committed, every registered
``HopPolicy`` inspects the state + proposed action and returns pass/fail with
detail. The engine records each result into a ``HopAudit`` and hard-blocks on a
failed *mandatory* policy. This makes governed, auditable traversal a first-class
property rather than a post-hoc log.

A ``STOP`` action always passes every policy here: stopping is the safe action, and
a policy that could veto STOP would force the controller to keep expanding — the
opposite of what governance is for.
"""

from __future__ import annotations

import fnmatch
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence

from kgat.data.schemas import Action, ActionType, TraversalState


class HopPolicy(ABC):
    """A single per-hop governance rule."""

    name: str = "policy"
    mandatory: bool = True  # a failed mandatory policy hard-blocks the hop

    @abstractmethod
    def check(self, state: TraversalState, action: Action) -> tuple[bool, dict]:
        """Return ``(passed, detail)`` for ``action`` at ``state``.

        ``detail`` is a small JSON-serializable dict of evidence (thresholds,
        matched rule ids, ...) folded into the ``HopAudit``.
        """
        raise NotImplementedError


class AllowAllPolicy(HopPolicy):
    """Trivial policy that passes everything — audit-path placeholder/baseline."""

    name = "allow_all"

    def check(self, state: TraversalState, action: Action) -> tuple[bool, dict]:
        return True, {"reason": "allow_all"}


class RelationAllowlistPolicy(HopPolicy):
    """Only permit expansions along approved relations (exact or fnmatch patterns).

    ``patterns`` like ``["people.person.*", "location.capital"]``. The default
    ``["*"]`` passes everything (useful as a wired-but-open baseline).
    """

    name = "relation_allowlist"

    def __init__(self, patterns: Sequence[str] = ("*",)) -> None:
        self.patterns = tuple(patterns)

    def check(self, state: TraversalState, action: Action) -> tuple[bool, dict]:
        if action.type is ActionType.STOP:
            return True, {"reason": "stop_is_always_allowed"}
        relation = action.relation or ""
        matched = next((p for p in self.patterns if fnmatch.fnmatch(relation, p)), None)
        return matched is not None, {"relation": relation, "matched_pattern": matched}


class ConfidenceFloorPolicy(HopPolicy):
    """Block expansions whose controller confidence is below ``floor``.

    ``Action.score`` semantics are controller-specific (the decoder policy reports
    ``exp(mean token logprob)`` in [0, 1]; the dummy reports relation degree), so
    set the floor per-controller in config.
    """

    name = "confidence_floor"

    def __init__(self, floor: float = 0.0) -> None:
        self.floor = float(floor)

    def check(self, state: TraversalState, action: Action) -> tuple[bool, dict]:
        if action.type is ActionType.STOP:
            return True, {"reason": "stop_is_always_allowed"}
        passed = action.score >= self.floor
        return passed, {"score": action.score, "floor": self.floor}


class ProvenanceRequiredPolicy(HopPolicy):
    """Require every expanded edge to carry provenance source ids.

    ``lookup(head_node, relation) -> tuple[source_id, ...]`` supplies provenance for
    an edge. Preprocessed WebQSP/CWQ subgraphs ship no provenance, so without a
    lookup this policy (correctly) fails every expansion — wire it only on KGs that
    carry sources (the M9 transfer KG). The check queries every frontier node the
    expansion would depart from and requires at least one sourced edge.
    """

    name = "provenance_required"

    def __init__(self, lookup: Callable[[str, str], tuple[str, ...]] | None = None) -> None:
        self.lookup = lookup

    def check(self, state: TraversalState, action: Action) -> tuple[bool, dict]:
        if action.type is ActionType.STOP:
            return True, {"reason": "stop_is_always_allowed"}
        if self.lookup is None:
            return False, {"reason": "no_provenance_source_configured"}
        relation = action.relation or ""
        sources: list[str] = []
        for node in state.frontier_nodes:
            sources.extend(self.lookup(node, relation))
        return bool(sources), {"relation": relation, "n_sources": len(sources)}


# -- config wiring -------------------------------------------------------------

_REGISTRY = {
    "allow_all": lambda cfg: AllowAllPolicy(),
    "relation_allowlist": lambda cfg: RelationAllowlistPolicy(
        tuple(cfg.get("allowlist") or ("*",))
    ),
    "confidence_floor": lambda cfg: ConfidenceFloorPolicy(float(cfg.get("confidence_floor", 0.0))),
    "provenance_required": lambda cfg: ProvenanceRequiredPolicy(),
}


def build_policies(gov_cfg) -> list[HopPolicy]:
    """Instantiate the policies named in a ``governance`` config node.

    ``gov_cfg`` is mapping-like: ``{"policies": [...names...], "allowlist": [...],
    "confidence_floor": 0.2}``. Unknown names raise (a silently-skipped policy would
    be a governance hole). An enabled-but-empty list gets ``AllowAllPolicy`` so the
    audit path is still exercised.
    """
    names = list(gov_cfg.get("policies") or [])
    if not names:
        return [AllowAllPolicy()]
    policies: list[HopPolicy] = []
    for name in names:
        if name not in _REGISTRY:
            raise KeyError(f"unknown governance policy {name!r}; known: {sorted(_REGISTRY)}")
        policies.append(_REGISTRY[name](gov_cfg))
    return policies


__all__ = [
    "HopPolicy",
    "AllowAllPolicy",
    "RelationAllowlistPolicy",
    "ConfidenceFloorPolicy",
    "ProvenanceRequiredPolicy",
    "build_policies",
]
