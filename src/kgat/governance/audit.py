"""Audit certificate assembly (IMPLEMENTED datamodel; policy checks live in policy.py).

Collects per-hop ``HopAudit`` records over a traversal and assembles the final
``AuditCertificate``. The certificate is emitted for *every* run — even when
governance is disabled (no policies), it records the hops with empty check sets and a
``final_verdict`` of ``True`` — so auditability is uniform, not conditional.
"""

from __future__ import annotations

from kgat.data.schemas import Action, ActionType, AuditCertificate, HopAudit
from kgat.eval.cost import CostRecord
from kgat.governance.policy import HopPolicy


def run_hop_policies(
    policies: list[HopPolicy],
    state: object,  # kgat.data.schemas.TraversalState — untyped to avoid an import cycle
    action: Action,
) -> tuple[dict[str, bool], bool]:
    """Run every policy against ``(state, action)``.

    Returns ``(checks_passed, hard_block)`` where ``checks_passed`` maps policy name
    -> pass/fail and ``hard_block`` is True iff a *mandatory* policy failed.
    """
    checks_passed: dict[str, bool] = {}
    hard_block = False
    for policy in policies:
        passed, _detail = policy.check(state, action)  # type: ignore[arg-type]
        checks_passed[policy.name] = passed
        if not passed and policy.mandatory:
            hard_block = True
    return checks_passed, hard_block


class AuditCertificateBuilder:
    """Accumulates ``HopAudit`` records and assembles an ``AuditCertificate``."""

    def __init__(self, qid: str) -> None:
        self.qid = qid
        self._hops: list[HopAudit] = []

    def record_hop(
        self,
        *,
        step: int,
        action: Action,
        checks_passed: dict[str, bool],
        confidence: float,
        provenance: tuple[str, ...] = (),
    ) -> HopAudit:
        """Record one hop's governance outcome and return the ``HopAudit``.

        For a ``STOP`` action the recorded relation is the empty string (there is no
        edge), which keeps the certificate uniform across expand/stop hops.
        """
        relation = action.relation if action.type is ActionType.EXPAND else ""
        hop = HopAudit(
            step=step,
            relation=relation or "",
            checks_passed=dict(checks_passed),
            confidence=confidence,
            provenance=provenance,
        )
        self._hops.append(hop)
        return hop

    @property
    def hops(self) -> list[HopAudit]:
        return list(self._hops)

    def build(self, cost: CostRecord) -> AuditCertificate:
        """Assemble the certificate. ``final_verdict`` is True iff every recorded
        check on every hop passed (vacuously True when no policies ran)."""
        final_verdict = all(all(hop.checks_passed.values()) for hop in self._hops)
        return AuditCertificate(
            qid=self.qid,
            hops=self.hops,
            final_verdict=final_verdict,
            cost=cost,
        )


__all__ = ["AuditCertificateBuilder", "run_hop_policies"]
