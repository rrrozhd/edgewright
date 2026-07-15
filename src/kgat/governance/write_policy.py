"""Write-path governance — per-edge policies, audits, and the governed commit seam.

Mirror of the read path (``policy.py`` / ``audit.py``) with the unit of
governance changed from hop to EDGE: before a proposed edge reaches the
``KGWriteStore``, every registered ``EdgePolicy`` inspects it and a failed
mandatory policy hard-blocks the commit (fail-closed — DESIGN-KG-FILLING.md).
Every proposal, committed or blocked, lands in the filing's
``WriteCertificate`` so each backfilled filing has an auditable receipt: what
was read, what was claimed, what was committed, what it cost.

Single-sourcing invariant: ``EvidenceGatesPolicy`` runs the SAME
``kgat.train.edge_judge`` gates the RL reward uses. Training-time gate failure
zeroes reward (soft); commit-time failure blocks the write (hard) — the policy
cannot learn to satisfy a reward that governance then rejects, or vice versa.

``SchemaAllowlistPolicy`` is deliberately redundant for grammar-decoded edges
(the trie makes off-schema emission physically impossible) — it exists as
defense in depth for edge sources that do NOT decode through the grammar (the
phase-2 GNN completion proposer, escalated teacher output).
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from kgat.data.backfill_export import RELATIONSHIP_TYPES, ExtractionPair
from kgat.data.schemas import Triple
from kgat.eval.cost import CostRecord
from kgat.graph.write_store import EdgeProvenance, KGWriteStore
from kgat.train.backfill_routing import ROUTE_ESCALATE, ROUTE_EXTRACT, ChunkDecision
from kgat.train.edge_judge import RuleGates, ground_span


@dataclass(frozen=True)
class EdgeProposal:
    """One candidate edge headed for the graph, with everything policies need."""

    filer: str
    relation: str
    target: str
    chunk: str  # source chunk id
    text: str  # the chunk text (evidence universe for the gates)
    route: str  # ROUTE_EXTRACT | ROUTE_ESCALATE
    confidence: float
    judge_score: float | None = None  # distilled-judge faithfulness, when scored
    extractor: str = ""  # adapter/model identity


class EdgePolicy(ABC):
    """A single per-edge governance rule (mirror of ``HopPolicy``)."""

    name: str = "edge_policy"
    mandatory: bool = True

    @abstractmethod
    def check(self, proposal: EdgeProposal) -> tuple[bool, dict]:
        """Return ``(passed, detail)`` — detail is folded into the ``EdgeAudit``."""
        raise NotImplementedError


class SchemaAllowlistPolicy(EdgePolicy):
    """Only taxonomy relations may be written (exact membership, no patterns)."""

    name = "schema_allowlist"

    def __init__(self, relations: Sequence[str] = RELATIONSHIP_TYPES) -> None:
        self.relations = frozenset(relations)

    def check(self, proposal: EdgeProposal) -> tuple[bool, dict]:
        passed = proposal.relation in self.relations
        return passed, {"relation": proposal.relation}


class EvidenceGatesPolicy(EdgePolicy):
    """Fail-closed existence gates — the SAME gates the RL reward runs.

    Grounding additionally locates the span; its offsets go into the audit (and
    the provenance record) so every committed edge pins to chunk + chars, the
    edge-level extension of alphina's ``evidence_start/end``.
    """

    name = "evidence_gates"

    def __init__(self, gates: RuleGates | None = None) -> None:
        self.gates = gates or RuleGates()

    def check(self, proposal: EdgeProposal) -> tuple[bool, dict]:
        pair = ExtractionPair(
            text=proposal.text, filer=proposal.filer, triples=(), filing="", chunk=proposal.chunk
        )
        report = self.gates.evaluate(pair, proposal.target)
        span = ground_span(proposal.target, proposal.text, self.gates.aliases)
        return report.passed, {
            "grounded": report.grounded,
            "is_company": report.is_company,
            "filer_party": report.filer_party,
            "span": list(span) if span else None,
        }


class EdgeConfidenceFloorPolicy(EdgePolicy):
    """Commit floor on the judge score (falling back to decode confidence)."""

    name = "edge_confidence_floor"

    def __init__(self, floor: float = 0.0) -> None:
        self.floor = float(floor)

    def check(self, proposal: EdgeProposal) -> tuple[bool, dict]:
        score = proposal.judge_score if proposal.judge_score is not None else proposal.confidence
        return score >= self.floor, {"score": score, "floor": self.floor}


def run_edge_policies(
    policies: Sequence[EdgePolicy], proposal: EdgeProposal
) -> tuple[dict[str, bool], bool]:
    """Run every policy; returns ``(checks_passed, hard_block)`` like the read path."""
    checks_passed: dict[str, bool] = {}
    hard_block = False
    for policy in policies:
        passed, _detail = policy.check(proposal)
        checks_passed[policy.name] = passed
        if not passed and policy.mandatory:
            hard_block = True
    return checks_passed, hard_block


@dataclass(frozen=True)
class EdgeAudit:
    """Per-proposal governance record — ``HopAudit`` with edge semantics."""

    chunk: str
    relation: str
    target: str
    route: str
    checks_passed: dict[str, bool]
    confidence: float
    judge_score: float | None
    span: tuple[int, int] | None
    committed: bool
    extractor: str = ""


@dataclass(frozen=True)
class WriteCertificate:
    """The auditable receipt for one backfilled filing."""

    filing: str
    edges: tuple[EdgeAudit, ...]  # every proposal, committed or blocked
    n_committed: int
    n_blocked: int
    final_verdict: bool  # every COMMITTED edge passed every recorded check
    cost: CostRecord


class WriteCertificateBuilder:
    """Accumulates ``EdgeAudit`` records and assembles a ``WriteCertificate``."""

    def __init__(self, filing: str) -> None:
        self.filing = filing
        self._edges: list[EdgeAudit] = []

    def record_edge(
        self,
        proposal: EdgeProposal,
        checks_passed: dict[str, bool],
        *,
        committed: bool,
        span: tuple[int, int] | None,
    ) -> EdgeAudit:
        audit = EdgeAudit(
            chunk=proposal.chunk,
            relation=proposal.relation,
            target=proposal.target,
            route=proposal.route,
            checks_passed=dict(checks_passed),
            confidence=proposal.confidence,
            judge_score=proposal.judge_score,
            span=span,
            committed=committed,
            extractor=proposal.extractor,
        )
        self._edges.append(audit)
        return audit

    def build(self, cost: CostRecord) -> WriteCertificate:
        committed = [e for e in self._edges if e.committed]
        verdict = all(all(e.checks_passed.values()) for e in committed)
        return WriteCertificate(
            filing=self.filing,
            edges=tuple(self._edges),
            n_committed=len(committed),
            n_blocked=len(self._edges) - len(committed),
            final_verdict=verdict,
            cost=cost,
        )


def proposals_from_decision(
    pair: ExtractionPair, decision: ChunkDecision, *, extractor: str = ""
) -> list[EdgeProposal]:
    """Bridge one routing decision into edge proposals.

    Extract routes propose the policy's own triples at decode confidence;
    escalated chunks propose the TEACHER's edges (its labels) at the teacher's
    confidence — both flow through the same policy chain, so escalation is not a
    governance bypass. Skips propose nothing.
    """
    if decision.route == ROUTE_EXTRACT:
        decode_conf = math.exp(decision.logprob / max(1, decision.n_choices))
        triples, route, confidence = decision.triples, ROUTE_EXTRACT, decode_conf
    elif decision.route == ROUTE_ESCALATE:
        triples, route, confidence = pair.triples, ROUTE_ESCALATE, pair.confidence or 1.0
    else:
        return []
    chunk = pair.chunk or f"{pair.filing}:chunk"
    return [
        EdgeProposal(
            filer=pair.filer,
            relation=relation,
            target=target,
            chunk=chunk,
            text=pair.text,
            route=route,
            confidence=confidence,
            extractor=extractor,
        )
        for relation, target in triples
    ]


def governed_commit(
    proposals: Iterable[EdgeProposal],
    policies: Sequence[EdgePolicy],
    store: KGWriteStore,
    builder: WriteCertificateBuilder,
) -> list[EdgeProposal]:
    """Run every proposal through the policy chain; commit survivors, audit all.

    Fail-closed: a failed mandatory policy blocks the write. Returns the
    committed proposals. The certificate (``builder.build(cost)``) is the
    caller's to assemble once the filing's cost record is final.
    """
    committed: list[EdgeProposal] = []
    for proposal in proposals:
        checks_passed, hard_block = run_edge_policies(policies, proposal)
        span = ground_span(proposal.target, proposal.text)
        if not hard_block:
            store.add_triple(
                Triple(proposal.filer, proposal.relation, proposal.target),
                provenance=EdgeProvenance(
                    chunk=proposal.chunk,
                    span=span,
                    route=proposal.route,
                    confidence=proposal.confidence,
                    extractor=proposal.extractor,
                ),
            )
            committed.append(proposal)
        builder.record_edge(proposal, checks_passed, committed=not hard_block, span=span)
    return committed


__all__ = [
    "EdgeProposal",
    "EdgePolicy",
    "SchemaAllowlistPolicy",
    "EvidenceGatesPolicy",
    "EdgeConfidenceFloorPolicy",
    "run_edge_policies",
    "EdgeAudit",
    "WriteCertificate",
    "WriteCertificateBuilder",
    "proposals_from_decision",
    "governed_commit",
]
