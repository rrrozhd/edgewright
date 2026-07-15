"""Tests for write-path governance: gates-with-offsets, policies, governed commit."""

from __future__ import annotations

import math

from kgat.data.backfill_export import ExtractionPair
from kgat.data.schemas import Triple
from kgat.eval.cost import CostRecord
from kgat.governance.write_policy import (
    EdgeConfidenceFloorPolicy,
    EdgeProposal,
    EvidenceGatesPolicy,
    SchemaAllowlistPolicy,
    WriteCertificateBuilder,
    governed_commit,
    proposals_from_decision,
    run_edge_policies,
)
from kgat.graph.write_store import InMemoryKGWriteStore
from kgat.train.backfill_routing import ROUTE_ESCALATE, ROUTE_EXTRACT, ROUTE_SKIP, ChunkDecision
from kgat.train.edge_judge import RuleGates, ground_span, normalize_people

CHUNK = (
    "We rely on Taiwan Semiconductor Manufacturing Company for wafer fabrication. "
    "We compete directly with Intel Corporation and NVIDIA."
)


def proposal(
    relation="competitor", target="Intel Corporation", text=CHUNK, chunk="ch-1", **kw
) -> EdgeProposal:
    return EdgeProposal(
        filer="Advanced Micro Devices, Inc.",
        relation=relation,
        target=target,
        chunk=chunk,
        text=text,
        route=ROUTE_EXTRACT,
        confidence=0.9,
        **kw,
    )


def test_ground_span_returns_original_coordinates():
    span = ground_span("Intel Corporation", CHUNK)
    assert span is not None
    assert CHUNK[span[0] : span[1]] == "Intel Corporation"
    # Suffix drift: the located span is the shorter surface form actually present.
    span = ground_span("NVIDIA Corp", CHUNK)
    assert CHUNK[span[0] : span[1]] == "NVIDIA"
    # Case-insensitive match still yields exact original offsets.
    span = ground_span("intel", CHUNK)
    assert CHUNK[span[0] : span[1]] == "Intel"
    assert ground_span("Broadcom", CHUNK) is None


def test_policy_matrix():
    good = proposal()
    assert run_edge_policies(
        [SchemaAllowlistPolicy(), EvidenceGatesPolicy(), EdgeConfidenceFloorPolicy(0.5)], good
    ) == ({"schema_allowlist": True, "evidence_gates": True, "edge_confidence_floor": True}, False)

    # Off-taxonomy relation (e.g. legacy 'holds') hard-blocks despite the grammar
    # never emitting it — defense in depth for non-grammar edge sources.
    checks, block = run_edge_policies([SchemaAllowlistPolicy()], proposal(relation="holds"))
    assert block and checks["schema_allowlist"] is False

    # Ungrounded target fails the gates, fail-closed.
    checks, block = run_edge_policies([EvidenceGatesPolicy()], proposal(target="Broadcom Inc"))
    assert block and checks["evidence_gates"] is False

    # Person target fails via the known-people set (single-sourced with the reward).
    gates = RuleGates(known_people=normalize_people(["Jensen Huang"]))
    bio = proposal(target="Jensen Huang", text="We rely on Jensen Huang. Our board...")
    checks, block = run_edge_policies([EvidenceGatesPolicy(gates)], bio)
    assert block

    # Judge score drives the floor when present; decode confidence otherwise.
    floor = EdgeConfidenceFloorPolicy(0.5)
    assert floor.check(proposal(judge_score=0.4))[0] is False
    assert floor.check(proposal(judge_score=None))[0] is True  # confidence=0.9


def test_governed_commit_blocks_and_audits():
    pairs_policies = [SchemaAllowlistPolicy(), EvidenceGatesPolicy()]
    store = InMemoryKGWriteStore()
    builder = WriteCertificateBuilder(filing="acc-1")
    proposals = [
        proposal(),  # clean -> committed
        proposal(target="Broadcom Inc"),  # ungrounded -> blocked
    ]
    committed = governed_commit(proposals, pairs_policies, store, builder)
    assert len(committed) == 1 and len(store) == 1

    edge = Triple("Advanced Micro Devices, Inc.", "competitor", "Intel Corporation")
    assert store.has_triple(edge)
    prov = store.provenance_of(edge)[0]
    assert prov.chunk == "ch-1" and prov.route == ROUTE_EXTRACT
    assert CHUNK[prov.span[0] : prov.span[1]] == "Intel Corporation"

    cert = builder.build(CostRecord(llm_calls=0))
    assert cert.n_committed == 1 and cert.n_blocked == 1
    assert cert.final_verdict is True  # every COMMITTED edge passed; blocks don't taint
    blocked = next(e for e in cert.edges if not e.committed)
    assert blocked.checks_passed["evidence_gates"] is False and blocked.span is None


def test_repeat_commit_accumulates_corroborating_provenance():
    store = InMemoryKGWriteStore()
    builder = WriteCertificateBuilder(filing="acc-1")
    governed_commit([proposal(), proposal(chunk="ch-2")], [SchemaAllowlistPolicy()], store, builder)
    edge = Triple("Advanced Micro Devices, Inc.", "competitor", "Intel Corporation")
    assert len(store) == 1  # one edge...
    assert len(store.provenance_of(edge)) == 2  # ...two evidence records


def test_proposals_from_decision_bridges_all_routes():
    pair = ExtractionPair(
        text=CHUNK,
        filer="Advanced Micro Devices, Inc.",
        triples=(("customer", "Taiwan Semiconductor Manufacturing Company"),),
        filing="acc-1",
        confidence=0.88,
        chunk="ch-9",
    )
    extract = ChunkDecision(
        route=ROUTE_EXTRACT,
        triples=(("competitor", "Intel Corporation"),),
        gen_tokens=12,
        logprob=-0.5,
        n_choices=5,
    )
    props = proposals_from_decision(pair, extract, extractor="r4")
    assert len(props) == 1
    assert props[0].relation == "competitor" and props[0].chunk == "ch-9"
    assert props[0].extractor == "r4"
    assert math.isclose(props[0].confidence, math.exp(-0.5 / 5))

    # Escalation proposes the TEACHER's edges through the same policy chain.
    escalate = ChunkDecision(
        route=ROUTE_ESCALATE, triples=(), gen_tokens=5, logprob=-0.1, n_choices=1
    )
    props = proposals_from_decision(pair, escalate)
    assert len(props) == 1
    assert props[0].route == ROUTE_ESCALATE and props[0].confidence == 0.88
    assert props[0].relation == "customer"

    skip = ChunkDecision(route=ROUTE_SKIP, triples=(), gen_tokens=5, logprob=-0.1, n_choices=1)
    assert proposals_from_decision(pair, skip) == []
