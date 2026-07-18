"""ESCALATE warmup: target encoding + hard-chunk selection (pure parts, no torch).

The warmup exists because a plain SFT extractor puts P(ESCALATE) ~6e-9 — RL cannot
sample what the policy never emits, so the action is taught here.
"""

from __future__ import annotations

from kgat.controller.constrained_decoding import build_triple_grammar, encode_triples_target
from kgat.data.backfill_export import RELATIONSHIP_TYPES, ExtractionPair
from kgat.train.backfill_routing import ESCALATE_LABEL
from kgat.train.routing_warmup import (
    encode_warmup_example,
    is_hard_chunk,
    select_escalate_ids,
)

EOS = 3


class FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(c) % 256 for c in text]


TOK = FakeTokenizer()
GRAMMAR = build_triple_grammar(
    RELATIONSHIP_TYPES, ("Acme Corp", "Bolt Inc"), TOK,
    eos_id=EOS, max_triples=4, sentinels=(ESCALATE_LABEL,),
)


def pair(text="We supply parts to Bolt Inc.", triples=(("supplier", "Bolt Inc"),), filer="Acme Corp"):
    return ExtractionPair(text=text, filer=filer, triples=tuple(triples), filing="f:1")


def test_grammar_exposes_escalate_target():
    assert ESCALATE_LABEL in GRAMMAR.enc_sentinel
    assert GRAMMAR.enc_sentinel[ESCALATE_LABEL][-1] == EOS  # complete, terminated


def test_escalate_example_targets_the_sentinel():
    enc = encode_warmup_example(pair(), TOK, GRAMMAR, max_seq_len=4096, escalate=True)
    target = list(GRAMMAR.enc_sentinel[ESCALATE_LABEL])
    assert enc["labels"][-len(target):] == target
    assert enc["labels"][-1] == EOS
    # prompt is masked
    assert enc["labels"][0] == -100


def test_non_escalate_example_targets_normal_triples():
    p = pair()
    enc = encode_warmup_example(p, TOK, GRAMMAR, max_seq_len=4096, escalate=False)
    target = encode_triples_target(p.triples, GRAMMAR)
    assert enc["labels"][-len(target):] == target
    # and it is NOT the escalate sentinel
    esc = list(GRAMMAR.enc_sentinel[ESCALATE_LABEL])
    assert enc["labels"][-len(esc):] != esc


def test_hard_chunk_many_edges():
    many = [("supplier", "Bolt Inc")] * 4
    assert is_hard_chunk(pair(triples=many), min_edges=4) is True


def test_hard_chunk_non_verbatim_target():
    # target absent from the chunk text -> model must recall from global vocab
    p = pair(text="We supply parts to someone.", triples=(("supplier", "Bolt Inc"),))
    assert is_hard_chunk(p) is True


def test_easy_chunk_is_not_hard():
    p = pair(text="We supply parts to Bolt Inc.", triples=(("supplier", "Bolt Inc"),))
    assert is_hard_chunk(p) is False


def test_empty_gold_never_hard():
    assert is_hard_chunk(pair(triples=())) is False


def test_select_escalate_ids_includes_hard_and_is_deterministic():
    pairs = [
        pair(text="We supply parts to Bolt Inc."),                      # easy
        pair(text="We supply parts to someone.",                        # hard (non-verbatim)
             triples=(("supplier", "Bolt Inc"),)),
    ]
    a = select_escalate_ids(pairs, extra_random=0.0, seed=1)
    b = select_escalate_ids(pairs, extra_random=0.0, seed=1)
    assert a == b            # deterministic
    assert 1 in a            # the hard one is selected
    assert 0 not in a        # the easy one is not (extra_random=0)
