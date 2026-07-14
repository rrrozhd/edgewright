"""Tests for the extractor SFT encoding (pure parts, no torch)."""

from __future__ import annotations

import pytest

from kgat.controller.constrained_decoding import build_triple_grammar, encode_triples_target
from kgat.controller.prompting import format_extraction_prompt
from kgat.data.backfill_export import RELATIONSHIP_TYPES, ExtractionPair
from kgat.train.sft_extractor import encode_extraction_example, load_vocab

EOS = 3


class FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(c) for c in text]


TOK = FakeTokenizer()
GRAMMAR = build_triple_grammar(
    RELATIONSHIP_TYPES, ("Acme Corp", "Bolt Inc"), TOK, eos_id=EOS, max_triples=4
)


def make_pair(triples) -> ExtractionPair:
    return ExtractionPair(
        text="We purchase key components from Bolt Inc.",
        filer="Filer Co",
        triples=tuple(triples),
        filing="synthetic:Filer Co",
    )


def test_completion_only_labels_and_canonical_target():
    pair = make_pair([("customer", "Bolt Inc")])
    enc = encode_extraction_example(pair, TOK, GRAMMAR, max_seq_len=4096)
    target_ids = encode_triples_target(pair.triples, GRAMMAR)
    prompt_ids = TOK.encode(format_extraction_prompt(pair.filer, pair.text, GRAMMAR.relations))

    assert enc["input_ids"] == prompt_ids + target_ids
    assert enc["labels"][: len(prompt_ids)] == [-100] * len(prompt_ids)
    assert enc["labels"][len(prompt_ids) :] == target_ids  # grammar-canonical, ends with eos
    assert enc["labels"][-1] == EOS


def test_negative_pair_targets_none():
    enc = encode_extraction_example(make_pair([]), TOK, GRAMMAR, max_seq_len=4096)
    assert enc["labels"][-len(GRAMMAR.enc_none) :] == list(GRAMMAR.enc_none)


def test_left_truncation_preserves_target():
    pair = make_pair([("customer", "Bolt Inc")])
    target_ids = encode_triples_target(pair.triples, GRAMMAR)
    max_seq_len = len(target_ids) + 20
    enc = encode_extraction_example(pair, TOK, GRAMMAR, max_seq_len=max_seq_len)
    assert len(enc["input_ids"]) == max_seq_len
    assert enc["input_ids"][-len(target_ids) :] == target_ids
    # The surviving prompt tail is the "...extraction:" cue.
    tail = "".join(chr(i) for i in enc["input_ids"][: -len(target_ids)])
    assert tail.endswith("extraction:")

    with pytest.raises(ValueError):
        encode_extraction_example(pair, TOK, GRAMMAR, max_seq_len=len(target_ids))


def test_load_vocab_missing_is_actionable(tmp_path):
    with pytest.raises(FileNotFoundError, match="backfill_export"):
        load_vocab(tmp_path)
