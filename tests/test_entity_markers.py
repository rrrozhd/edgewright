"""Entity-marker RE cue: filer-mention marking + prompt/encoding wiring (no torch).

The markers ([F]…[/F]) anchor relation DIRECTION — the largest genuine-error bucket
in the 2026-07 human audit. Marking is single-sourced in ``prompting`` so SFT,
cascade eval, and GRPO all mark identically (any drift silently degrades the model).
"""

from __future__ import annotations

from kgat.controller.constrained_decoding import build_triple_grammar
from kgat.controller.prompting import (
    FILER_CLOSE,
    FILER_OPEN,
    format_extraction_prompt,
    mark_filer_mentions,
)
from kgat.data.backfill_export import RELATIONSHIP_TYPES, ExtractionPair
from kgat.train.sft_extractor import encode_extraction_example

EOS = 3


class FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(c) % 256 for c in text]


TOK = FakeTokenizer()
GRAMMAR = build_triple_grammar(
    RELATIONSHIP_TYPES, ("Acme Corp", "Bolt Inc"), TOK, eos_id=EOS, max_triples=4
)


def _wrap(s: str) -> str:
    return f"{FILER_OPEN}{s}{FILER_CLOSE}"


def test_marks_first_person():
    out = mark_filer_mentions("We purchase components from Bolt Inc.", "Acme Corp")
    assert out == f"{_wrap('We')} purchase components from Bolt Inc."


def test_marks_filer_name_and_suffix_stripped_core():
    # Full name and the suffix-stripped core both count as the filer.
    out = mark_filer_mentions("Acme Corporation sells widgets; Acme also leases them.", "Acme Corporation")
    assert out == f"{_wrap('Acme Corporation')} sells widgets; {_wrap('Acme')} also leases them."


def test_marks_the_company_phrase():
    out = mark_filer_mentions("the Company supplies parts to X.", "Acme Corp")
    assert out.startswith(_wrap("the Company"))


def test_no_double_wrap_on_overlap():
    # "Acme" (name core) and no first-person overlap here, but ensure adjacent
    # matches don't produce nested markers.
    out = mark_filer_mentions("Acme Corp and we co-develop.", "Acme Corp")
    assert out == f"{_wrap('Acme Corp')} and {_wrap('we')} co-develop."
    assert FILER_OPEN + FILER_OPEN not in out


def test_returns_text_unchanged_when_no_mention():
    text = "Bolt Inc supplies parts to Zed LLC."
    assert mark_filer_mentions(text, "Acme Corp") == text


def test_country_US_not_marked_but_pronoun_us_is():
    # "US" the country must NOT be marked; lowercase "us" the pronoun must be.
    out = mark_filer_mentions("Our US operations supply parts to us weekly.", "Acme Corp")
    assert _wrap("US") not in out
    assert "US operations" in out  # country left intact
    assert out.endswith(f"to {_wrap('us')} weekly.")
    assert out.startswith(_wrap("Our"))


def test_common_word_core_not_marked_lowercase():
    # Filer "Gap Inc" → core "Gap". The capitalized company mention is marked; the
    # lowercase common word "gap" is not (case-sensitive core matching).
    out = mark_filer_mentions("Gap operates stores; the gap widened this year.", "Gap Inc")
    assert out.startswith(_wrap("Gap"))
    assert "the gap widened" in out
    assert _wrap("gap") not in out


def test_prompt_off_by_default_matches_unmarked():
    text = "We supply parts to Bolt Inc."
    base = format_extraction_prompt("Acme Corp", text, GRAMMAR.relations)
    assert FILER_OPEN not in base
    assert text in base


def test_prompt_on_marks_and_adds_hint():
    text = "We supply parts to Bolt Inc."
    marked = format_extraction_prompt("Acme Corp", text, GRAMMAR.relations, mark_filer=True)
    assert _wrap("We") in marked
    assert "marks the filer" in marked


def test_encoding_prompt_matches_marked_prompt():
    # The tokenized prompt inside encode_extraction_example must equal the marked
    # prompt — proves train-time encoding uses the same marker path as inference.
    pair = ExtractionPair(
        text="We supply parts to Bolt Inc.",
        filer="Acme Corp",
        triples=(("supplier", "Bolt Inc"),),
        filing="synthetic:Acme Corp",
    )
    enc = encode_extraction_example(pair, TOK, GRAMMAR, max_seq_len=4096, mark_filer=True)
    marked_prompt = format_extraction_prompt(
        pair.filer, pair.text, GRAMMAR.relations, mark_filer=True
    )
    prompt_ids = TOK.encode(marked_prompt)
    assert enc["input_ids"][: len(prompt_ids)] == prompt_ids
    assert enc["labels"][: len(prompt_ids)] == [-100] * len(prompt_ids)
