"""Tests for chunk-local target candidates (pure string processing)."""

from __future__ import annotations

from kgat.controller.constrained_decoding import build_triple_grammar, decode_triples
from kgat.data.backfill_export import RELATIONSHIP_TYPES
from kgat.data.chunk_targets import chunk_target_candidates, match_candidate, normalize_name

CHUNK = (
    "We rely on Taiwan Semiconductor Manufacturing Company for substantially all of "
    "our wafer fabrication. We compete directly with Intel Corporation, NVIDIA and "
    "S&P Global. John A. Smith has served as our Chief Executive Officer since 2019. "
    "The Company operates under United States regulations. See Item 1A."
)


def test_candidates_find_entities_and_drop_furniture():
    cands = chunk_target_candidates(
        CHUNK,
        filer="Advanced Micro Devices, Inc.",
        known_people=["John A. Smith"],
    )
    assert "Taiwan Semiconductor Manufacturing Company" in cands
    assert "Intel Corporation" in cands
    assert "NVIDIA" in cands  # all-caps single token survives
    assert "S&P Global" in cands
    assert "John A. Smith" not in cands  # known person filtered
    for junk in ("We", "The", "See", "The Company"):
        assert junk not in cands
    # Order = first appearance; dedupe by surface form.
    assert cands.index("Taiwan Semiconductor Manufacturing Company") < cands.index(
        "Intel Corporation"
    )
    assert len(cands) == len(set(cands))


def test_filer_and_its_prefix_are_excluded():
    text = "Advanced Micro Devices competes with Intel Corporation."
    cands = chunk_target_candidates(text, filer="Advanced Micro Devices, Inc.")
    assert "Intel Corporation" in cands
    assert all("Advanced Micro" not in c for c in cands)


def test_match_candidate_handles_suffix_drift():
    cands = ["NVIDIA Corporation", "Intel Corporation", "Bolt"]
    assert match_candidate("NVIDIA Corp", cands) == "NVIDIA Corporation"  # label -> surface
    assert match_candidate("intel corporation", cands) == "Intel Corporation"
    assert match_candidate("Bolt Inc", cands) == "Bolt"
    assert match_candidate("Broadcom", cands) is None
    assert normalize_name("NVIDIA Corp") == normalize_name("NVIDIA Corporation")


def test_targetless_grammar_collapses_to_none_and_sentinels():
    class FakeTokenizer:
        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            return [ord(c) for c in text]

    grammar = build_triple_grammar(
        RELATIONSHIP_TYPES, [], FakeTokenizer(), eos_id=3, sentinels=("ESCALATE",)
    )
    assert grammar.targets == ()

    # Greedy decode can only reach NONE or the sentinel — relations are not offered.
    def prefer_first(generated, allowed):
        return [1.0 if i == 0 else 0.0 for i in range(len(allowed))]

    result = decode_triples(prefer_first, grammar)
    assert result.triples == ()
    assert result.sentinel in (None, "ESCALATE")


def test_candidates_cap_and_empty_text():
    text = " ".join(f"Company{i} Corp announces." for i in range(100))
    cands = chunk_target_candidates(text, max_candidates=10)
    assert len(cands) == 10
    assert chunk_target_candidates("no capitals here at all.") == []
