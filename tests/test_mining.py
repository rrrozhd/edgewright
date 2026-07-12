"""Tests for trajectory mining (M3) on the bundled sample dataset."""

from __future__ import annotations

import json
from pathlib import Path

from kgat.controller.constrained_decoding import STOP_TOKEN
from kgat.data.loaders import load_records
from kgat.data.schemas import ActionType, Triple
from kgat.train.mine_trajectories import (
    bfs_relation_script,
    mine_dataset,
    trajectory_to_sft_examples,
    write_sft_jsonl,
)

SAMPLE_DIR = Path(__file__).parent.parent / "data" / "sample"


def _sample_records():
    return load_records(SAMPLE_DIR, split="dev", dataset="sample")


def test_bfs_finds_shortest_script():
    triples = (
        Triple("a", "r1", "b"),
        Triple("b", "r2", "c"),
        Triple("a", "r_direct", "c"),  # 1-hop shortcut must win over the 2-hop path
    )
    assert bfs_relation_script(triples, ("a",), ("c",), max_hops=4) == ("r_direct",)


def test_bfs_unreachable_and_depth_cap():
    triples = (Triple("a", "r1", "b"), Triple("b", "r2", "c"))
    assert bfs_relation_script(triples, ("a",), ("zzz",), max_hops=4) is None
    assert bfs_relation_script(triples, ("a",), ("c",), max_hops=1) is None  # needs 2 hops


def test_bfs_topic_is_gold_gives_empty_script():
    triples = (Triple("a", "r1", "b"),)
    assert bfs_relation_script(triples, ("a",), ("a",), max_hops=4) == ()


def test_mine_sample_dataset_all_hit():
    mined, stats = mine_dataset(_sample_records(), beam_size=16, max_hops=4)
    assert stats.n_questions == 5
    assert stats.n_mined == 5  # every sample question is oracle-reachable
    assert stats.n_unreachable == 0
    assert all(t.hit for t in mined)
    # Depths match the dataset design: two 1-hop, two needing 1 hop... histogram
    # keys are expansion counts: sample-1/3/4 -> 1 hop, sample-2 -> 2, sample-5 -> 3.
    assert stats.depth_histogram == {1: 3, 2: 1, 3: 1}


def test_mined_trajectory_teaches_adaptive_stop():
    mined, _ = mine_dataset(_sample_records(), beam_size=16, max_hops=4)
    by_qid = {t.qid: t for t in mined}

    # sample-4: answer one hop away; the oracle expands once then STOPs — the
    # "don't overshoot" training signal.
    shallow = by_qid["sample-4"]
    kinds = [s.action.type for s in shallow.steps]
    assert kinds == [ActionType.EXPAND, ActionType.STOP]
    assert shallow.steps[0].action.relation == "seq.next"

    # sample-5: three hops deep; the oracle goes all the way — the "go deep when
    # needed" signal. Its answer is a dead-end node, so the traversal ends on the
    # graph running out (no explicit STOP decision — the controller is never asked
    # a question it wouldn't be asked at inference).
    deep = by_qid["sample-5"]
    relations = [s.action.relation for s in deep.steps if s.action.type is ActionType.EXPAND]
    assert relations == ["gen.child", "gen.child", "gen.child"]
    assert all(s.action.type is ActionType.EXPAND for s in deep.steps)


def test_sft_examples_shape_and_roundtrip(tmp_path):
    mined, _ = mine_dataset(_sample_records(), beam_size=16, max_hops=4)
    all_examples = [ex for t in mined for ex in trajectory_to_sft_examples(t, dataset="sample")]
    for ex in all_examples:
        assert ex["prompt"].endswith("next:")
        assert "candidates:" in ex["prompt"]
        assert f"- {STOP_TOKEN}" in ex["prompt"]
        assert ex["target"]  # relation string or STOP_TOKEN
    # STOP examples exist exactly where a real STOP decision exists: sample-4's
    # answer node has an overshooting out-edge, so its trajectory ends in [STOP].
    stop_targets = [ex for ex in all_examples if ex["target"] == STOP_TOKEN]
    assert len(stop_targets) == 1
    assert stop_targets[0]["qid"] == "sample-4"

    out = tmp_path / "sft.jsonl"
    n = write_sft_jsonl(mined, "sample", out)
    lines = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(lines) == n
    assert all({"qid", "prompt", "target", "step"} <= set(rec) for rec in lines)
