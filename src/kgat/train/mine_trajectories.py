"""Trajectory mining (M3): oracle traversals -> SFT dataset.

For every question we BFS the per-question subgraph for a shortest relation path
from the topic entities to a gold answer, replay that script through the real
``TraversalEngine`` (so recorded states match inference exactly), keep only
traversals that actually hit, and emit one SFT example per controller decision:

    {"qid", "dataset", "step", "prompt", "target", "n_candidates"}

``prompt`` is ``format_prompt(state_repr, candidates)``; ``target`` is the chosen
relation or ``[STOP]``. The STOP decisions are as important as the EXPANDs — they
are what teaches budget-adaptive stopping.

Runs offline, no model deps. CLI (Hydra)::

    python -m kgat.train.mine_trajectories dataset=webqsp dataset.split=train
    python -m kgat.train.mine_trajectories dataset=sample   # offline smoke
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path as FsPath

from kgat.controller.constrained_decoding import STOP_TOKEN
from kgat.controller.oracle import OracleController
from kgat.controller.prompting import format_prompt
from kgat.data.schemas import ActionType, Entity, Relation, Trajectory, Triple
from kgat.data.subgraph import SubgraphRecord
from kgat.graph.inmemory import InMemoryKGStore
from kgat.synthesis.base import DummySynthesizer
from kgat.traversal.engine import TraversalEngine


def bfs_relation_script(
    triples: tuple[Triple, ...],
    topics: tuple[Entity, ...],
    golds: tuple[Entity, ...],
    max_hops: int,
) -> tuple[Relation, ...] | None:
    """Shortest relation sequence from any topic entity to any gold answer.

    Multi-source BFS with deterministic tie-breaking (edges visited in stored
    order). Returns ``None`` if no gold is reachable within ``max_hops``. A gold
    that IS a topic entity yields the empty script (pure-STOP example).
    """
    gold_set = set(golds)
    if not gold_set or not topics:
        return None
    if any(t in gold_set for t in topics):
        return ()

    adj: dict[Entity, list[tuple[Relation, Entity]]] = {}
    for t in triples:
        adj.setdefault(t.head, []).append((t.relation, t.tail))

    # parent[node] = (prev_node, relation); topics are roots.
    parent: dict[Entity, tuple[Entity, Relation] | None] = {t: None for t in topics}
    queue: deque[tuple[Entity, int]] = deque((t, 0) for t in topics)
    while queue:
        node, depth = queue.popleft()
        if depth >= max_hops:
            continue
        for relation, tail in adj.get(node, ()):
            if tail in parent:
                continue
            parent[tail] = (node, relation)
            if tail in gold_set:
                script: list[Relation] = [relation]
                cur = node
                while parent[cur] is not None:
                    prev, rel = parent[cur]  # type: ignore[misc]
                    script.append(rel)
                    cur = prev
                return tuple(reversed(script))
            queue.append((tail, depth + 1))
    return None


@dataclass
class MiningStats:
    n_questions: int = 0
    n_unreachable: int = 0
    n_replay_missed: int = 0  # script found but replay did not hit (e.g. beam pruning)
    n_mined: int = 0
    n_examples: int = 0
    depth_histogram: dict[int, int] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "n_questions": self.n_questions,
            "n_unreachable": self.n_unreachable,
            "n_replay_missed": self.n_replay_missed,
            "n_mined": self.n_mined,
            "n_examples": self.n_examples,
            "depth_histogram": dict(sorted(self.depth_histogram.items())),
        }


def mine_dataset(
    records: list[SubgraphRecord],
    *,
    beam_size: int = 64,
    max_hops: int = 4,
) -> tuple[list[Trajectory], MiningStats]:
    """Mine oracle trajectories for every reachable question in ``records``."""
    store = InMemoryKGStore.from_records(records)
    synth = DummySynthesizer()
    stats = MiningStats()
    mined: list[Trajectory] = []

    for rec in records:
        stats.n_questions += 1
        q = rec.question
        script = bfs_relation_script(rec.triples, q.topic_entities, q.gold_answers, max_hops)
        if script is None:
            stats.n_unreachable += 1
            continue

        engine = TraversalEngine(
            store,
            OracleController(script),
            synth,
            beam_size=beam_size,
            max_steps=max_hops + 1,
        )
        result = engine.run(q)
        traj = result.trajectory
        if not traj.hit:
            stats.n_replay_missed += 1
            continue

        mined.append(traj)
        stats.n_mined += 1
        stats.n_examples += len(traj.steps)
        depth = sum(1 for s in traj.steps if s.action.type is ActionType.EXPAND)
        stats.depth_histogram[depth] = stats.depth_histogram.get(depth, 0) + 1

    return mined, stats


def trajectory_to_sft_examples(traj: Trajectory, dataset: str) -> list[dict]:
    """One SFT example per recorded controller decision."""
    examples: list[dict] = []
    for i, step in enumerate(traj.steps):
        target = step.action.relation if step.action.type is ActionType.EXPAND else STOP_TOKEN
        examples.append(
            {
                "qid": traj.qid,
                "dataset": dataset,
                "step": i,
                "prompt": format_prompt(step.state_repr, step.candidates),
                "target": target,
                "n_candidates": len(step.candidates),
            }
        )
    return examples


def write_sft_jsonl(trajectories: list[Trajectory], dataset: str, out_path: str | FsPath) -> int:
    """Write all trajectories' SFT examples as JSONL. Returns the example count."""
    out_path = FsPath(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for traj in trajectories:
            for ex in trajectory_to_sft_examples(traj, dataset):
                fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
                n += 1
    return n


def _main() -> None:
    import hydra
    from omegaconf import DictConfig

    from kgat.data.loaders import load_records
    from kgat.utils.paths import resolve_path
    from kgat.utils.seed import set_seed

    @hydra.main(version_base=None, config_path="../../../configs", config_name="config")
    def main(cfg: DictConfig) -> None:
        set_seed(int(cfg.seed))
        records = load_records(
            resolve_path(cfg.dataset.data_dir),
            split=cfg.dataset.split,
            dataset=cfg.dataset.name,
            limit=cfg.dataset.get("limit"),
        )
        mined, stats = mine_dataset(
            records,
            beam_size=int(cfg.mine.beam_size),
            max_hops=int(cfg.mine.max_hops),
        )
        out_path = resolve_path(cfg.mine.out_path)
        n = write_sft_jsonl(mined, cfg.dataset.name, out_path)
        print(json.dumps(stats.as_dict(), indent=2))
        print(f"wrote {n} SFT examples -> {out_path}")

    main()


if __name__ == "__main__":
    _main()


__all__ = [
    "bfs_relation_script",
    "mine_dataset",
    "trajectory_to_sft_examples",
    "write_sft_jsonl",
    "MiningStats",
]
