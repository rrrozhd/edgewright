# kgat — Budget-Adaptive, Governed KG Traversal

A cost-adaptive controller for multi-hop knowledge-graph question answering (KGQA).
A small model (~0.6B) acts as a **traversal policy**: given a question and the
current frontier of a reasoning path, it picks the next relation to expand or emits
`STOP`. A separate, swappable **synthesizer** turns the retrieved paths into answers.
The whole traversal is wrapped in a **governance layer** that enforces per-hop policy
and emits an audit certificate.

The contribution is not "a small model can do KGQA" — it is:

1. **Budget-adaptive stopping** — the controller learns to go deep only when needed,
   traced as a cost/quality frontier via a cost-penalized RL reward.
2. **Governed, auditable traversal** as a first-class property.

Model size is a *swept variable*, not a fixed choice.

---

## Status

The foundation (M0–M1) **and the training pipeline (M3–M5)** are implemented. The
whole chain — mine → SFT → constrained-decode eval → GRPO — has been validated
end-to-end on CPU/MPS with a tiny model (`model=tiny-test`); real-model training
targets a GPU (see `notebooks/colab_kgat.ipynb`).

| Milestone | What | State |
|-----------|------|-------|
| M0 | Skeleton, schemas, ABCs, `DummyController`, pytest green | ✅ implemented |
| M1 | Data + eval foundation (metrics, cost, frontier, harness) | ✅ implemented |
| M2 | Baseline reproduction harness (RoG / GCR / GNN-RAG wrappers) | 🟡 stubbed (needs verified official repos + published numbers) |
| M3 | Trajectory mining (BFS oracle → engine replay → SFT JSONL) | ✅ implemented + tested |
| M4 | Decoder controller + trie-constrained decoding + LoRA/QLoRA SFT | ✅ implemented; CPU/MPS-validated, GPU run pending |
| M5 | Trajectory-level GRPO + λ frontier | ✅ implemented; smoke-tested on MPS, **not yet GPU-validated** |
| M6 | Size sweep + cross-encoder floor | ⏳ stub (`crossencoder_policy`) |
| M7 | Arch B / Arch C arms | ⏳ stub (`gnn_proposer`; dynamic trie shares `constrained_decoding`) |
| M8 | Governance layer + audit + overhead measurement | 🟡 policies + audit implemented & wired; overhead study pending |
| M9 | Ablations, transfer KG, write-up | ⏳ future |

---

## Install

Requires **Python 3.11+**. The foundation installs with *no model dependencies*.

```bash
# with uv (recommended)
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"

# or with pip
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Extras: `.[ml]` (torch/transformers/peft/trl — the model arm), `.[gnn]`
(torch-geometric), `.[data]` (HuggingFace `datasets` for real dataset downloads),
`.[wandb]`, `.[neo4j]`.

## Quickstart

```bash
pytest -q                       # foundation tests — green with zero model deps

# End-to-end dummy run on the bundled sample dataset (offline, no GPU):
python -m kgat.eval.harness controller=dummy synth=dummy dataset=sample

# Trivial cost/quality frontier on the sample:
bash scripts/eval_frontier.sh
```

### Training workflow (M3 → M5)

```bash
# 1. Mine oracle trajectories (CPU-only)
python -m kgat.train.mine_trajectories dataset=webqsp dataset.split=train

# 2. SFT the controller (QLoRA on CUDA; LoRA on MPS/CPU)
python -m kgat.train.sft train=sft dataset=webqsp dataset.split=train model=qwen3-0.6b

# 3. Evaluate the trained controller (trie-constrained decoding)
python -m kgat.eval.harness dataset=webqsp model=qwen3-0.6b controller=decoder \
    controller.adapter_path=outputs/adapters/qwen3-0.6b-sft

# 4. GRPO + the lambda frontier sweep
python -m kgat.train.grpo -m dataset=webqsp dataset.split=train model=qwen3-0.6b \
    train.grpo.lam=0.0,0.05,0.1,0.2,0.4
```

The full chain smoke-tests anywhere (CPU/MPS, ~1 min, tiny random model):

```bash
bash scripts/run_sft.sh sample tiny-test dev
```

On Colab, open `notebooks/colab_kgat.ipynb` — T4 covers mining/SFT/eval; use an
A100 for the full GRPO sweep. **Note:** GRPO is implemented but not yet validated
on a real GPU run; smoke-test (`train.grpo.max_questions=32`) before a long sweep.

Multirun sweeps (the reason we use Hydra):

```bash
python -m kgat.eval.harness -m model=qwen2.5-0.5b,qwen2.5-1.5b,qwen2.5-3b dataset=webqsp controller=dummy
python -m kgat.eval.harness -m train.grpo.lam=0.0,0.05,0.1,0.2,0.4 dataset=webqsp controller=dummy
```

## Datasets

We reuse the **preprocessed per-question subgraph** format released with the
baselines rather than rebuilding Freebase. Expected on-disk schema (one JSON object
per line, `*.jsonl`):

```json
{
  "id": "WebQTrn-0",
  "question": "what is the name of justin bieber brother",
  "q_entity": ["m.06w2sn5"],
  "a_entity": ["m.0gxnnwc"],
  "graph": [["m.06w2sn5", "people.person.sibling_s", "m.0gxnnwc"], ...]
}
```

> **Assumption to verify at M2:** this matches the `rmanluo/RoG-webqsp` /
> `rmanluo/RoG-cwq` HuggingFace release. `data/loaders.py` documents this and the
> real download path; the bundled `data/sample/*.jsonl` follows the same schema so
> tests and the offline smoke run need no network. Confirm the field names against
> the actual release before running M2.

## Repository layout

```
configs/          Hydra config groups (model / dataset / train / experiment / controller / synth)
src/kgat/
  data/           schemas (THE contract), loaders, subgraph normalization
  graph/          KGStore ABC + in-memory impl (+ Neo4j adapter stub)
  controller/     TraversalController ABC + DummyController (+ neural policy stubs)
  synthesis/      AnswerSynthesizer ABC + DummySynthesizer (+ path_reader stub)
  governance/     HopPolicy ABC + AuditCertificate datamodel (concrete policies stubbed)
  traversal/      engine (main loop) + budget ledger  — IMPLEMENTED
  train/          reward fn (IMPLEMENTED) + mining/sft/grpo (stubs)
  eval/           metrics, cost, frontier, harness      — IMPLEMENTED
  baselines/      RoG / GCR / GNN-RAG wrappers           — stubs
  utils/          JSONL + optional W&B logging, seeding
scripts/          download_data / reproduce_baselines / run_sft / run_grpo / eval_frontier / sweep
tests/            pytest suite (schemas, graph, engine, reward, metrics, cost)
```

## Design notes (deviations from the brief, and why)

- **`Path.root`** — the brief's `Path` holds only `triples` and `current_node`
  raises when empty, yet the frontier is seeded from `question.topic_entities`. An
  unexpanded path has no triples, so it cannot report a position. We add an optional
  `root: Entity | None` to anchor the starting topic entity. `current_node` returns
  the tail of the last triple, else `root`, and still **raises for a truly empty
  path** (no triples *and* no root) — honoring the literal contract.
- **`controller` and `synth` config groups** — not in the brief's config tree, but
  `controller=dummy` appears in its example command. We add both so every swappable
  piece is a config choice (`hydra.utils.instantiate` via `_target_`).
- **`bind_store` hook** on `TraversalController` — a default no-op the engine calls
  once per question so a controller *may* consult the store (the `DummyController`'s
  highest-degree heuristic needs it). Neural controllers ignore it; `select()`'s
  signature stays `(state, candidates)`.
- **`dataset=sample`** — a tiny bundled dataset (same schema as the real releases)
  so `pytest` and `scripts/eval_frontier.sh` run fully offline. Real WebQSP/CWQ/MetaQA
  configs point at downloaded data via `scripts/download_data.sh`.

## License

Apache-2.0.
