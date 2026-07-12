#!/usr/bin/env bash
# Reproduce RoG / GCR / GNN-RAG baselines through our harness (M2).
#
# STATUS: the baseline wrappers in src/kgat/baselines/ are stubs. Before this script
# does anything real you must, at M2:
#   1. Locate + VERIFY the official RoG and GNN-RAG repos (do not hardcode unverified
#      URLs). GCR official repo (user-provided): RManLuo/graph-constrained-reasoning.
#   2. Pull each baseline's published WebQSP/CWQ Hit/F1 numbers from the paper/repo and
#      record the reproduction tolerance — do NOT invent target numbers.
#   3. Implement RoGBaseline/GCRBaseline/GNNRAGBaseline.predict and score via the harness.
set -euo pipefail
cd "$(dirname "$0")/.."

cat <<'MSG'
[reproduce_baselines] Baseline wrappers are stubbed (M2).
  - src/kgat/baselines/rog.py       (RoG   — locate/verify official repo + numbers)
  - src/kgat/baselines/gcr.py       (GCR   — RManLuo/graph-constrained-reasoning; verify)
  - src/kgat/baselines/gnn_rag.py   (GNN-RAG — locate/verify official repo + numbers)
Implement these and re-run to score them through kgat.eval.harness.
MSG
exit 0
