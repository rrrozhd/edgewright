#!/usr/bin/env bash
# Model-size sweep (M6): evaluate the controller across the size ladder + the
# cross-encoder floor, then build one frontier across sizes.
#
# STATUS: real controllers arrive M4+. This script shows the intended Hydra multirun
# and runs the harness with controller=dummy so the sweep plumbing is exercisable now.
#
# Usage: bash scripts/sweep_model_size.sh [dataset]   (default: sample)
set -euo pipefail
cd "$(dirname "$0")/.."

DATASET="${1:-sample}"
RUNS_DIR="outputs/size_sweep"
rm -rf "$RUNS_DIR"

# Intended (M6): sweep real models with the decoder/crossencoder controllers:
#   python -m kgat.eval.harness -m \
#     model=qwen2.5-0.5b,qwen3-0.6b,qwen2.5-1.5b,qwen2.5-3b \
#     controller=decoder dataset="${DATASET}"
#   python -m kgat.eval.harness model=crossencoder-modernbert controller=crossencoder dataset="${DATASET}"

# Runnable now (dummy) — exercises the multirun + frontier plumbing offline:
python -m kgat.eval.harness -m \
  model=qwen2.5-0.5b,qwen3-0.6b,qwen2.5-1.5b,qwen2.5-3b \
  dataset="${DATASET}" controller=dummy synth=dummy \
  paths.output_dir="${RUNS_DIR}"

python -m kgat.eval.frontier --runs-dir "${RUNS_DIR}" --out-dir "${RUNS_DIR}"
echo ">>> size-sweep artifacts in ${RUNS_DIR}/"
