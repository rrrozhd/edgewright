#!/usr/bin/env bash
# Mine trajectories (if needed) and SFT the controller. Requires .[ml] (or .[colab]).
#
# Usage: bash scripts/run_sft.sh [dataset] [model] [split]
#   defaults: webqsp qwen3-0.6b train
# Smoke test (CPU, tiny model, offline sample):
#   bash scripts/run_sft.sh sample tiny-test dev
set -euo pipefail
cd "$(dirname "$0")/.."

DATASET="${1:-webqsp}"
MODEL="${2:-qwen3-0.6b}"
SPLIT="${3:-train}"
MINED="outputs/sft/${DATASET}_${SPLIT}.jsonl"

if [ ! -f "$MINED" ]; then
  echo ">>> mining trajectories: ${DATASET}/${SPLIT}"
  python -m kgat.train.mine_trajectories dataset="${DATASET}" dataset.split="${SPLIT}"
fi

echo ">>> SFT: model=${MODEL} data=${MINED}"
python -m kgat.train.sft train=sft dataset="${DATASET}" dataset.split="${SPLIT}" model="${MODEL}"

echo ">>> done. Evaluate with:"
echo "    python -m kgat.eval.harness dataset=${DATASET} model=${MODEL} controller=decoder \\"
echo "        controller.adapter_path=outputs/adapters/${MODEL}-sft"
