#!/usr/bin/env bash
# GRPO-train the controller with the cost-penalized reward. Requires .[ml]/.[colab].
# Warm-starts from the SFT adapter when present (run scripts/run_sft.sh first).
#
# Usage: bash scripts/run_grpo.sh [dataset] [model] [lam]
#   defaults: webqsp qwen3-0.6b 0.1
# Full lambda sweep for the frontier:
#   python -m kgat.train.grpo -m train.grpo.lam=0.0,0.05,0.1,0.2,0.4 dataset=webqsp
set -euo pipefail
cd "$(dirname "$0")/.."

DATASET="${1:-webqsp}"
MODEL="${2:-qwen3-0.6b}"
LAM="${3:-0.1}"

echo ">>> GRPO: model=${MODEL} dataset=${DATASET} lam=${LAM}"
python -m kgat.train.grpo dataset="${DATASET}" dataset.split=train model="${MODEL}" \
    train.grpo.lam="${LAM}"

echo ">>> done. Evaluate the RL'd adapter with:"
echo "    python -m kgat.eval.harness dataset=${DATASET} model=${MODEL} controller=decoder \\"
echo "        controller.adapter_path=outputs/adapters/${MODEL}-grpo-lam${LAM}"
