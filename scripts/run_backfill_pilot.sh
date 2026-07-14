#!/usr/bin/env bash
# Backfill extractor pilot (DESIGN-BACKFILL.md): synthetic pairs -> extractor SFT
# -> cascade threshold sweep -> frontier. Requires .[ml] (or .[colab]).
#
# Usage: bash scripts/run_backfill_pilot.sh [model] [filings] [seed]
#   defaults: qwen3-0.6b 150 42
# Smoke test (CPU/MPS, tiny model, minutes):
#   bash scripts/run_backfill_pilot.sh tiny-test 20
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${1:-qwen3-0.6b}"
FILINGS="${2:-150}"
SEED="${3:-42}"
DATA_DIR="data/backfill/synthetic"
ADAPTER="outputs/adapters/${MODEL}-extractor"
OUT_DIR="outputs/backfill/cascade-${MODEL}"

if [ ! -f "${DATA_DIR}/vocab.json" ]; then
  echo ">>> generating synthetic backfill pairs (${FILINGS} filings)"
  python -m kgat.data.backfill_export --out "${DATA_DIR}" --filings "${FILINGS}" --seed "${SEED}"
fi

HF_ID="$(python -c "import yaml; print(yaml.safe_load(open('configs/model/${MODEL}.yaml'))['hf_id'])")"

echo ">>> extractor SFT: model=${MODEL} data=${DATA_DIR}"
python -m kgat.train.sft_extractor train=sft_extractor model="${MODEL}" \
    train.sft_extractor.data_dir="${DATA_DIR}"

echo ">>> cascade frontier: held-out test filings"
python -m kgat.eval.extractor_cascade --model-id "${HF_ID}" --adapter "${ADAPTER}" \
    --data-dir "${DATA_DIR}" --split test --out-dir "${OUT_DIR}"

echo ">>> artifacts in ${OUT_DIR} (frontier.csv / frontier.png / outcomes.jsonl)"
