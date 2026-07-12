#!/usr/bin/env bash
# Download a preprocessed KGQA dataset into ${KGAT_DATA_DIR}/<dataset> as <split>.jsonl.
# Requires the .[data] extra (HuggingFace datasets). See kgat.data.subgraph for the
# on-disk schema (assumed to match the rmanluo/RoG-* releases; verify at M2).
#
# Usage: bash scripts/download_data.sh <webqsp|cwq|metaqa>
set -euo pipefail
cd "$(dirname "$0")/.."

DATASET="${1:?usage: download_data.sh <webqsp|cwq|metaqa>}"
DATA_ROOT="${KGAT_DATA_DIR:-data}"
OUT_DIR="${DATA_ROOT}/${DATASET}"

echo ">>> downloading ${DATASET} into ${OUT_DIR}"
python - "$DATASET" "$OUT_DIR" <<'PY'
import sys
from kgat.data.loaders import download_dataset
dataset, out_dir = sys.argv[1], sys.argv[2]
download_dataset(dataset, out_dir)
print(f"wrote splits to {out_dir}")
PY
