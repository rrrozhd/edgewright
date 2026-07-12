#!/usr/bin/env bash
# Build a (trivial, dummy-controller) cost/quality frontier — the M1 acceptance.
# Sweeps the DummyController's hop budget; deeper search costs more and (on the sample)
# answers more questions, tracing an upward accuracy-vs-cost curve. Runs fully offline.
#
# Usage: bash scripts/eval_frontier.sh [dataset]   (default: sample)
set -euo pipefail
cd "$(dirname "$0")/.."

DATASET="${1:-sample}"
RUNS_DIR="outputs/frontier"
rm -rf "$RUNS_DIR"

for HOPS in 0 1 2 3; do
  echo ">>> run: ${DATASET} dummy max_hops=${HOPS}"
  python -m kgat.eval.harness \
    dataset="${DATASET}" \
    controller=dummy synth=dummy \
    controller.max_hops="${HOPS}" \
    run_label="${DATASET}_dummy_hops${HOPS}" \
    paths.output_dir="${RUNS_DIR}"
done

echo ">>> building frontier"
python -m kgat.eval.frontier --runs-dir "${RUNS_DIR}" --out-dir "${RUNS_DIR}" --accuracy-metric hit --cost-axis llm_calls

echo ">>> done. artifacts in ${RUNS_DIR}/ (frontier.csv, frontier.png, *.summary.json)"
