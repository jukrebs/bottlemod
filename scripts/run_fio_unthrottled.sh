#!/usr/bin/env bash
set -euo pipefail

# Unthrottled buffered run (more noise, but quick to compare).
#
# Usage:
#   bash scripts/run_fio_unthrottled.sh

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

ROOT="$(repo_root)"
activate_venv_if_present

OUT_DIR="${OUT_DIR:-/var/tmp/bottlemod_exp_unthrottled}"
TRIALS="${TRIALS:-3}"
SIZE="${SIZE:-2G}"

echo "Unthrottled: OUT_DIR=$OUT_DIR SIZE=$SIZE TRIALS=$TRIALS"
echo "Set FORCE=1 to skip confirmation."
confirm_or_exit "Proceed with unthrottled fio run (OS disk)?"

python "$ROOT/fio_runner.py" \
  --out-dir "$OUT_DIR" \
  --size "$SIZE" \
  --trials "$TRIALS"

python "$ROOT/fio_parse_and_model.py" \
  --in-dir "$OUT_DIR" \
  --out "$ROOT/experiment_ground_truth_fio.json"

python "$ROOT/experiment_plots.py"

echo "Done. Plots: $ROOT/experiment_plots/*.png"
